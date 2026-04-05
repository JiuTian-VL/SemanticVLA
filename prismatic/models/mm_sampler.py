import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForTextEncoding


def get_mlp(in_dim, hidden_dim, out_dim, zero_init=False):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim)
    )
    if zero_init:
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
    return mlp


def softmax_with_temperature(logits, temperature=1.0, dim=-1):
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=dim)


class TextGuidedSampler(nn.Module):
    def __init__(self, config, vision_embed_dim):
        super().__init__()
        self.temp = config.get("temp", 1.0)
        self.vision_topk = config.get("vision_topk", 32)
        self.text_topk = config.get("text_topk", 5)

        text_encoder = config.get("text_encoder_path", None)
        if text_encoder is not None:
            from transformers import AutoTokenizer, SiglipTextModel, CLIPTextModel
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

            if 'siglip' in text_encoder:
                self.text_encoder = SiglipTextModel.from_pretrained(text_encoder)
                self.tokenizer_padding = "max_length"
            elif 'clip' in text_encoder:
                self.text_encoder = CLIPTextModel.from_pretrained(text_encoder)
                self.tokenizer_padding = True
            else:
                raise NotImplementedError

            text_embed_dim = self.text_encoder.config.hidden_size
            # self.text_post_projector = nn.Linear(text_embed_dim, vision_embed_dim)
            self.text_post_projector = get_mlp(text_embed_dim, vision_embed_dim, vision_embed_dim)

    def get_similarity(self, image_embeddings, text_embeddings, attention_mask=None):
        # image: [bs, N, D], text_embedding: [bs, M, D]
        normed_img = F.normalize(image_embeddings, p=2, dim=-1)
        normed_txt = F.normalize(text_embeddings, p=2, dim=-1)
        similarity_matrix = torch.bmm(normed_img, normed_txt.transpose(1, 2))  # [B, N, M]

        # Add gumbel noise
        if self.training:
            gumbel_noise = torch.randn_like(similarity_matrix) * 0.1
            similarity_matrix = (similarity_matrix + gumbel_noise)

        # The minimum value of cos similarity is -1.
        # Note that using -inf may cause average operation overflow under bf16,
        # so we use -1 for masking.
        similarity_matrix = similarity_matrix.masked_fill(~attention_mask.unsqueeze(1), -1)
        return similarity_matrix

    def get_vision_topk(self, sim, vision_embedding):
        # average across text
        probs = torch.mean(sim, dim=2)  # [B, N]

        # Sort the probabilities in descending order and get the corresponding indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        selected_indices = sorted_indices[:, :self.vision_topk]
        selected_indices, _ = selected_indices.sort(descending=False, dim=-1)

        expanded_idx = selected_indices.unsqueeze(-1).expand(-1, -1, vision_embedding.shape[-1])
        outs = torch.gather(vision_embedding, 1, expanded_idx)  # [B, Kv, D]
        return outs

    def get_text_topk(self, sim, vision_embedding, text_embedding):
        probs = sim.transpose(1, 2)  # [B, M, N]
        probs = softmax_with_temperature(probs, temperature=self.temp)  # [B, M, N]
        weigted_vision = torch.bmm(probs, vision_embedding)  # [B, M, D]

        # average across vision
        probs = torch.mean(sim, dim=1)  # [B, M]

        # Sort the probabilities in descending order and get the corresponding indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        selected_indices = sorted_indices[:, :self.text_topk]
        selected_indices, _ = selected_indices.sort(descending=False, dim=-1)

        expanded_idx = selected_indices.unsqueeze(-1).expand(-1, -1, vision_embedding.shape[-1])
        outs = torch.gather(weigted_vision, 1, expanded_idx)  # [B, Kt, D]
        return outs

    def forward(self, vision_embedding, text_ids=None, text_embedding=None, attention_mask=None):
        if (text_ids is None) ^ (text_embedding is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if text_embedding is None:
            if isinstance(text_ids, str) or (isinstance(text_ids, list) and isinstance(text_ids[0], str)):
                # important: make sure to set padding="max_length" for SiglipTextModel as that's how the model was trained
                text_ids = self.tokenizer(text_ids, padding=self.tokenizer_padding, return_tensors="pt").input_ids.to(self.text_encoder.device)
            attention_mask = text_ids.ne(self.tokenizer.pad_token_id)
            text_embedding = self.text_encoder(text_ids, attention_mask=attention_mask).last_hidden_state

            # min_pad_len = (~attention_mask).sum(dim=1).min()
            # if min_pad_len != 0:
            #     text_embedding = text_embedding[:, :-min_pad_len, :]
            #     attention_mask = attention_mask[:, :-min_pad_len]

            text_embedding = self.text_post_projector(text_embedding)

        similarity_matrix = self.get_similarity(vision_embedding, text_embedding, attention_mask)  # [B, N, M]

        # # Add gumbel noise
        # if self.training:
        #     gumbel_noise = torch.randn_like(similarity_matrix) * 0.1
        #     similarity_matrix = (similarity_matrix + gumbel_noise)

        vision_topk = self.get_vision_topk(similarity_matrix, vision_embedding)  # [B, Kv, D]
        if self.text_topk > 0:
            text_topk = self.get_text_topk(similarity_matrix, vision_embedding, text_embedding)  # [B, Kt, D]
            outs = torch.cat([text_topk, vision_topk], dim=1)
        else:
            outs = vision_topk

        return outs


if __name__ == '__main__':
    from transformers import PretrainedConfig
    cfg = PretrainedConfig()
    router = TextGuidedSampler(cfg)
    router.eval()

    x = torch.randn((3, 512, 4096), dtype=torch.float)
    text_embedding = torch.randn((3, 10, 4096), dtype=torch.float)
    x = router(x, text_embedding)
    print(x.shape)