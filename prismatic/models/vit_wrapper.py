"""Implementation of additional modules for the VLA's vision transformer."""

from icecream import ic
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Union, List

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from .film_vit_wrapper import FiLMedVisionTransformerBlock, NullVisionTransformerBlockWrapper, unpack_tuple
from .mm_sampler import TextGuidedSampler


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


# Renns dev
class CrossVisionTransformerInteractionWrapper(nn.Module):
    def __init__(self, vits=None, target_pairs=None, cross_vit_type='default'):
        super().__init__()

        self.vits = vits
        self.target_pairs = target_pairs  # layer idx starts from 0
        self.num_vits = len(self.vits)

        if self.target_pairs is None:
            self.target_pairs = [tuple([1] * self.num_vits), tuple([-2] * self.num_vits)]

        assert all(self.num_vits == len(p) for p in self.target_pairs)

        self.cross_vit_type = cross_vit_type
        if self.cross_vit_type == 'default' or self.cross_vit_type == None:
            total_dim = sum(vit.embed_dim for vit in self.vits)
            proj_list = []
            for _ in range(len(self.target_pairs)):
                for vit in self.vits:
                    proj_list.append(get_mlp(total_dim, total_dim, vit.embed_dim))
            self.aggregate_proj = nn.ModuleList(proj_list)
        else:
            raise NotImplementedError

    def cross_vit_interact(self, x, pair_id):
        """
        x: List[torch.Tensor]
        """
        if self.cross_vit_type == 'default' or self.cross_vit_type == None:
            patch_token_range = [
                (vit.num_prefix_tokens, vit.num_prefix_tokens + vit.patch_embed.grid_size[0] * vit.patch_embed.grid_size[1])
                for vit in self.vits
            ]
            x_cat = torch.cat([
                x_vit[:, r[0]: r[1], :]
                for r, x_vit in zip(patch_token_range, x)
            ], dim=-1)

            st = pair_id * self.num_vits
            ed = st + self.num_vits
            for vit_idx, proj in enumerate(self.aggregate_proj[st: ed]):
                r = patch_token_range[vit_idx]
                x[vit_idx][:, r[0]: r[1], :] += proj(x_cat)
        else:
            raise NotImplementedError

        return x

    def forward_blks_range(self, vit, x, language_embeddings, st, ed, take_indices):
        outputs = []
        for l in range(st, ed):
            x = vit.blocks[l](x, language_embeddings)
            if l in take_indices:
                outputs.append(x)

        return x, outputs

    def process_outputs(self, vit, outputs, bs, reshape=False, return_prefix_tokens=False, norm=False):
        if norm:
            outputs = [vit.norm(out) for out in outputs]

        prefix_tokens = [out[:, 0: vit.num_prefix_tokens] for out in outputs]
        outputs = [out[:, vit.num_prefix_tokens:] for out in outputs]

        if vit.get_vision_queries() is not None:
            # treat register as visual embedding
            outputs = [out[:, vit.patch_embed.grid_size[0] * vit.patch_embed.grid_size[1]:] for out in outputs]

        if reshape:
            grid_size = vit.patch_embed.grid_size
            outputs = [
                out.reshape(bs, grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward(
        self,
        x: List[torch.Tensor],
        language_embeddings: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        bs = x[0].shape[0]

        outputs = [[] for _ in range(self.num_vits)]
        num_blocks = [len(vit.blocks) for vit in self.vits]
        take_indices = [set(range(n_blk - n, n_blk) if isinstance(n, int) else n) for n_blk in num_blocks]

        # forward pass
        x = [vit.prepare_inputs(x_in, vit.get_vision_queries()) for vit, x_in in zip(self.vits, x)]
        start_layers = [0] * self.num_vits

        for pair_id, target_pair in enumerate(self.target_pairs):
            for vit_idx, vit in enumerate(self.vits):
                _x, _outs = self.forward_blks_range(
                    vit, x[vit_idx], language_embeddings, st=start_layers[vit_idx], ed=target_pair[vit_idx] + 1, take_indices=take_indices[vit_idx]
                )
                x[vit_idx] = _x
                outputs[vit_idx].extend(_outs)
                start_layers[vit_idx] = target_pair[vit_idx] + 1

            x = self.cross_vit_interact(x, pair_id)

        for vit_idx, vit in enumerate(self.vits):
            _x, _outs = self.forward_blks_range(
                vit, x[vit_idx], language_embeddings, st=start_layers[vit_idx], ed=len(vit.blocks), take_indices=take_indices[vit_idx]
            )
            x[vit_idx] = _x
            outputs[vit_idx].extend(_outs)

        outputs = [
            self.process_outputs(vit, outs, bs, reshape=reshape, return_prefix_tokens=return_prefix_tokens, norm=norm)
            for vit, outs in zip(self.vits, outputs)
        ]
        outputs = [
            outs[0] if isinstance(outs, tuple) else outs
            for outs in outputs
        ]
        return outputs



class VisionTransformerRegister(VisionTransformer):
    """
    Wrapper for timm.models.vision_transformer.VisionTransformer that overrides functions to enable infusing language
    embeddings into visual embeddings via FiLM.
    """

    def prepare_inputs(self, x, vision_queries):
        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if vision_queries is not None:
            vision_queries_batch = vision_queries.expand(x.shape[0], -1, -1)
            x = torch.cat([
                x, vision_queries_batch.to(x.dtype).to(x.device)
            ], dim=1)

        return x

    def _intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        vision_queries: torch.Tensor = None,
        n: Union[int, Sequence] = 1,
    ):
        """
        Copy of timm.models.vision_transformer.VisionTransformer._intermediate_layers() with modifications
        to take in language embeddings as additional input.
        """
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        x = self.prepare_inputs(x, vision_queries)

        for i, blk in enumerate(self.blocks):
            x = blk(x, language_embeddings)  # Modified to receive language_embeddings
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        language_embeddings: torch.Tensor,
        vision_queries: torch.Tensor = None,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        Copy of timm.models.vision_transformer.VisionTransformer.get_intermediate_layers() with modifications
        to allow language embeddings as additional input.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(
            x, language_embeddings=language_embeddings, vision_queries=vision_queries, n=n
        )
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if vision_queries is not None:
            # treat register as visual embedding
            outputs = [out[:, self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1]:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


class PrismaticHybridCompressionVisionBackbone(nn.Module):
    """
    Wraps the Vision Transformers in the vision backbone to enable hybrid compression and language conditioning through FiLM.
    Supports processing 1-3 images using dual vision backbones (SigLIP + DINOv2).
    """

    def __init__(
        self,
        vision_backbone,
        vla_cfg,
        llm_dim: int = 4096,  # 4096 for Llama-2 7B
    ) -> None:
        """
        Initializes FiLM wrapper.

        Args:
            vision_backbone (PrismaticVisionBackbone): Base vision backbone.
            llm_dim (int): Dimension of language model embeddings.
        """
        super().__init__()
        self.vla_cfg = vla_cfg
        self.vision_backbone = vision_backbone
        self.llm_dim = llm_dim

        self.featurizer_use_reg = vla_cfg.featurizer_cfg.get("use_reg", False)
        self.fused_featurizer_use_reg = vla_cfg.fused_featurizer_cfg.get("use_reg", False)

        self.featurizer_use_mm_sampler = vla_cfg.featurizer_cfg.get("use_mm_sampler", False)
        self.fused_featurizer_use_mm_sampler = vla_cfg.fused_featurizer_cfg.get("use_mm_sampler", False)

        # =======================================
        self.vision_queries = None
        if self.featurizer_use_reg:
            self.featurizer_num_vision_queries = vla_cfg.featurizer_cfg.get("num_vision_queries", None)
            self.vision_queries = nn.Parameter(
                torch.randn(1, self.featurizer_num_vision_queries, self.vision_backbone.featurizer.embed_dim)
            )

        self._wrap_vit(
            self.vision_backbone.featurizer,
            use_film=vla_cfg.featurizer_cfg.get("use_film", False),
            vision_queries=self.vision_queries,
        )  # DINOv2

        if self.featurizer_use_mm_sampler:
            msg = (f"Initializing `TextGuidedSampler` for `{self.vision_backbone.featurizer.default_cfg['architecture']}`, "
                   f"`vision_topk={vla_cfg.featurizer_cfg['vision_topk']}`, `text_topk={vla_cfg.featurizer_cfg['text_topk']}`")
            ic(msg)
            self.featurizer_mm_sampler = TextGuidedSampler(
                vla_cfg.featurizer_cfg, vision_embed_dim=self.vision_backbone.featurizer.embed_dim
            )
        # =======================================

        # =======================================
        if self.vision_backbone.use_fused_vision_backbone:
            self.vision_queries_fused = None
            if self.fused_featurizer_use_reg:
                self.fused_featurizer_num_vision_queries = vla_cfg.fused_featurizer_cfg.get("num_vision_queries", None)
                self.vision_queries_fused = nn.Parameter(
                    torch.randn(1, self.fused_featurizer_num_vision_queries, self.vision_backbone.fused_featurizer.embed_dim)
                )

            self._wrap_vit(
                self.vision_backbone.fused_featurizer,
                use_film=vla_cfg.fused_featurizer_cfg.get("use_film", False),
                vision_queries=self.vision_queries_fused
            )  # SigLIP

            if self.fused_featurizer_use_mm_sampler:
                msg = (
                    f"Initializing `TextGuidedSampler` for `{self.vision_backbone.fused_featurizer.default_cfg['architecture']}`, "
                    f"`vision_topk={vla_cfg.fused_featurizer_cfg['vision_topk']}`, `text_topk={vla_cfg.fused_featurizer_cfg['text_topk']}`")
                ic(msg)
                self.fused_featurizer_mm_sampler = TextGuidedSampler(
                    vla_cfg.fused_featurizer_cfg, vision_embed_dim=self.vision_backbone.fused_featurizer.embed_dim
                )
        # =======================================

        self.enable_cross_vit_interact = getattr(vla_cfg, "enable_cross_vit_interact", False)
        if self.enable_cross_vit_interact:
            target_pairs = getattr(vla_cfg, "cross_vit_target_pairs", [(1, 1), (-1, -1)])
            cross_vit_type = getattr(vla_cfg, "cross_vit_type", 'default')

            msg = f"Initializing cross ViT interaction module in layer pairs {target_pairs}, interaction type: {cross_vit_type}"
            ic(msg)

            self.cross_vit_model = CrossVisionTransformerInteractionWrapper(
                vits=[self.vision_backbone.featurizer, self.vision_backbone.fused_featurizer],
                target_pairs=target_pairs,
                cross_vit_type=cross_vit_type,
            )

    def _wrap_vit(self, vit, use_film=False, vision_queries=None) -> None:
        """
        Args:
            vit (VisionTransformer): Original vision transformer.
        """
        msg = f"Wrapping `{vit.default_cfg['architecture']}` with: `FiLm={use_film}`, `registers={vision_queries.shape[1] if vision_queries is not None else 0}`"
        ic(msg)

        # Wrap vision transformer blocks
        block_wrappers = []
        for block in vit.blocks:
            block_wrappers.append(
                NullVisionTransformerBlockWrapper(block=block) if not use_film else
                FiLMedVisionTransformerBlock(block=block, vision_dim=vit.num_features, llm_dim=self.llm_dim)
            )
        vit.blocks = nn.Sequential(*block_wrappers)

        # Wrap vision transformer with new class that overrides functions used for forward pass
        vit.__class__ = VisionTransformerRegister
        vit.forward = unpack_tuple(partial(vit.get_intermediate_layers, vision_queries=vision_queries, n={len(vit.blocks) - 2}))
        vit.get_vision_queries = lambda :vision_queries

    # def get_num_patches(self) -> int:
    #     """Returns the number of vision patches output by the vision backbone."""
    #     return self.vision_backbone.get_num_patches()

    def get_num_patches_single(self, vit, vit_cfg) -> int:
        num_patches = None
        num_text_patches = 0
        if vit_cfg.get("use_mm_sampler", False):
            num_patches = vit_cfg.get("vision_topk")
            num_text_patches = vit_cfg.get("text_topk")
        elif vit_cfg.get("use_reg", False):
            num_patches = vit_cfg.get("num_vision_queries")
        else:
            num_patches = vit.patch_embed.num_patches
        return num_patches, num_text_patches

    def get_num_patches(self) -> int:
        """Returns the number of vision patches output by the vision backbone."""
        num_patches, num_text_patches = self.get_num_patches_single(self.vision_backbone.featurizer, self.vla_cfg.featurizer_cfg)
        num_patches_fused, num_text_patches_fused = self.get_num_patches_single(self.vision_backbone.fused_featurizer, self.vla_cfg.fused_featurizer_cfg)

        assert num_patches == num_patches_fused
        return num_patches + num_text_patches + num_text_patches_fused

    def get_num_images_in_input(self) -> int:
        """Returns the number of input images for the vision backbone."""
        return self.vision_backbone.get_num_images_in_input()

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        """Sets the number of input images for the vision backbone."""
        self.vision_backbone.set_num_images_in_input(num_images_in_input)

    def forward(self, pixel_values: torch.Tensor, language_embeddings: torch.Tensor = None, instructions = None) -> torch.Tensor:
        """
        Implements the forward pass for the vision backbone with FiLM to infuse language inputs into visual features.

        Identical to PrismaticVisionBackbone.forward() except that language embeddings are also used as input.

        Args:
            pixel_values (torch.Tensor): Pixels for input image(s), (B, C, H, W).
            language_embeddings (torch.Tensor): Language embeddings for the task description, (B, seq_len, llm_dim).
        """
        # For FiLM: Average the language embeddings of the task description
        average_language_embedding = language_embeddings.mean(dim=1) if language_embeddings is not None else None

        if self.get_num_images_in_input() == 1:
            if not self.vision_backbone.use_fused_vision_backbone:
                return self.vision_backbone.featurizer(pixel_values, average_language_embedding)

            # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)

            if self.enable_cross_vit_interact:
                patches, patches_fused = self.cross_vit_model([img, img_fused], average_language_embedding)
            else:
                patches = self.vision_backbone.featurizer(img, average_language_embedding)
                patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

            outputs = (patches, patches_fused)  # Tuple[Tensor]
        else:
            assert self.vision_backbone.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"

            # Split `pixel_values` into individual images (each with 6 channels: 3 for SigLIP + 3 for DINOv2)
            images = torch.split(pixel_values, [6] * self.get_num_images_in_input(), dim=1)

            # Process each image and collect patches
            all_patches = []
            for img in images:
                # Split each image further into two stacks of channels (each with 3 channels)
                img_regular, img_fused = torch.split(img, [3, 3], dim=1)

                if self.enable_cross_vit_interact:
                    patches, patches_fused = self.cross_vit_model([img_regular, img_fused], average_language_embedding)
                else:
                    # Get patches from both SigLIP and DINOv2 vision transformers
                    patches = self.vision_backbone.featurizer(img_regular, average_language_embedding)
                    patches_fused = self.vision_backbone.fused_featurizer(img_fused, average_language_embedding)

                all_patches.append((patches, patches_fused))

            outputs = all_patches  # List[Tuple[Tensor]]

        new_outputs = []
        outputs = [outputs] if not isinstance(outputs, list) else outputs
        for feats in outputs:
            patches, patches_fused = feats
            if self.featurizer_use_mm_sampler:
                patches = self.featurizer_mm_sampler(feats[0], instructions)
            if self.fused_featurizer_use_mm_sampler:
                patches_fused = self.fused_featurizer_mm_sampler(feats[1], instructions)
            new_outputs.append((patches, patches_fused))

        new_outputs = new_outputs[0] if len(new_outputs) == 1 else new_outputs
        return new_outputs
