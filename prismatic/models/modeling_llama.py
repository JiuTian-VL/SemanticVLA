import re
import torch
import time
import transformers
from dataclasses import dataclass
from transformers.models.llama.modeling_llama import *
from transformers import LlamaConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def llama_sdpa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    num_vision: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # ======================== NEW ========================
    if attention_mask is None:
        # We need to modify attention mask later, so it can not be None
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        attention_mask = _prepare_4d_causal_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states, past_key_values_length=0)
    # =====================================================

    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # In case static cache is used, it is an instance attribute.
    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # In case we are not compiling, we may set `causal_mask` to None, which is required to dispatch to SDPA's Flash Attention 2 backend, rather
    # relying on the `is_causal` argument.

    # Causal Mask Example:
    #
    # No pad tokens:
    #   0 -inf -inf
    #   0   0  -inf
    #   0   0    0
    # 1 pad token:
    #   0 -inf -inf
    #   0   0  -inf
    #   0   0  -inf
    # 2 pad tokens:
    #   0 -inf -inf
    #   0 -inf -inf
    #   0 -inf -inf

    # ================= NEW =================
    if causal_mask is not None:
        min_dtype = torch.finfo(hidden_states.dtype).min
        num_pad = (causal_mask[:, 0, -1, :] == min_dtype).sum(dim=1)
        num_act = ACTION_DIM * NUM_ACTIONS_CHUNK if self.num_action_tokens == -1 else self.num_action_tokens * NUM_ACTIONS_CHUNK
        num_act += 1  # stop token

        new_mask = causal_mask.clone()
        for idx, n_pad in enumerate(num_pad):
            if n_pad == 0:
                new_mask[idx, :, -num_act:, -num_act:] = 0
            else:
                new_mask[idx, :, -(num_act + n_pad):-n_pad, -(num_act + n_pad):-n_pad] = 0

            # start = -(num_act + n_pad)
            # end = -n_pad if n_pad else None
            # causal_mask[idx, :, start:end, start:end] = 0

        # num_vision = self.get_num_patches() * self.get_num_images_in_input() if num_vision is None else num_vision
        # causal_mask[:, :, 1: 1+num_vision, 1: 1+num_vision] = 0  # bi-direction attention for visual token
        causal_mask = new_mask
    # =======================================

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        # is_causal=causal_mask is None and q_len > 1,
        is_causal=False,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def replace_llama_spda_forward():
    transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attention_forward