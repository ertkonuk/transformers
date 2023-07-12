# coding=utf-8
# Copyright 2022 Tugrul Konuk The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch NVGPT model. """


import math
import os
import numbers
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
from torch.nn.parameter import Parameter
from einops import rearrange
from torch import einsum
from torch import Tensor, Size

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None

from ...activations import ACT2FN
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_nvgpt import NVGPTConfig
from .tokenization_nvgpt import NVGPTTokenizer

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "nvidia/nvgpt-2b-001"
_CONFIG_FOR_DOC = "NVGPTConfig"

NVGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/nvgpt-2b-001",
    # See all NVGPT models at https://huggingface.co/models?filter=nvgpt
]

NVGPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`NVGPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

def load_tf_weights_in_nvgpt(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

@add_start_docstrings(
    "The bare NVGPT Model outputting raw hidden-states without any specific head on top.",
    NVGPT_START_DOCSTRING,
)

class NVGPTPreTrainedModel(PreTrainedModel):
    config_class = NVGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NVGPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, NVGPTModel):
            module.gradient_checkpointing = value


NVGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

_shape_t = Union[int, List[int], Size]
class NVGPTLayerNorm1P(torch.nn.LayerNorm):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight + 1., self.bias, self.eps)     

class NVGPTLayerNorm(torch.nn.LayerNorm):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)  

# Adapted from https://github.com/NVIDIA/apex/blob/ba027dd0bac621a5c3d16bfb90d73e2d48d2588e/apex/normalization/fused_layer_norm.py#L300

# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input

class NVGPTRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs
    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)

    def forward(self, input):
        return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
    
class NVGPTRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return rearrange(emb, 'n d -> n 1 1 d')

def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)

class NVGPTMLP(nn.Module):
    def __init__(
        self,
        config: NVGPTConfig,
    ):
        super().__init__()
        self.dense_h_to_4h = torch.nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=config.bias)
        self.dense_4h_to_h = torch.nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=config.bias)        
        self.activation_func = ACT2FN[config.activation]

    def forward(self, x):        
        intermediate_parallel = self.dense_h_to_4h(x)

        intermediate_parallel, intermediate_parallel_2 = torch.chunk(intermediate_parallel, 2, dim=-1)
        intermediate_parallel = self.activation_func(intermediate_parallel) * intermediate_parallel_2

        return self.dense_4h_to_h(intermediate_parallel)
    
class NVGPTCoreAttention(torch.nn.Module):
    """ Causal scaled dot product attention    
    """
    def __init__(self, 
                 config: NVGPTConfig, 
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================        

        if past_key_values is not None:
            past_key, past_value = past_key_values            
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        
        # Keep previous key and value tensors
        past_key_values = (key_layer, value_layer) if use_cache else None

        # [b, np, sq, sk]
        output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)        

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device, #device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)        

        # ===========================
        # Attention probs and dropout
        # ===========================
        if attention_mask is not None:
            attention_scores.masked_fill_(attention_mask == 0, -10000.0)
        
        # Casual mask
        _, _, sq, sk = attention_scores.size()
        causal_mask = torch.tril(torch.ones((sq, sk))).unsqueeze(0).unsqueeze(0)

        causal_mask = causal_mask.to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -10000.0)

        attention_scores = torch.nn.Softmax(dim=-1)(attention_scores)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_scores = attention_scores.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_scores, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if not output_attentions:
            attention_scores = None
            
        return context_layer, attention_scores, past_key_values

class NVGPTFlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            self.dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output

class PyTorchNVGPTFlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax using pytorch scaled_dot_product_attention().
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()

        torch.backends.cuda.enable_flash_sdp(True)        
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, query, key, value):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            query, key, value: The tensor containing the query, key, and value. (B, S, H, D)
        """
        #assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        #assert all((i.is_cuda for i in (q,k,v)))
        batch_size, seqlen_q = query.shape[0], query.shape[1]
        seqlen_k = key.shape[1]

        query, key, value = [x.transpose(1,2) for x in [query, key, value]]

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            self.dropout_p = 0

        output = torch.nn.functional.scaled_dot_product_attention(query=query, 
                                                                  key=key, 
                                                                  value=value, 
                                                                  dropout_p=self.dropout_p,
                                                                  is_causal=True)
        output = output.transpose(1,2).contiguous()
        #output = rearrange(output, 'b s h d -> b s (h d)')
        return output
        
class NVGPTAttention(torch.nn.Module):
    """Multi-headed casual attention
        
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
            self, 
            config: NVGPTConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.use_flash_attention = config.use_flash_attention
        self.use_native_pytorch_flash_attention = config.use_native_pytorch_flash_attention

        if self.use_flash_attention:
            if self.use_native_pytorch_flash_attention:
                if flash_attn_unpadded_func is None:
                    raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')
            
        if (self.hidden_size_per_attention_head * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )
        
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.num_attention_heads * self.hidden_size_per_attention_head, bias=config.bias)
        self.dense = nn.Linear(self.num_attention_heads * self.hidden_size_per_attention_head, self.hidden_size, bias=config.bias)

        self.core_attention = NVGPTCoreAttention(config=config)
        if self.use_flash_attention:
            if self.use_native_pytorch_flash_attention:
                self.core_attention_flash = PyTorchNVGPTFlashSelfAttention(
                    causal=True, attention_dropout=config.attention_dropout
                )
            else:
                self.core_attention_flash = NVGPTFlashSelfAttention(
                    causal=True, attention_dropout=config.attention_dropout
                )                
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_attention_heads, self.hidden_size_per_attention_head).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)
                
        
        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads,
            3 * self.hidden_size_per_attention_head,
        )

        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.chunk(mixed_x_layer, 3, dim=-1)
        
        # ===================================================
        # Adjust key, value, and attention mask for inference
        # ===================================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            rotary_pos_emb = rotary_pos_emb if isinstance(rotary_pos_emb, tuple) else ((rotary_pos_emb,) * 2)
        
        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)

        if not self.use_flash_attention:
            context_layer, attention_scores, past_key_values = self.core_attention(query_layer, 
                                                                                   key_layer, 
                                                                                   value_layer,
                                                                                   attention_mask,
                                                                                   past_key_values,
                                                                                   rotary_pos_emb,
                                                                                   output_attentions,
                                                                                   use_cache)
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.dense(context_layer)

        if not output_attentions:
            attention_scores = None

        return output, attention_scores, past_key_values
        
class NVGPTDecoderLayer(nn.Module):
    def __init__(self, config: NVGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attention = NVGPTAttention(config=config)
        self.mlp = NVGPTMLP(config=config)

        if config.normalization not in ['layernorm', 'layernorm1p', 'rmsnorm']:
            raise ValueError(f'normalization must be "layernorm", "layernorm1p" or "rmsnorm", found {config.normalization}')
        
        if config.normalization == 'layernorm':
            self.input_layernorm = NVGPTLayerNorm(config.hidden_size, eps=config.layernorm_eps)
            self.post_attention_layernorm = NVGPTLayerNorm(config.hidden_size, eps=config.layernorm_eps)
        elif config.normalization == 'layernorm1p':
            self.input_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)
            self.post_attention_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)
        else:
            self.input_layernorm = NVGPTRMSNorm(config.hidden_size, eps=config.layernorm_eps)       
            self.post_attention_layernorm = NVGPTRMSNorm(config.hidden_size, eps=config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,        
        rotary_pos_emb: Optional[Tuple[torch.Tensor]] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            past_key_values (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # Casual self attention.
        if rotary_pos_emb is not None:
            # self attention pos_emb is (q, q)
            self_attention_pos_emb = (rotary_pos_emb, rotary_pos_emb)            
        else:
            self_attention_pos_emb = None            

        residual = hidden_states
        # Layer norm at the beginning of the transformer layer.
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, attention_weights, present_key_value = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            rotary_pos_emb=self_attention_pos_emb,
            output_attentions=output_attentions,
            use_cache = use_cache,
        )

        # Add to the residual stream
        hidden_states = residual + hidden_states

        residual = hidden_states
        # Post-LN normalization after residual
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)

        # Add to the residual stream
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

NVGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config ([`~NVGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

NVGPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`NVGPTTokenizer`].
            See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert *input_ids* indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare NVGPT Model outputting raw hidden-states without any specific head on top.",
    NVGPT_START_DOCSTRING,
)
class NVGPTModel(NVGPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`NVGPTDecoderLayer`]

    Args:
        config: NVGPTConfig
    """

    def __init__(self, config: NVGPTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        rotary_dim = config.hidden_size // config.num_attention_heads
        assert 0 < config.rotary_percentage <= 1
        if config.rotary_percentage < 1:
            rotary_dim = int(rotary_dim * config.rotary_percentage)
        self.rotary_pos_emb = NVGPTRotaryEmbedding(rotary_dim)

        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList([NVGPTDecoderLayer(config) for _ in range(config.num_layers)])

        if config.normalization not in ['layernorm', 'layernorm1p', 'rmsnorm']:
            raise ValueError(f'normalization must be "layernorm", "layernorm1p" or "rmsnorm", found {config.normalization}')
        
        if config.normalization == 'layernorm':
            self.final_layernorm = NVGPTLayerNorm(config.hidden_size, eps=config.layernorm_eps)
        elif config.normalization == 'layernorm1p':
            self.final_layernorm = NVGPTLayerNorm1P(config.hidden_size, eps=config.layernorm_eps)
        else:
            self.final_layernorm = NVGPTRMSNorm(config.hidden_size, eps=config.layernorm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing
        
    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    @add_start_docstrings_to_model_forward(NVGPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.LongTensor] = None,              
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,        
        output_hidden_states: Optional[bool] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and input_embeddings is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            seq_length, batch_size = input_ids.shape
        elif input_embeddings is not None:
            seq_length, batch_size, _ = input_embeddings.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length        
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[0]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else input_embeddings.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(1).view(seq_length, -1)

        if input_embeddings is None:
            input_embeddings = self.embedding(input_ids)
            input_embeddings = input_embeddings.transpose(0, 1).contiguous()
                 
        rotary_pos_emb = self.rotary_pos_emb(input_embeddings.size(0))

        hidden_states = input_embeddings

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
       
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,                    
                    rotary_pos_emb=rotary_pos_emb,
                    past_key_values=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]        

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attentions] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class NVGPTForCausalLM(NVGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.model = NVGPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
   
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.output_layer = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(NVGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,        
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        input_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, NVGPTForCausalLM

        >>> model = NVGPTForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,            
            past_key_values=past_key_values,
            input_embeddings=input_embeddings,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            use_cache=use_cache,                
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states)

        # [s b h] -> [b s h]
        logits = logits.transpose(0,1).contiguous()
    
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for past_key_values in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in past_key_values),)
        return reordered_past

    @classmethod
    def from_megatron_weights_and_config(cls, model_weights, config):
        checkpoint = torch.load(model_weights)
        ckpt_keys = list(checkpoint.keys())
        keymap = {}
        for key in ckpt_keys:
            if not 'output_layer' in key:
                new_key = key.replace('language_model.','').replace('encoder.','').replace('word_embeddings.','')
            else:
                new_key = key.replace('model.language_model.','').replace('encoder.','').replace('word_embeddings.','').replace('output_layer','lm_head')

            keymap[key] = new_key

        def rekey(inp_dict, keys_replace):
            return {keys_replace.get(k, k): v for k, v in inp_dict.items()}

        new_ckpt = rekey(checkpoint, keymap)

        del checkpoint

        model = cls(config)
        model.load_state_dict(new_ckpt)
        return model

    @classmethod
    def from_nemo_file(cls, nemo_file):
        import tarfile
        import yaml
        #from configuration_nvgpt import NVGPTConfig

        with tarfile.open(nemo_file) as tar:
            # Instantiate the model first, then load weights
            try:
                nemo_cfg = yaml.safe_load(tar.extractfile('model_config.yaml').read())
            except:
                nemo_cfg = yaml.safe_load(tar.extractfile('./model_config.yaml').read())

            config = NVGPTConfig(
                            hidden_size=nemo_cfg['hidden_size'],
                            ffn_hidden_size=nemo_cfg['ffn_hidden_size'],
                            num_layers=nemo_cfg['num_layers'],
                            num_attention_heads=nemo_cfg['num_attention_heads'],
                            normalization=nemo_cfg['normalization'],
            )
            
            # Initialize and cast weights
            dtype = torch.bfloat16 if nemo_cfg['precision'] == 'bf16' else torch.float16
            model = cls(config).to(dtype)

            try:
                checkpoint = torch.load(tar.extractfile('model_weights.ckpt'))  
            except:
                checkpoint = torch.load(tar.extractfile('./model_weights.ckpt'))
            ckpt_keys = list(checkpoint.keys())
            
            for key in ckpt_keys:
                if not 'output_layer' in key:
                    new_key = key.replace('language_model.','').replace('encoder.','').replace('word_embeddings.','')
                else:
                    new_key = key.replace('model.language_model.','').replace('encoder.','').replace('word_embeddings.','').replace('output_layer','lm_head')

                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]            

            model.load_state_dict(checkpoint)
            del checkpoint
        return model