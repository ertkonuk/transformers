# coding=utf-8
# Copyright 2022 Tugrul Konuk and The HuggingFace Inc. team. All rights reserved.
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
""" NVGPT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

NVGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/nvgpt-2b-001": "https://huggingface.co/nvidia/nvgpt-2b-001/resolve/main/config.json",
    # See all NVGPT models at https://huggingface.co/models?filter=nvgpt
}


class NVGPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~NVGPTModel`].
    It is used to instantiate an NVGPT model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the NVGPT [nvidia/nvgpt-2b-001](https://huggingface.co/nvidia/nvgpt-2b-001) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the NVGPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~NVGPTModel`] or
            [`~TFNVGPTModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~NVGPTModel`] or
            [`~TFNVGPTModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:

    ```python
    >>> from transformers import NVGPTModel, NVGPTConfig

    >>> # Initializing a NVGPT nvidia/nvgpt-2b-001 style configuration
    >>> configuration = NVGPTConfig()

    >>> # Initializing a model from the nvidia/nvgpt-2b-001 style configuration
    >>> model = NVGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "nvgpt"
    

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2048,
        ffn_hidden_size=5440,
        num_layers=24,
        num_attention_heads=16,
        activation="silu",
        normalization="layernorm1p",
        max_position_embeddings=4096,
        initializer_range=0.014,
        layernorm_eps=1.0e-05,
        rotary_percentage=0.5,
        use_flash_attention=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        kv_channels=None,
        use_cache=False,
        gradient_checkpointing=False,        
        bias=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_native_pytorch_flash_attention=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.activation = activation
        self.initializer_range = initializer_range
        self.layernorm_eps = layernorm_eps
        self.rotary_percentage = rotary_percentage
        self.use_flash_attention = use_flash_attention
        self.use_cache = use_cache
        self.bias = bias
        self.kv_channels = kv_channels        
        self.gradient_checkpointing = gradient_checkpointing
        self.normalization = normalization
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.use_native_pytorch_flash_attention = use_native_pytorch_flash_attention
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    