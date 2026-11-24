#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemCache_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Synthetic BF16 cache layouts for 128 original tokens; no model weights are loaded."""

DEFAULT_MODEL = "deepseek-r1"
MODEL_CHOICES = (
    DEFAULT_MODEL,
    "glm-5.1",
    "glm-5.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
)
TOKENS_PER_BLOCK = 128
BF16_BYTES = 2
LATENT_DIM = 512
ROPE_DIM = 64
INDEX_DIM = 128
R1_LAYERS = 61
GLM_LAYERS = 78
V4_FLASH_LAYERS = 43
V4_PRO_LAYERS = 61
CSA_RATIO = 4
HCA_RATIO = 128
PURE_SWA_LAYERS = 2
GLM_INITIAL_INDEX_LAYERS = 3
GLM_SHARED_INDEX_START = 6
GLM_INDEX_INTERVAL = 4
V41_ENCODER_LAYERS = 20
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_ENCODER_RATIO = 2


def get_model_sizes(model):
    """Return ordered byte lengths, counting each persistent cache owner once.

    GLM includes the indexer K of full indexer layers. V4/V4.1 include only
    global prefix KV and indexer K, excluding SWA, compressor state and MTP.
    Sizes represent BF16 payloads, not a framework's packed quantized layout.
    """
    if model not in MODEL_CHOICES:
        raise ValueError(f"Unknown model {model!r}; choose from {', '.join(MODEL_CHOICES)}")

    latent_size = TOKENS_PER_BLOCK * LATENT_DIM * BF16_BYTES
    rope_size = TOKENS_PER_BLOCK * ROPE_DIM * BF16_BYTES
    index_size = TOKENS_PER_BLOCK * INDEX_DIM * BF16_BYTES
    sizes = []
    if model == DEFAULT_MODEL:
        return [size for _ in range(R1_LAYERS) for size in (latent_size, rope_size)]
    if model.startswith("glm-"):
        full_index_layers = set(range(GLM_INITIAL_INDEX_LAYERS))
        full_index_layers.update(range(GLM_SHARED_INDEX_START, GLM_LAYERS, GLM_INDEX_INTERVAL))
        for layer in range(GLM_LAYERS):
            sizes.extend((latent_size, rope_size))
            if model == "glm-5.1" or layer in full_index_layers:
                sizes.append(index_size)
        return sizes
    if model == "deepseek-v4.1-flash":
        for layer in V41_KV_SOURCE_LAYERS:
            ratio = V41_ENCODER_RATIO if layer < V41_ENCODER_LAYERS else 1
            sizes.extend((latent_size // ratio, index_size // ratio))
        return sizes

    flash = model == "deepseek-v4-flash"
    layer_count = V4_FLASH_LAYERS if flash else V4_PRO_LAYERS
    for layer in range(layer_count):
        if flash and layer < PURE_SWA_LAYERS:
            continue
        ratio = CSA_RATIO if layer >= PURE_SWA_LAYERS and layer % 2 == 0 else HCA_RATIO
        sizes.append(latent_size // ratio)
        if ratio == CSA_RATIO:
            sizes.append(index_size // ratio)
    return sizes


def get_key_prefix(model, data_dim):
    # Keep existing R1 and contiguous keys compatible with separate write/read runs.
    if data_dim != 2 or model == DEFAULT_MODEL:
        return "key_"
    return f"key_{model}_bf16_{TOKENS_PER_BLOCK}_v1_"
