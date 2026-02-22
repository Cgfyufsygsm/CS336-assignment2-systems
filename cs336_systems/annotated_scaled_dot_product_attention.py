from __future__ import annotations

import math

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.nn_utils import softmax


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with nvtx.range("scaled dot product attention"):
        d_k = K.shape[-1]

        with nvtx.range("computing attention scores"):
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx.range("final matmul"):
            output = torch.matmul(attention_weights, V)

    return output


def install_annotated_scaled_dot_product_attention() -> None:
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

