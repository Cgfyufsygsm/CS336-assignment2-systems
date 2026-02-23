from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class NaiveDDP(nn.Module):
    """
    Naive DDP wrapper:
    - Broadcast initial model state from rank 0.
    - Expose forward() as a thin passthrough.
    - Synchronize gradients by all-reducing each parameter gradient.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing NaiveDDP.")

        self.module = module
        self.world_size = dist.get_world_size()

        self._broadcast_initial_state()

    def _broadcast_initial_state(self) -> None:
        # Broadcast both parameters and buffers to match rank-0 initialization.
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.div_(self.world_size)

class MinimalDDPFlat(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing NaiveDDP.")

        self.module = module
        self.world_size = dist.get_world_size()

        self._broadcast_initial_state()

    def _broadcast_initial_state(self) -> None:
        # Broadcast both parameters and buffers to match rank-0 initialization.
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        grads = []
        params = []
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad)
                params.append(p)
        if not grads:
            return
        flat_grad = _flatten_dense_tensors(grads) # Flatten
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=False) # All-reduce
        flat_grad.div_(self.world_size) # Average
        synced = _unflatten_dense_tensors(flat_grad, grads) # Unflatten

        for p, synced_grad in zip(params, synced):
            p.grad.copy_(synced_grad)
