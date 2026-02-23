from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


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
