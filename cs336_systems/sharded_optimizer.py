from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Optimizer-state sharding wrapper.

    Each rank owns optimizer state for a shard of parameters and performs
    optimizer updates only on that shard. Updated parameters are then broadcast
    from their owner rank to keep model weights synchronized across ranks.
    """

    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing ShardedOptimizer.")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self._all_params: list[torch.nn.Parameter] = []
        self._owner_rank_by_param_id: dict[int, int] = {}
        self._next_param_index = 0

        self._local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: torch.optim.Optimizer | None = None

        # Required by the assignment spec; this also routes initial parameter
        # groups through our add_param_group implementation.
        super().__init__(params, defaults={})

        if self._local_param_groups:
            self._local_optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)

    def _assign_owner_if_new(self, param: torch.nn.Parameter) -> int:
        param_id = id(param)
        if param_id not in self._owner_rank_by_param_id:
            owner_rank = self._next_param_index % self.world_size
            self._owner_rank_by_param_id[param_id] = owner_rank
            self._all_params.append(param)
            self._next_param_index += 1
        return self._owner_rank_by_param_id[param_id]

    def add_param_group(self, param_group: dict[str, Any]):
        # Keep wrapper param_groups/state compatible with Optimizer behavior.
        super().add_param_group(param_group)
        full_group = self.param_groups[-1]

        local_params: list[torch.nn.Parameter] = []
        for param in full_group["params"]:
            owner_rank = self._assign_owner_if_new(param)
            if owner_rank == self.rank:
                local_params.append(param)

        if not local_params:
            return

        local_group = {k: v for k, v in full_group.items() if k != "params"}
        local_group["params"] = local_params
        self._local_param_groups.append(local_group)

        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)
        else:
            self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None
        if self._local_optimizer is not None:
            if closure is None:
                if kwargs:
                    loss = self._local_optimizer.step(**kwargs)
                else:
                    loss = self._local_optimizer.step()
            else:
                loss = self._local_optimizer.step(closure=closure, **kwargs)

        # Synchronize updated parameters across ranks.
        for param in self._all_params:
            owner_rank = self._owner_rank_by_param_id[id(param)]
            dist.broadcast(param.data, src=owner_rank)
        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        # Clear gradients on all model parameters (not just this rank's shard),
        # otherwise non-local grads can accumulate across training steps.
        for param in self._all_params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()
