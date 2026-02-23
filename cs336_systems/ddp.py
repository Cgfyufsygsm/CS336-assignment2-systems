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


class DDPOverlapIndividualParameters(nn.Module):
    """
    DDP wrapper that overlaps communication with backward computation by
    asynchronously all-reducing each parameter gradient once it is ready.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP.")

        self.module = module
        self.world_size = dist.get_world_size()
        self._pending_works: list[dist.Work] = []
        self._grad_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._broadcast_initial_state()
        self._register_gradient_hooks()

    def _broadcast_initial_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _register_gradient_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            handle = param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))
            self._grad_hook_handles.append(handle)

    def _make_post_accumulate_hook(self, param: nn.Parameter):
        def _hook(_: torch.Tensor) -> None:
            if param.grad is None:
                return
            param.grad.div_(self.world_size)
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_works.append(work)

        return _hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for work in self._pending_works:
            work.wait()
        self._pending_works.clear()


class DDPOverlapBucketed(nn.Module):
    """
    DDP wrapper that buckets gradients and asynchronously all-reduces
    each bucket once all gradients in that bucket are ready.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float | None):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP.")

        if bucket_size_mb is not None and bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be > 0 or None.")

        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb

        self._buckets: list[list[nn.Parameter]] = []
        self._param_id_to_bucket: dict[int, int] = {}
        self._bucket_pending_count: list[int] = []
        self._bucket_comm_started: list[bool] = []
        self._param_ready_this_step: dict[int, bool] = {}
        self._inflight_bucket_works: list[tuple[dist.Work, torch.Tensor, list[nn.Parameter], list[torch.Tensor]]] = []
        self._grad_hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._broadcast_initial_state()
        self._build_buckets()
        self._register_gradient_hooks()
        self.on_train_batch_start()

    def _broadcast_initial_state(self) -> None:
        for tensor in self.module.state_dict().values():
            dist.broadcast(tensor, src=0)

    def _build_buckets(self) -> None:
        max_bucket_bytes = None
        if self.bucket_size_mb is not None:
            max_bucket_bytes = int(self.bucket_size_mb * 1_000_000)

        params_reversed = [param for param in reversed(list(self.module.parameters())) if param.requires_grad]
        if not params_reversed:
            return

        current_bucket: list[nn.Parameter] = []
        current_bucket_bytes = 0
        for param in params_reversed:
            param_bytes = param.numel() * param.element_size()
            would_overflow = (
                max_bucket_bytes is not None
                and current_bucket
                and (current_bucket_bytes + param_bytes > max_bucket_bytes)
            )
            if would_overflow:
                self._append_bucket(current_bucket)
                current_bucket = []
                current_bucket_bytes = 0
            current_bucket.append(param)
            current_bucket_bytes += param_bytes

        if current_bucket:
            self._append_bucket(current_bucket)

    def _append_bucket(self, bucket_params: list[nn.Parameter]) -> None:
        bucket_idx = len(self._buckets)
        self._buckets.append(bucket_params)
        for param in bucket_params:
            self._param_id_to_bucket[id(param)] = bucket_idx

    def _register_gradient_hooks(self) -> None:
        for bucket in self._buckets:
            for param in bucket:
                handle = param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))
                self._grad_hook_handles.append(handle)

    def _make_post_accumulate_hook(self, param: nn.Parameter):
        param_id = id(param)
        bucket_idx = self._param_id_to_bucket[param_id]

        def _hook(_: torch.Tensor) -> None:
            if param.grad is None:
                return
            if self._param_ready_this_step.get(param_id, False):
                return

            self._param_ready_this_step[param_id] = True
            self._bucket_pending_count[bucket_idx] -= 1
            if self._bucket_pending_count[bucket_idx] == 0 and not self._bucket_comm_started[bucket_idx]:
                self._launch_bucket_all_reduce(bucket_idx)

        return _hook

    def _launch_bucket_all_reduce(self, bucket_idx: int) -> None:
        bucket_params = self._buckets[bucket_idx]
        bucket_grads: list[torch.Tensor] = []
        for param in bucket_params:
            if param.grad is None:
                return
            bucket_grads.append(param.grad)

        flat_grads = _flatten_dense_tensors(bucket_grads)
        flat_grads.div_(self.world_size)
        work = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
        self._bucket_comm_started[bucket_idx] = True
        self._inflight_bucket_works.append((work, flat_grads, bucket_params, bucket_grads))

    def on_train_batch_start(self) -> None:
        if self._inflight_bucket_works:
            self.finish_gradient_synchronization()

        self._bucket_pending_count = [len(bucket) for bucket in self._buckets]
        self._bucket_comm_started = [False for _ in self._buckets]
        self._param_ready_this_step.clear()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # In case hooks were bypassed for any bucket where all grads still exist.
        for bucket_idx in range(len(self._buckets)):
            if self._bucket_comm_started[bucket_idx]:
                continue
            if all(param.grad is not None for param in self._buckets[bucket_idx]):
                self._launch_bucket_all_reduce(bucket_idx)

        for work, flat_grads, bucket_params, bucket_grads in self._inflight_bucket_works:
            work.wait()
            synced_grads = _unflatten_dense_tensors(flat_grads, bucket_grads)
            for param, synced_grad in zip(bucket_params, synced_grads):
                if param.grad is not None:
                    param.grad.copy_(synced_grad)

        self._inflight_bucket_works.clear()
