"""
Gradient all_reduce overlapped with backward (DDP-style).

When each parameter's gradient is ready during ``loss.backward()``, launch
async ``all_reduce``. Autograd continues onto earlier layers while communication
is in flight. Call ``synchronize()`` before ``optimizer.step()``.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def _default_group(group: Optional[dist.ProcessGroup]) -> Optional[dist.ProcessGroup]:
    if group is not None:
        return group
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    return None


def _world_size(group: Optional[dist.ProcessGroup]) -> int:
    if group is None and not (dist.is_available() and dist.is_initialized()):
        return 1
    return dist.get_world_size(group)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class AsyncGradAllReducer:
    """
    Hook-based async gradient all_reduce with backward overlap.

    Example::

        reducer = AsyncGradAllReducer(model.parameters(), group=group)
        reducer.broadcast_params()

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        reducer.backward(loss)     # AR overlaps with remaining backward
        reducer.synchronize()      # wait + average
        optimizer.step()

        reducer.close()

    Gradient accumulation::

        optimizer.zero_grad(set_to_none=True)
        with reducer.no_sync():
            reducer.backward(loss_micro1)
        reducer.backward(loss_micro2)
        reducer.synchronize()
        optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        group: Optional[dist.ProcessGroup] = None,
        average: bool = True,
        bucket_size_mb: float = 25.0,
    ) -> None:
        self.group = _default_group(group)
        self.average = average
        self.bucket_size_bytes = max(int(bucket_size_mb * 1024 * 1024), 1)
        self.require_backward_grad_sync = True

        self._params: List[nn.Parameter] = [p for p in params if p.requires_grad]

        # Keep AccumulateGrad nodes alive; otherwise hooks can be dropped.
        self._grad_accs: List[object] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._pending: List[Tuple[dist.Work, torch.Tensor, List[nn.Parameter]]] = []

        self._scheduled: Dict[int, bool] = {}
        self._bucket_params: List[nn.Parameter] = []
        self._bucket_bytes = 0

        self._install_hooks()

    def _install_hooks(self) -> None:
        self.close()
        for param in self._params:
            # Leaf → AccumulateGrad (same pattern historically used by DDP).
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            self._grad_accs.append(grad_acc)
            handle = grad_acc.register_hook(self._make_ready_hook(param))
            self._hook_handles.append(handle)

    def _make_ready_hook(self, param: nn.Parameter) -> Callable:
        def on_ready(*_unused) -> None:
            if not self.require_backward_grad_sync:
                return
            if _world_size(self.group) <= 1:
                return
            pid = id(param)
            if self._scheduled.get(pid, False):
                return
            if param.grad is None:
                return
            self._scheduled[pid] = True
            self._enqueue(param)

        return on_ready

    def _enqueue(self, param: nn.Parameter) -> None:
        assert param.grad is not None
        size = _nbytes(param.grad)
        if self._bucket_params and self._bucket_bytes + size > self.bucket_size_bytes:
            self._flush_bucket()

        self._bucket_params.append(param)
        self._bucket_bytes += size
        if self._bucket_bytes >= self.bucket_size_bytes:
            self._flush_bucket()

    def _flush_bucket(self) -> None:
        if not self._bucket_params:
            return

        params = self._bucket_params
        self._bucket_params = []
        self._bucket_bytes = 0

        grads = [p.grad.detach().contiguous() for p in params]  # type: ignore[union-attr]
        if len(grads) == 1:
            buf = grads[0].clone()
        else:
            buf = torch._utils._flatten_dense_tensors(grads).clone()

        work = dist.all_reduce(buf, group=self.group, async_op=True)
        self._pending.append((work, buf, params))

    def broadcast_params(self, src: int = 0) -> None:
        if _world_size(self.group) <= 1:
            return
        for p in self._params:
            dist.broadcast(p.data, src=src, group=self.group)

    def prepare_for_backward(self) -> None:
        self._scheduled = {id(p): False for p in self._params}
        self._bucket_params = []
        self._bucket_bytes = 0

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        self.prepare_for_backward()
        loss.backward(retain_graph=retain_graph)
        if self.require_backward_grad_sync:
            self._flush_bucket()

    def synchronize(self) -> None:
        """Wait for in-flight all_reduces; write averaged grads into ``.grad``."""
        if _world_size(self.group) <= 1:
            self._pending.clear()
            return

        self._flush_bucket()
        world = _world_size(self.group)

        for work, buf, params in self._pending:
            work.wait()
            if len(params) == 1:
                p = params[0]
                assert p.grad is not None
                p.grad.copy_(buf)
                if self.average:
                    p.grad.mul_(1.0 / world)
            else:
                ref = [p.grad.contiguous() for p in params]  # type: ignore[union-attr]
                pieces = torch._utils._unflatten_dense_tensors(buf, ref)
                for p, g in zip(params, pieces):
                    assert p.grad is not None
                    p.grad.copy_(g)
                    if self.average:
                        p.grad.mul_(1.0 / world)

        self._pending.clear()

    def no_sync(self) -> "_NoSyncContext":
        return _NoSyncContext(self)

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._grad_accs.clear()
        self._pending.clear()
        self._bucket_params.clear()
        self._bucket_bytes = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _NoSyncContext:
    def __init__(self, reducer: AsyncGradAllReducer) -> None:
        self.reducer = reducer
        self._prev = True

    def __enter__(self) -> AsyncGradAllReducer:
        self._prev = self.reducer.require_backward_grad_sync
        self.reducer.require_backward_grad_sync = False
        return self.reducer

    def __exit__(self, *args) -> None:
        self.reducer.require_backward_grad_sync = self._prev


class DistributedDataParallelOverlap(nn.Module):
    """
    DDP-like wrapper: module forward + async gradient all_reduce overlap.

    Example::

        model = DistributedDataParallelOverlap(model, group=group)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        model.backward(loss)
        model.synchronize()
        optimizer.step()
    """

    def __init__(
        self,
        module: nn.Module,
        group: Optional[dist.ProcessGroup] = None,
        average: bool = True,
        bucket_size_mb: float = 25.0,
        broadcast_buffers: bool = True,
    ) -> None:
        super().__init__()
        self.module = module
        self.group = _default_group(group)
        self.broadcast_buffers = broadcast_buffers
        self.reducer = AsyncGradAllReducer(
            module.parameters(),
            group=self.group,
            average=average,
            bucket_size_mb=bucket_size_mb,
        )
        self.reducer.broadcast_params()
        if broadcast_buffers:
            self._broadcast_buffers()

    def _broadcast_buffers(self) -> None:
        if _world_size(self.group) <= 1:
            return
        for buf in self.module.buffers():
            dist.broadcast(buf.data, src=0, group=self.group)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor, *, retain_graph: bool = False) -> None:
        self.reducer.backward(loss, retain_graph=retain_graph)

    def synchronize(self) -> None:
        self.reducer.synchronize()

    def no_sync(self) -> _NoSyncContext:
        return self.reducer.no_sync()

    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)

    def eval(self):
        self.module.eval()
        return super().eval()


def allreduce_grads_async(
    params: Sequence[nn.Parameter],
    group: Optional[dist.ProcessGroup] = None,
    average: bool = True,
) -> None:
    """
    Post-backward async all_reduce over existing ``param.grad`` tensors.

    Prefer ``AsyncGradAllReducer`` when overlap with backward compute is needed.
    """
    group = _default_group(group)
    if _world_size(group) <= 1:
        return

    handles: List[dist.Work] = []
    for p in params:
        if p.grad is None:
            continue
        handles.append(dist.all_reduce(p.grad, group=group, async_op=True))

    for h in handles:
        h.wait()

    if average:
        scale = 1.0 / _world_size(group)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
