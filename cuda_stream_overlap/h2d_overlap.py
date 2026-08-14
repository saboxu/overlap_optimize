"""
Overlap DataLoader H2D (CPU -> GPU) with model forward using CUDA streams.

Naive path: batch.to(device) on the default stream, then forward.
Overlap path: a side copy stream prefetches batch i+1 while default-stream
forward runs on batch i.

Requires a CUDA GPU. DataLoader must use pin_memory=True, and .to(..., non_blocking=True).
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .d2h_overlap import ChunkFFN


Batch = Tuple[torch.Tensor, torch.Tensor]


class H2DPrefetcher:
    """
    One-batch lookahead: copy stream DMAs the next CPU batch while the caller
    computes on the current GPU batch (default stream).

    Usage::

        pref = H2DPrefetcher(loader, device)
        x, y = pref.next()
        while x is not None:
            out = model(x)
            x, y = pref.next()
    """

    def __init__(self, loader: DataLoader, device: torch.device) -> None:
        self.device = device
        self._it: Iterator[Batch] = iter(loader)
        self.copy_stream = torch.cuda.Stream(device=device)
        self.next_x: Optional[torch.Tensor] = None
        self.next_y: Optional[torch.Tensor] = None
        self._preload()

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True)

    def _preload(self) -> None:
        try:
            x_cpu, y_cpu = next(self._it)
        except StopIteration:
            self.next_x = None
            self.next_y = None
            return

        with torch.cuda.stream(self.copy_stream):
            self.next_x = self._to_device(x_cpu)
            self.next_y = self._to_device(y_cpu)

    def next(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Default stream must not read next_x until this H2D has finished.
        torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
        x, y = self.next_x, self.next_y
        if x is not None:
            # Allocator: these tensors were born on copy_stream but will be
            # consumed on the current (default) stream. record_stream keeps
            # the memory alive until default-stream kernels using x/y finish.
            x.record_stream(torch.cuda.current_stream(self.device))
            assert y is not None
            y.record_stream(torch.cuda.current_stream(self.device))
        self._preload()  # launch H2D of the following batch; overlaps with forward
        return x, y


def make_loader(
    n_batches: int,
    batch: int,
    dim: int,
    *,
    pin_memory: bool,
    seed: int = 0,
) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_batches * batch, dim, generator=g)
    y = torch.randn(n_batches * batch, dim, generator=g)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=pin_memory,
    )


def epoch_sync(loader: DataLoader, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Blocking H2D on the default stream, then forward."""
    acc = torch.zeros((), device=device)
    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        pred = model(x)
        acc = acc + (pred - y).square().mean()
    return acc


def epoch_overlap(loader: DataLoader, model: nn.Module, device: torch.device) -> torch.Tensor:
    """Prefetch H2D on a side stream; forward on the default stream."""
    acc = torch.zeros((), device=device)
    pref = H2DPrefetcher(loader, device)
    x, y = pref.next()
    while x is not None:
        assert y is not None
        pred = model(x)
        acc = acc + (pred - y).square().mean()
        x, y = pref.next()
    return acc


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _run(
    n_batches: int,
    batch: int,
    dim: int,
    n_layers: int,
    warmup: int,
    iters: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. This example uses cuda.Stream / non_blocking H2D."
        )

    device = torch.device("cuda")
    model = ChunkFFN(dim, n_layers=n_layers).to(device).eval()
    loader = make_loader(n_batches, batch, dim, pin_memory=True)

    with torch.inference_mode():
        loss_sync = epoch_sync(loader, model, device)
        loss_overlap = epoch_overlap(loader, model, device)
        if not torch.allclose(loss_sync, loss_overlap, atol=1e-4, rtol=1e-3):
            raise RuntimeError(
                f"sync vs overlap mismatch  sync={loss_sync.item()}  "
                f"overlap={loss_overlap.item()}"
            )

        def _sync():
            epoch_sync(loader, model, device)

        def _overlap():
            epoch_overlap(loader, model, device)

        for _ in range(warmup):
            _sync()
            _overlap()

        sync_ms = min(_time_ms(_sync) for _ in range(iters))
        overlap_ms = min(_time_ms(_overlap) for _ in range(iters))

    nbytes = batch * dim * 4 * 2  # x and y, float32
    print(
        f"ok  batches={n_batches}  batch={batch}  dim={dim}  n_layers={n_layers}  "
        f"h2d/step≈{nbytes / 1024**2:.1f} MiB"
    )
    print(f"sync     {sync_ms:.3f} ms")
    print(f"overlap  {overlap_ms:.3f} ms")
    print(f"speedup  {sync_ms / overlap_ms:.2f}x")


def main() -> None:
    """
    Launch example:

      python -m cuda_stream_overlap.h2d_overlap
    """
    import argparse

    p = argparse.ArgumentParser(description="CUDA stream DataLoader H2D / forward overlap")
    p.add_argument("--n-batches", type=int, default=32)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=8)
    args = p.parse_args()

    _run(
        n_batches=args.n_batches,
        batch=args.batch,
        dim=args.dim,
        n_layers=args.n_layers,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
