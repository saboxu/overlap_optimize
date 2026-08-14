"""
DataLoader H2D / forward overlap in a plain for-loop (prev-handle template).

Same idea as d2h_overlap / MoE all_gather: launch current copy, then wait the
previous copy and compute on that batch. No Prefetcher / iterator wrapper.

Requires a CUDA GPU, pin_memory=True, and non_blocking=True.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .d2h_overlap import ChunkFFN
from .h2d_overlap import _time_ms, epoch_sync, make_loader


def epoch_overlap(loader: DataLoader, model: nn.Module, device: torch.device) -> torch.Tensor:
    """
    for each batch:
      (1) async H2D of the current CPU batch on copy_stream
      (2) wait previous H2D, then forward on that GPU batch (default stream)
    drain the last batch after the loop.
    """
    copy_stream = torch.cuda.Stream(device=device)
    default_stream = torch.cuda.current_stream(device)

    acc = torch.zeros((), device=device)
    prev_x: Optional[torch.Tensor] = None
    prev_y: Optional[torch.Tensor] = None
    prev_done: Optional[torch.cuda.Event] = None

    for x_cpu, y_cpu in loader:
        with torch.cuda.stream(copy_stream):
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            done = torch.cuda.Event()
            done.record(copy_stream)

        if prev_x is not None:
            assert prev_y is not None and prev_done is not None
            default_stream.wait_event(prev_done)
            prev_x.record_stream(default_stream)
            prev_y.record_stream(default_stream)
            pred = model(prev_x)
            acc = acc + (pred - prev_y).square().mean()

        prev_x, prev_y, prev_done = x, y, done

    if prev_x is not None:
        assert prev_y is not None and prev_done is not None
        default_stream.wait_event(prev_done)
        prev_x.record_stream(default_stream)
        prev_y.record_stream(default_stream)
        pred = model(prev_x)
        acc = acc + (pred - prev_y).square().mean()

    return acc


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

    nbytes = batch * dim * 4 * 2
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

      python -m cuda_stream_overlap.h2d_overlap_loop
    """
    import argparse

    p = argparse.ArgumentParser(
        description="CUDA stream DataLoader H2D / forward overlap (plain for-loop)"
    )
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
