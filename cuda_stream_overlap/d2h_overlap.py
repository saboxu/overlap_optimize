"""
Overlap GPU compute with device-to-host copies using CUDA streams.

Naive path: for each chunk, FFN on GPU then blocking D2H.
Overlap path: FFN stays on the default stream; a side copy stream DMAs
chunk i while the default stream computes chunk i+1.

Requires a CUDA GPU. Pinned host buffers are mandatory for async D2H.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class ChunkFFN(nn.Module):
    """GPU compute workload: a few Linear + GELU layers, last dim unchanged."""

    def __init__(self, dim: int, hidden: int | None = None, n_layers: int = 4):
        super().__init__()
        hidden = hidden or dim * 2
        layers: List[nn.Module] = []
        for i in range(n_layers):
            in_f = dim if i == 0 else hidden
            out_f = dim if i == n_layers - 1 else hidden
            layers.append(nn.Linear(in_f, out_f))
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _chunk_ranges(n: int, n_chunks: int) -> List[Tuple[int, int]]:
    if n_chunks <= 0 or n % n_chunks != 0:
        raise ValueError(f"n={n} must be divisible by n_chunks={n_chunks}")
    step = n // n_chunks
    return [(i * step, (i + 1) * step) for i in range(n_chunks)]


def forward_sync(
    x: torch.Tensor,
    module: nn.Module,
    n_chunks: int,
    host_out: torch.Tensor,
) -> torch.Tensor:
    """
    Blocking path on the default stream: compute chunk -> D2H -> next chunk.

    x: [N, D] CUDA
    host_out: [N, D] CPU (pinned or pageable)
    """
    ranges = _chunk_ranges(x.size(0), n_chunks)
    for lo, hi in ranges:
        y_i = module(x[lo:hi])
        host_out[lo:hi].copy_(y_i)
    return host_out


def forward_overlap(
    x: torch.Tensor,
    module: nn.Module,
    n_chunks: int,
    host_out: torch.Tensor,
) -> torch.Tensor:
    """
    Pipeline using the default stream for compute and one extra copy stream.

      default stream: FFN(chunk i)
      copy_stream:    D2H(chunk i) after that FFN, overlapping FFN(chunk i+1)

    This overlaps because ``torch.cuda.Stream()`` is a non-blocking stream
    (cudaStreamNonBlocking): default-stream kernels do not implicitly wait
    for it. Putting D2H on the default stream as well would serialize.

    host_out must be pin_memory=True, otherwise copy_ falls back to sync DMA.
    Device outputs are kept alive until the copy stream finishes.
    """
    if not host_out.is_pinned():
        raise ValueError("host_out must be pinned for async D2H (pin_memory=True)")

    copy_stream = torch.cuda.Stream()
    ranges = _chunk_ranges(x.size(0), n_chunks)

    # Keep each y_i live until D2H of that chunk completes.
    device_hold: List[torch.Tensor] = []

    for lo, hi in ranges:
        y_i = module(x[lo:hi]).contiguous()
        done = torch.cuda.Event()
        done.record()  # default stream
        device_hold.append(y_i)

        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(done)
            host_out[lo:hi].copy_(y_i, non_blocking=True)

    # Copy stream is non-blocking vs default; wait before the caller reads host_out.
    copy_stream.synchronize()
    return host_out


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
    batch: int,
    dim: int,
    n_chunks: int,
    n_layers: int,
    warmup: int,
    iters: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. This example uses cuda.Stream / cuda.Event / pinned D2H."
        )

    device = torch.device("cuda")
    module = ChunkFFN(dim, n_layers=n_layers).to(device).eval()

    torch.manual_seed(0)
    x = torch.randn(batch, dim, device=device)

    host_sync = torch.empty(batch, dim, pin_memory=True)
    host_overlap = torch.empty(batch, dim, pin_memory=True)

    with torch.inference_mode():
        forward_sync(x, module, n_chunks, host_sync)
        forward_overlap(x, module, n_chunks, host_overlap)

        if not torch.allclose(host_sync, host_overlap, atol=1e-4, rtol=1e-3):
            max_diff = (host_sync - host_overlap).abs().max().item()
            raise RuntimeError(f"sync vs overlap mismatch, max_abs={max_diff}")

        def _sync():
            forward_sync(x, module, n_chunks, host_sync)

        def _overlap():
            forward_overlap(x, module, n_chunks, host_overlap)

        for _ in range(warmup):
            _sync()
            _overlap()

        sync_ms = min(_time_ms(_sync) for _ in range(iters))
        overlap_ms = min(_time_ms(_overlap) for _ in range(iters))

    nbytes = host_sync.nbytes
    print(
        f"ok  x={tuple(x.shape)}  n_chunks={n_chunks}  n_layers={n_layers}  "
        f"d2h={nbytes / 1024**2:.1f} MiB"
    )
    print(f"sync     {sync_ms:.3f} ms")
    print(f"overlap  {overlap_ms:.3f} ms")
    print(f"speedup  {sync_ms / overlap_ms:.2f}x")


def main() -> None:
    """
    Launch example:

      python -m cuda_stream_overlap.d2h_overlap
    """
    import argparse

    p = argparse.ArgumentParser(description="CUDA stream D2H / compute overlap")
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--dim", type=int, default=2048)
    p.add_argument("--n-chunks", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    _run(
        batch=args.batch,
        dim=args.dim,
        n_chunks=args.n_chunks,
        n_layers=args.n_layers,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
