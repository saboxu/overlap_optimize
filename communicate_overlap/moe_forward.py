"""
MoE-style forward: LocalFFN -> all_gather(head_dim) -> SharedFFN on expert chunks.

Interview pattern: overlap all_gather communication with compute via async_op=True.
Tensor layout: [batch, num_experts, head_dim]
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class LocalExpertFFN(nn.Module):
    """Per-rank FFN on local head_dim shard. Keeps last dim size unchanged."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E_chunk, D_local]
        return self.net(x)


class SharedFFN(nn.Module):
    """FFN after full head_dim is gathered."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E_chunk, D_full]
        return self.net(x)


def forward_sync(
    x: torch.Tensor,
    local_ffn: Callable[[torch.Tensor], torch.Tensor],
    shared_ffn: Callable[[torch.Tensor], torch.Tensor],
    group: Optional[dist.ProcessGroup],
    n_chunks: int,
) -> torch.Tensor:
    """
    Naive path: for each expert chunk, LocalFFN -> blocking all_gather -> SharedFFN.

    x: [B, E, D_local]
    returns: [B, E, D_full]
    """
    world = dist.get_world_size(group)
    outs: List[torch.Tensor] = []

    for x_i in x.chunk(n_chunks, dim=1):
        y_i = local_ffn(x_i)  # [B, E/n, D_local]

        gather_list = [torch.empty_like(y_i) for _ in range(world)]
        dist.all_gather(gather_list, y_i.contiguous(), group=group)

        full = torch.cat(gather_list, dim=-1)  # [B, E/n, D_full]
        outs.append(shared_ffn(full))

    return torch.cat(outs, dim=1)


def forward_overlap(
    x: torch.Tensor,
    local_ffn: Callable[[torch.Tensor], torch.Tensor],
    shared_ffn: Callable[[torch.Tensor], torch.Tensor],
    group: Optional[dist.ProcessGroup],
    n_chunks: int,
) -> torch.Tensor:
    """
    Pipelined path: async all_gather + prev-handle template.

    Timeline (chunk i):
      Local_i  overlaps with AG_{i-1}
      Shared_{i-1} overlaps with AG_i

    x: [B, E, D_local]
    returns: [B, E, D_full]
    """
    world = dist.get_world_size(group)
    outs: List[torch.Tensor] = []

    prev_handle = None
    prev_list: Optional[List[torch.Tensor]] = None

    for x_i in x.chunk(n_chunks, dim=1):
        # (1) Compute before comm — can overlap with previous all_gather
        y_i = local_ffn(x_i)

        # (2) Launch async all_gather on head_dim; returns immediately
        gather_list = [torch.empty_like(y_i) for _ in range(world)]
        handle = dist.all_gather(
            gather_list,
            y_i.contiguous(),
            group=group,
            async_op=True,
        )

        # (3) Finish previous chunk's post-comm work — overlaps with current AG
        if prev_handle is not None:
            prev_handle.wait()
            assert prev_list is not None
            full = torch.cat(prev_list, dim=-1)
            outs.append(shared_ffn(full))

        prev_handle, prev_list = handle, gather_list

    # (4) Drain the last in-flight all_gather
    assert prev_handle is not None and prev_list is not None
    prev_handle.wait()
    full = torch.cat(prev_list, dim=-1)
    outs.append(shared_ffn(full))

    return torch.cat(outs, dim=1)


def _run_one_rank(
    rank: int,
    world_size: int,
    local_rank: int,
    backend: str,
    batch: int,
    num_experts: int,
    d_local: int,
    n_chunks: int,
) -> None:
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    group = dist.group.WORLD
    d_full = d_local * world_size

    local_ffn = LocalExpertFFN(d_local).to(device)
    shared_ffn = SharedFFN(d_full).to(device)

    # Same input seed on all ranks only for demo correctness check of shapes;
    # real TP would shard a global tensor along head_dim.
    torch.manual_seed(0)
    x = torch.randn(batch, num_experts, d_local, device=device)

    y_sync = forward_sync(x, local_ffn, shared_ffn, group, n_chunks)
    y_overlap = forward_overlap(x, local_ffn, shared_ffn, group, n_chunks)

    if not torch.allclose(y_sync, y_overlap, atol=1e-5, rtol=1e-4):
        raise RuntimeError(f"rank{rank}: sync vs overlap mismatch")

    if rank == 0:
        print(
            f"ok  x={tuple(x.shape)}  out={tuple(y_sync.shape)}  "
            f"world={world_size}  n_chunks={n_chunks}"
        )

    dist.destroy_process_group()


def main() -> None:
    """
    Launch example (2 ranks):

      torchrun --standalone --nproc_per_node=2 -m communicate_overlap.moe_forward

    CPU gloo also works for logic check; NCCL needed for real GPU overlap.
    """
    import os

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    backend = "nccl" if torch.cuda.is_available() and world_size > 1 else "gloo"
    if backend == "nccl":
        # torchrun sets LOCAL_RANK; map to that GPU
        os.environ.setdefault("LOCAL_RANK", str(local_rank))

    _run_one_rank(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=backend,
        batch=2,
        num_experts=8,
        d_local=16,
        n_chunks=4,
    )


if __name__ == "__main__":
    main()
