"""Smoke test for AsyncGradAllReducer / DistributedDataParallelOverlap."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from communicate_overlap.grad_allreduce_overlap import DistributedDataParallelOverlap


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="gloo")

    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
    model = DistributedDataParallelOverlap(base)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(4, 16) + rank
    opt.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    model.backward(loss)
    model.synchronize()

    g = next(model.parameters()).grad
    assert g is not None
    gathered = [torch.empty_like(g) for _ in range(world)]
    dist.all_gather(gathered, g)
    if rank == 0:
        ok = all(torch.allclose(gathered[0], t, atol=1e-5) for t in gathered[1:])
        print(f"ddp_overlap_ok={ok} grad_norm={g.norm().item():.4f}")
        assert ok

    opt.zero_grad(set_to_none=True)
    with model.no_sync():
        model.backward(model(x).square().mean())
    model.backward(model(x).square().mean())
    model.synchronize()
    g2 = next(model.parameters()).grad
    assert g2 is not None
    gathered2 = [torch.empty_like(g2) for _ in range(world)]
    dist.all_gather(gathered2, g2)
    if rank == 0:
        ok2 = all(torch.allclose(gathered2[0], t, atol=1e-5) for t in gathered2[1:])
        print(f"accum_ok={ok2}")
        assert ok2

    model.reducer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
