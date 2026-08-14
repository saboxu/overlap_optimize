# communicate_overlap

MoE `all_gather` 与 DDP 梯度 `all_reduce` 的通信计算重叠实现。

完整说明和运行方式见仓库根目录 [README.md](../README.md)。

| 文件 | 内容 |
|------|------|
| [`moe_forward.py`](moe_forward.py) | `forward_sync` / `forward_overlap` |
| [`grad_allreduce_overlap.py`](grad_allreduce_overlap.py) | `AsyncGradAllReducer` / `DistributedDataParallelOverlap` |
| [`test_grad_allreduce.py`](test_grad_allreduce.py) | DDP wrapper 冒烟测试 |
