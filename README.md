# overlap_optimize

分布式训练里 **通信与计算重叠** 的最小实现，覆盖两类常见模式：

1. **MoE forward**：`LocalFFN` 之后对 `head_dim` 做 `all_gather`，与下一段计算流水重叠
2. **DDP backward**：参数梯度就绪后立刻 `async all_reduce`，与更浅层的 backward 重叠

核心手段都是 `torch.distributed.*(..., async_op=True)` + 在使用结果前 `handle.wait()`。

---

## 目录

```text
overlap_optimize/
└── communicate_overlap/
    ├── moe_forward.py              # MoE all_gather 同步版 / 异步流水版
    ├── grad_allreduce_overlap.py   # 梯度 all_reduce 与 backward 重叠
    ├── test_grad_allreduce.py      # DDP wrapper 冒烟测试
    └── README.md                   # 模块说明
```

---

## 环境

- Python 3.10+
- PyTorch（需 `torch.distributed`）
- 逻辑正确性：CPU + `gloo` 即可
- 真实通信重叠：多卡 GPU + `nccl`

仓库里已有 `.venv` 时：

```bash
source .venv/bin/activate
```

从仓库根目录启动（把当前目录放进 `PYTHONPATH`）：

```bash
cd /Users/saboxu/Downloads/codes/overlap_optimize
export PYTHONPATH=.
```

---

## 1. MoE all_gather 重叠

### 输入与计算流程

每张卡本地输入：

\[
x \in \mathbb{R}^{B \times E \times D_{\text{local}}}
\]

| 符号 | 含义 |
|------|------|
| `B` | batch |
| `E` | `num_experts` |
| `D_local` | 本地 `head_dim` 分片（跨卡切在最后一维） |

朴素 Forward：

1. **LocalExpertFFN**：对 `[B, E, D_local]` 做 MLP
2. **all_gather + concat**：在 `head_dim` 维收集成 `[B, E, D_full]`
3. **chunk**：沿 `num_experts` 切开
4. **SharedFFN**：每段再过 MLP
5. **concat**：沿 expert 维拼回

### 朴素写法的问题

```text
[全部 LocalFFN] → [一次大 all_gather（阻塞）] → [chunk0 FFN][chunk1 FFN]...
```

`all_gather` 是同步墙，后面的 SharedFFN、下一段 LocalFFN 都只能干等。

### 优化：先按 expert 切 micro-chunk

不要「整段 gather 完再 chunk」。对每一段：

1. `LocalFFN`
2. `all_gather(async_op=True)`（仍 gather `head_dim`）
3. 用 **prev-handle 模板** 收回上一段，做 `SharedFFN`
4. 循环结束后再 `wait` 最后一段

```text
E0: [Local0][====AG0====][Shared0]
E1:         [Local1][====AG1====][Shared1]
E2:                 [Local2][====AG2====][Shared2]
                 ↑              ↑
            Local1 ∥ AG0   Shared0 ∥ AG1
```

| 计算 | 重叠的通信 | 为何合法 |
|------|------------|----------|
| `Local_i` | `AG_{i-1}` | Local 只用本地输入，不读 gather 结果 |
| `Shared_{i-1}` | `AG_i` | Shared 只用已 wait 完的 prev；`AG_i` 先发出再算 Shared |
| `Shared_i` | `AG_i` | **不能**，必须先 wait 自己的 gather |

`group` 是切 `head_dim` 的进程组（通常是 TP group），`gather_list` 长度 = `dist.get_world_size(group)`。

### API

| 函数 | 作用 |
|------|------|
| `forward_sync` | 同步正确版：`Local → blocking all_gather → Shared` |
| `forward_overlap` | 异步流水版（prev-handle 模板） |

核心循环：

```python
prev_handle, prev_list = None, None

for x_i in x.chunk(n_chunks, dim=1):
    y_i = local_ffn(x_i)                          # 通信前计算
    gather_list = [torch.empty_like(y_i) for _ in range(world)]
    handle = dist.all_gather(
        gather_list, y_i.contiguous(), group=group, async_op=True
    )                                             # 异步发出
    if prev_handle is not None:
        prev_handle.wait()                        # 等上一段
        outs.append(shared_ffn(torch.cat(prev_list, dim=-1)))
    prev_handle, prev_list = handle, gather_list

prev_handle.wait()                                # 收尾最后一段
outs.append(shared_ffn(torch.cat(prev_list, dim=-1)))
```

### 实现步骤

1. **先写 sync**：`Local → blocking all_gather → Shared`，保证 shape 对
2. **找气泡里能塞什么**：`Local_{i+1}`、`Shared_{i-1}`（不依赖当前 AG）
3. **套 prev 模板**：发当前 AG → wait prev → Shared(prev) → 当前变 prev → 最后收尾
4. **展开 2～3 轮验依赖**：没有对未 wait 的 buffer 做 Shared；最后一段有 wait

要点：先 chunk；每段 Local → async gather → wait **上一段** 做 Shared → 当前变 prev；最后再 wait 一次。

### 怎么分析「有没有重叠」

1. **依赖图**：无数据依赖的算子才可并行
2. **时间预算**：估 \(T_L, T_{AG}, T_S\)，流水线约  
   \(T_L + T_{AG} + T_S + (n-1)\max(T_L, T_{AG}, T_S)\)
3. **Profiler / Nsight**：看 compute stream 与 NCCL all_gather 是否时间交叠；`wait` 前是否有足够长的无关计算

### 运行

```bash
torchrun --standalone --nproc_per_node=2 -m communicate_overlap.moe_forward
```

脚本会对比 `forward_sync` 与 `forward_overlap` 输出是否一致。有多卡 GPU 时走 NCCL，更接近真实重叠场景。

---

## 2. 梯度 all_reduce 与 backward 重叠

实现见 [`communicate_overlap/grad_allreduce_overlap.py`](communicate_overlap/grad_allreduce_overlap.py)。

某层 `grad` 一就绪就发出 `async all_reduce`，autograd 继续算更浅层；`optimizer.step()` 前再 `synchronize()`。

```text
L3: [bwd3][====AR3====]
L2:        [bwd2][====AR2====]   # bwd2 ∥ AR3
wait_all → step
```

### API

| 类 / 函数 | 作用 |
|-----------|------|
| `AsyncGradAllReducer` | 挂在 AccumulateGrad 上，grad 就绪即 bucket + async all_reduce |
| `DistributedDataParallelOverlap` | DDP 风格包装（forward / backward / synchronize / no_sync） |
| `allreduce_grads_async` | backward **之后** 再异步 all_reduce（无计算重叠，仅异步） |

```python
model = DistributedDataParallelOverlap(module, group=group)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(x), y)
model.backward(loss)      # 通信与后续层 backward 重叠
model.synchronize()       # wait + average
optimizer.step()
```

梯度累积：

```python
optimizer.zero_grad(set_to_none=True)
with model.no_sync():
    model.backward(loss_micro1)
model.backward(loss_micro2)
model.synchronize()
optimizer.step()
```

### 和 MoE 模板对比

| | MoE all_gather | DDP all_reduce |
|--|----------------|----------------|
| 通信前计算 | LocalFFN | 本层 backward 产出 grad |
| 异步通信 | `all_gather(async)` | `all_reduce(grad, async)` |
| 中间 wait prev？ | **需要**（Shared 马上用） | **通常不需要**（step 才用） |
| 收尾 wait | 最后一段 Shared 前 | `synchronize()` / `optimizer.step()` 前 |

### 运行测试

```bash
torchrun --standalone --nproc_per_node=2 -m communicate_overlap.test_grad_allreduce
```

会检查各 rank 同步后的梯度是否一致，以及 `no_sync` 累积路径。
