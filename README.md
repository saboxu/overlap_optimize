# overlap_optimize

**通信与计算重叠** 的最小实现，覆盖四类常见模式：

1. **MoE forward**：`LocalFFN` 之后对 `head_dim` 做 `all_gather`，与下一段计算流水重叠
2. **DDP backward**：参数梯度就绪后立刻 `async all_reduce`，与更浅层的 backward 重叠
3. **CUDA D2H**：GPU 算下一块的同时，copy engine 把上一块拷回 pinned host
4. **CUDA H2D**：DataLoader 把下一批 `to(device)` 的同时，default stream 做当前批 forward

集合通信用 `torch.distributed.*(..., async_op=True)` + `handle.wait()`；PCIe 拷贝用 CUDA `Stream` / `Event` + `non_blocking=True`。

---

## 目录

```text
overlap_optimize/
├── communicate_overlap/
│   ├── moe_forward.py              # MoE all_gather 同步版 / 异步流水版
│   ├── grad_allreduce_overlap.py   # 梯度 all_reduce 与 backward 重叠
│   ├── test_grad_allreduce.py      # DDP wrapper 冒烟测试
│   └── README.md
└── cuda_stream_overlap/
    ├── d2h_overlap.py              # CUDA stream：GPU 计算 ∥ D2H
    ├── h2d_overlap.py              # CUDA stream：DataLoader H2D ∥ forward（Prefetcher）
    ├── h2d_overlap_loop.py         # 同上，普通 for + prev-handle
    └── README.md
```

---

## 环境

- Python 3.10+
- PyTorch（集合通信需 `torch.distributed`；D2H / H2D 重叠需 CUDA 版 PyTorch + NVIDIA GPU）
- 集合通信逻辑检查：CPU + `gloo` 即可
- 集合通信真实重叠：多卡 GPU + `nccl`
- D2H / H2D 重叠：单卡 GPU 即可；host buffer 必须 `pin_memory=True`

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

| | MoE all_gather | DDP all_reduce | CUDA D2H |
|--|----------------|----------------|----------|
| 通信前计算 | LocalFFN | 本层 backward 产出 grad | 本 chunk FFN |
| 异步通信 | `all_gather(async)` | `all_reduce(grad, async)` | `copy_(non_blocking=True)` |
| 中间 wait prev？ | **需要**（Shared 马上用） | **通常不需要**（step 才用） | Event 等的是 **当前块** FFN，不是 prev |
| 收尾 wait | 最后一段 Shared 前 | `synchronize()` / `optimizer.step()` 前 | 读 host 前 `cuda.synchronize()` |

### 运行测试

```bash
torchrun --standalone --nproc_per_node=2 -m communicate_overlap.test_grad_allreduce
```

会检查各 rank 同步后的梯度是否一致，以及 `no_sync` 累积路径。

---

## 3. CUDA stream：计算与 D2H 重叠

实现见 [`cuda_stream_overlap/d2h_overlap.py`](cuda_stream_overlap/d2h_overlap.py)。

把输入 `[N, D]` 沿 batch 切成若干 chunk。每块先在 GPU 上过 FFN，再拷回 CPU。

朴素路径（默认流，串行）：

```text
[FFN0][====D2H0====][FFN1][====D2H1====][FFN2][====D2H2====]
```

重叠路径（default stream 算 FFN，另开 copy stream 做 D2H）：

```text
default: [FFN0][FFN1][FFN2]
copy:          [====D2H0====][====D2H1====][====D2H2====]
                    ↑
              FFN1 ∥ D2H0
```

| 约束 | 原因 |
|------|------|
| `host_out` 必须 pinned | 只有页锁定内存才能异步 DMA；普通 `.cpu()` 会变成同步拷贝 |
| D2H 不要放回 default stream | 和 FFN 同一条流就会串行；copy 必须是 side stream |
| `torch.cuda.Stream()` 必须是 non-blocking | 这样 default 上的下一块 FFN 不会隐式等 D2H |
| `copy_stream.wait_event(FFN_i)` | 不能拷尚未写完的 device 输出 |
| 收尾 `copy_stream.synchronize()` | 读 host 结果前必须等 DMA 完成 |
| 暂存 `y_i` 直到拷贝结束 | 否则 device buffer 可能被释放或复用 |

```python
copy_stream = torch.cuda.Stream()

for lo, hi in ranges:
    y_i = module(x[lo:hi]).contiguous()
    done = torch.cuda.Event()
    done.record()
    device_hold.append(y_i)

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(done)
        host_out[lo:hi].copy_(y_i, non_blocking=True)

copy_stream.synchronize()
```

### 运行

```bash
python -m cuda_stream_overlap.d2h_overlap
```

脚本会对比 sync / overlap 的 host 输出，并打印耗时。FFN 时间与 D2H 时间接近时加速最明显，可用 `--n-chunks`、`--n-layers`、`--batch` 调节。

---

## 4. CUDA stream：DataLoader H2D 与 forward 重叠

实现见 [`cuda_stream_overlap/h2d_overlap_loop.py`](cuda_stream_overlap/h2d_overlap_loop.py)（普通 `for`）。Prefetcher 封装在 [`h2d_overlap.py`](cuda_stream_overlap/h2d_overlap.py)。

朴素路径：`x.to(device)` 阻塞后再 `model(x)`。重叠路径：copy stream 发当前批 H2D，default stream 等 **上一批** 到齐再 forward。

```text
copy:    [H2D0][H2D1][H2D2]
default:       [Fwd0][Fwd1][Fwd2]
                    ↑
              Fwd0 ∥ H2D1
```

要点：`DataLoader(pin_memory=True)`、`.to(device, non_blocking=True)`、forward 前 `wait_event(上一批 H2D)`（不要 `wait_stream` 整条 copy 流），以及 `record_stream`。

```bash
python -m cuda_stream_overlap.h2d_overlap_loop
```
