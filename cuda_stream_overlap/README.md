# cuda_stream_overlap

用 **CUDA stream** 把 GPU 计算和 **D2H**（device → host）拷贝叠在一起。

实现：[`d2h_overlap.py`](d2h_overlap.py)

- `forward_sync`：默认流上「算完一块立刻阻塞拷回」
- `forward_overlap`：FFN 留在 default stream，另开一条 copy stream 做异步 D2H

---

## 朴素写法

```text
[FFN0][====D2H0====][FFN1][====D2H1====][FFN2][====D2H2====]
```

计算和拷贝走同一条默认流，拷贝引擎空等，计算引擎也空等。

---

## 重叠写法

GPU 有独立的 **compute engine** 和 **copy engine**。FFN 留在 **default stream**，D2H 放到一条 `cudaStreamNonBlocking` 的 `copy_stream` 上，用 `Event` 只同步「这一块算完才能拷」：

```text
default: [FFN0][FFN1][FFN2]
copy:          [====D2H0====][====D2H1====][====D2H2====]
                    ↑
              FFN1 ∥ D2H0
```

| 步骤 | 为何合法 |
|------|----------|
| `FFN_i` ∥ `D2H_{i-1}` | 拷的是上一块已经 `record` 过的 buffer，不读当前块 |
| `D2H_i` 开始前 `wait_event(FFN_i)` | 不能拷尚未写完的 device 输出 |
| `host_out` 用 `pin_memory=True` | 只有页锁定内存才能真正异步 DMA；普通 `cpu()` 会静默变成同步拷贝 |
| copy 用 non-blocking stream | `torch.cuda.Stream()` 默认 `cudaStreamNonBlocking`，default stream 不会隐式等它；否则下一块 FFN 会卡住等 D2H |

核心循环：

```python
copy_stream = torch.cuda.Stream()  # non-blocking side stream

for lo, hi in ranges:
    y_i = module(x[lo:hi]).contiguous()   # default stream
    done = torch.cuda.Event()
    done.record()                         # 插在 default stream 队尾
    device_hold.append(y_i)

    with torch.cuda.stream(copy_stream):
        copy_stream.wait_event(done)
        host_out[lo:hi].copy_(y_i, non_blocking=True)

copy_stream.synchronize()                 # 读 host_out 之前收尾
```

少开一条 compute stream 是可以的：计算本来就在 default stream 上。不能少的是 copy 那条——D2H 如果也丢回 default stream，就和 FFN 排成一队，重叠没了。

---

## 和集合通信重叠的差别

| | MoE all_gather / DDP all_reduce | 本例 D2H |
|--|--------------------------------|----------|
| 通信通道 | NCCL / PCIe / NVLink（卡间） | CUDA copy engine（卡 → CPU） |
| 异步 API | `async_op=True` | `Stream` + `copy_(..., non_blocking=True)` |
| 同步原语 | `handle.wait()` | `Event` + `stream.wait_event` / `copy_stream.synchronize()` |
| 额外约束 | process group | **pinned host memory** |

---

## 常见问题

### `copy_stream.wait_event(done)` 是什么意思？

copy stream **先别做后面的 D2H**，等到 `done` 这面旗在 compute / default stream 上倒下再开始。

```python
done.record()                    # FFN 排进 default 后，在队尾插旗（不是立刻完成）
copy_stream.wait_event(done)     # copy stream：看见旗倒下再拷
host_out[lo:hi].copy_(y_i, ...)  # 这时 y_i 已经写完
```

`record` 只是把标记挂到那条流当前队列末尾，FFN kernel 跑完旗才倒下。

`wait_event` **不是 CPU 卡住**。CPU 把「等 event + 拷贝」排进 copy stream 后，马上继续 for 循环去排下一块 FFN。真正等的是 GPU 上的 copy engine。

约束的是「同一块算完才能拷」，不是「整条 compute 流都停下来」。下一块 FFN 仍然可以和这一块 D2H 并行。

没有这句，copy stream 可能在 FFN 还没写完 `y_i` 时就开始拷。

### `non_blocking=True` 一定要加吗？

这个重叠例子里 **要加**。不加的话，跨 chunk 的流水线基本叠不上。

`copy_` 默认 `non_blocking=False`。即使写在 `with torch.cuda.stream(copy_stream)` 里，PyTorch 仍会等这次 D2H **完成之后** 才把 CPU 还给你，循环变成：

```text
排 FFN0 → 死等 D2H0 拷完 → 再排 FFN1 → 死等 D2H1 ...
```

`FFN0` 和它自己的 `D2H0` 也许还能前后衔接，但 `FFN1` 根本还没提交，没法跟 `D2H0` 并行。

`non_blocking=True` 对应 `cudaMemcpyAsync`：CPU 只把拷贝丢进 copy stream，立刻去排下一块 FFN。

光加 flag 不够：`host_out` 必须是 pinned。目的地是普通 pageable 内存时，会退化成同步拷贝。

和 CUDA C 的差别：C 里 `cudaMemcpyAsync(..., stream)` 本身就是异步的；PyTorch 里 **stream 上下文不会自动让 `copy_` 异步**，要靠 `non_blocking=True` 才走 async DMA。

### `torch.cuda.synchronize()` / `copy_stream.synchronize()` 必须加吗？

**读 `host_out` 之前必须等拷贝结束。** `synchronize` 不是重叠本身需要的同步，块内顺序已经由 `wait_event` 保证。

`copy_(..., non_blocking=True)` 只是把 D2H 丢进 copy stream。函数若立刻 `return host_out`，DMA 往往还没写完，这时 `allclose` / 打印 / 存盘会读到旧数据。

循环中间不要 `synchronize()`，否则每块都把 CPU/GPU 栅栏拉齐，流水线会断。

本例收尾用 `copy_stream.synchronize()` 就够：这条流已经 `wait_event` 过每一块 FFN，它空了，计算和 D2H 都结束了。`torch.cuda.synchronize()` 更粗，会等当前设备上所有 stream。

`torch.cuda.Stream()` 默认是 **non-blocking stream**，不会和 default stream 隐式互等，不能指望「函数返回后 default stream 会自动帮你等完」。

### 能用 default stream、少开一条 compute stream 吗？

可以。计算直接走 default stream，只再开一条 copy stream。本仓库就是这样写的。

能重叠，是因为 `torch.cuda.Stream()` 创建的是 **`cudaStreamNonBlocking`**：default 上的下一块 FFN **不会**隐式等这条 copy stream。于是 `FFN1` 和 `D2H0` 可以同时跑。

少不掉的是 copy 那条。D2H 如果也丢回 default，就和 FFN 排在同一队列里，一定串行。

### CUDA 里「一条 stream wait 另一条 stream」是什么意思？

CUDA 没有「一条 stream 一直盯着另一条 stream 直到它彻底结束」这种原语。所谓 stream wait 另一条 stream，是：

**当前这条流后面的活先别开；等另一条流上已经排进去的某段工作做完，再开始。**

等的是 GPU 自己，CPU 不会卡住。两条流不会合并成一条，只是给当前流加了一个开工条件。

常见两种粒度：

| API | 等的范围 |
|-----|----------|
| `wait_event(done)` | 只等另一条流上 **record 那一时刻之前** 的活 |
| `wait_stream(other)` | 在 `other` **当前已经排队** 的活后面插一个 event，再 wait。之后新排到 `other` 上的活不等 |

和 `synchronize()` 的差别：`wait_event` / `wait_stream` 是 GPU 上两条流之间的依赖，CPU 继续往下排；`cuda.synchronize()` / `stream.synchronize()` 是 CPU 停住，直到 GPU 干完。

### `wait_stream` 怎么用？是另一条流结束才能走当前流吗？

`wait_stream` 是 PyTorch 对「用 Event 等另一条流」的封装：

```python
copy_stream.wait_stream(other)
```

等价于：

```python
evt = torch.cuda.Event()
evt.record(other)            # 插在 other 当前队尾
copy_stream.wait_event(evt)  # copy 上之后的 op 都等这面旗
```

理解要对齐这句话：**等的不是「那条流彻底结束」，而是「调用 `wait_stream` 那一刻，对面已经排进去的 op」。** 这时刻之后再往另一条流里新排的活，不等。

```text
时刻 T0（CPU 调用 wait_stream）:
  other:   [A][B]          ← 已经排队，要等完
  self:    wait ──► [C][D] ← C、D 必须等 A、B 做完

时刻 T0 之后又往 other 上排了 E:
  other:   [A][B][E]
  self:          [C][D]

E 不必先结束，C 可以和 E 重叠。
```

本例 FFN 都在同一条 default 上（FIFO），`wait_stream(default)` 在排完 `FFN_i` 之后调用，等整条流当前队尾 = 等 `FFN_i`，和手动 `record` + `wait_event` 等价：

```python
y_i = module(x[lo:hi]).contiguous()
copy_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(copy_stream):
    host_out[lo:hi].copy_(y_i, non_blocking=True)
```

下一轮才排的 `FFN_{i+1}` 不在这次 wait 范围内，仍能和 `D2H_i` 重叠。

要等流 **中间某个点**（例如只等 kernel A、让 B 和拷贝重叠），必须自己 `Event`，`wait_stream` 会把当时已经排队的 A+B 都等完，粒度更粗。

如果真要等另一条流「现在到将来所有活都干完」，那是 `other.synchronize()`，而且那是 **CPU 卡住**。

读 `host_out` 的是 CPU，`current_stream().wait_stream(copy_stream)` 只约束后续 GPU op，不够，还是得 `copy_stream.synchronize()`。

---

## 运行

需要 NVIDIA GPU + CUDA 版 PyTorch：

```bash
cd /Users/saboxu/Downloads/codes/overlap_optimize
export PYTHONPATH=.
python -m cuda_stream_overlap.d2h_overlap
```

可调 chunk 数和计算量，让 FFN 时间和 D2H 时间接近时加速最明显：

```bash
python -m cuda_stream_overlap.d2h_overlap --batch 32768 --dim 4096 --n-chunks 8 --n-layers 6
```

用 Nsight Systems 看两条 stream 是否交叠：

```bash
nsys profile -t cuda --stats=true python -m cuda_stream_overlap.d2h_overlap
```
