# cuda_stream_overlap

用 **CUDA stream** 把 GPU 计算和 PCIe 拷贝叠在一起。

| 文件 | 方向 | 重叠的是 |
|------|------|----------|
| [`d2h_overlap.py`](d2h_overlap.py) | GPU → CPU | 本块 FFN ∥ 上一块 D2H |
| [`h2d_overlap.py`](h2d_overlap.py) | CPU → GPU（DataLoader） | Prefetcher 迭代器版 |
| [`h2d_overlap_loop.py`](h2d_overlap_loop.py) | CPU → GPU（DataLoader） | 普通 for + prev-handle，更直观 |

---

## D2H：计算与拷回重叠

实现：[`d2h_overlap.py`](d2h_overlap.py)

- `forward_sync`：默认流上「算完一块立刻阻塞拷回」
- `forward_overlap`：FFN 留在 default stream，另开一条 copy stream 做异步 D2H

### 朴素写法

```text
[FFN0][====D2H0====][FFN1][====D2H1====][FFN2][====D2H2====]
```

计算和拷贝走同一条默认流，拷贝引擎空等，计算引擎也空等。

### 重叠写法

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

### 为什么计算可以走 default、copy 不能？反过来行不行？

可以反过来，但 **计算和拷贝不能待在同一条流上**。「copy 不能走 default」不是 CUDA 禁止 copy 用 default，而是：**别和 FFN 挤在同一条 default 里**。

同一条流是 FIFO，FFN 和 D2H 排在一起一定串行。拆开之后，两种方向在 side stream 是 **non-blocking**（`torch.cuda.Stream()` 默认就是）时都能叠：

```text
# 本仓库：计算走 default
default: [FFN0][FFN1][FFN2]
copy:          [D2H0][D2H1]

# 反过来：拷贝走 default，计算新开一条
compute: [FFN0][FFN1][FFN2]
default:       [D2H0][D2H1]
```

反过来大致是：

```python
compute_stream = torch.cuda.Stream()  # 必须 non-blocking

with torch.cuda.stream(compute_stream):
    y_i = module(x[lo:hi]).contiguous()
    done.record(compute_stream)

torch.cuda.current_stream().wait_event(done)
host_out[lo:hi].copy_(y_i, non_blocking=True)  # D2H 排进 default
```

更推荐「计算 default、拷贝 side」，因为 PyTorch 的 `Linear` / `GELU` 会跑到 **当前流**（通常是 default）。计算留在 default，不用把整段 FFN 包进 `with torch.cuda.stream(...)`。D2H 只有几次 `copy_`，隔离更干净。若 copy 走 default，后面任何忘了切流的 kernel 都会和拷贝排成一队。

硬约束：多出来的那条流必须是 **non-blocking**。如果是会和 default 隐式互等的 blocking stream，不管谁走 default，`FFN_{i+1}` 都会等完 `D2H_i` 才开工。

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

### `record_stream` 是什么意思？

`record_stream` 跟 `Event.record()` 不是一回事。它是告诉 **PyTorch 显存分配器**：这块内存还会被另一条 stream 用，别急着回收。

tensor 在 `copy_stream` 上 `to(device)` 出来，分配器默认认为「copy stream 上的活干完，这块显存就可以给别的 tensor」。但后面 `model(x)` / `(pred - y)` 跑在 **default stream** 上，可能还在读。

若不 `record_stream`，下一轮在 copy stream 上再 `to(device)`，缓存分配器可能把旧 buffer 复用掉，forward 会读到被覆盖的数据。

```text
copy stream:   分配并填好 x ──► 分配器以为 x 用完了 ──► 下一批可能占同一块显存
default:                         wait 完之后 model(x) 还在读
                                 ↑ 必须告诉分配器：default 也在用
```

`x.record_stream(default)`：把 default stream 记到这块 storage 的「还在用的 stream 列表」里。分配器会等到 **default 上用过这块内存的 kernel 都结束**，才回收。

| | 作用 |
|--|------|
| `wait_stream` / `wait_event` | GPU 执行顺序：forward 不能早于 H2D |
| `record_stream` | 显存寿命：这块 buffer 不能在消费流用完之前被复用 |

跨 stream 生产、跨 stream 消费时几乎都要成对出现。如果 `to(device)` 和 `model(x)` 都在同一条流上，分配器自己能看出来，就不必 `record_stream`。

### 为什么要 `prev_y.record_stream(default_stream)`？

`prev_y` 是在 copy stream 上 `to(device)` 出来的，但真正读它的是 default 上的：

```python
acc = acc + (pred - prev_y).square().mean()
```

copy stream 把 H2D 做完，分配器就认为 `prev_y` 的显存可以给下一批 `y = y_cpu.to(...)`。而 Python 下一轮马上：

```python
prev_x, prev_y, prev_done = x, y, done   # 丢掉对「上一块 y」的引用
```

旧 `prev_y` 引用计数掉到 0 → 显存进缓存池。没有 `record_stream(default)`，分配器不知道 default 上的减法还在读，下一批 H2D 可能写到同一地址。

- `wait_event(prev_done)`：保证 **算的时候拷已经结束**（执行顺序）
- `prev_y.record_stream(default)`：保证 **算完之前这块显存不被下一批抢走**（内存寿命）

`prev_x` 同理，它被 `model(prev_x)` 读。`y` 虽然没进网络，但进了 loss，一样要 record。

---

## DataLoader H2D 与 forward 重叠

训练里更常见的是反方向：CPU 上的 batch 要 `to(device)`，再 `model.forward`。

实现：推荐看 [`h2d_overlap_loop.py`](h2d_overlap_loop.py)（普通 `for` + prev-handle）。[`h2d_overlap.py`](h2d_overlap.py) 是同一套逻辑的 Prefetcher 封装。

### 朴素写法

```python
for x, y in loader:
    x, y = x.to(device), y.to(device)   # 阻塞 H2D，走 default stream
    pred = model(x)
```

```text
[H2D0][Fwd0][H2D1][Fwd1][H2D2][Fwd2]
```

H2D 和 forward 同一条 default 流，拷贝时计算空等。

### 重叠写法：prefetch 下一批

计算仍走 **default stream**，H2D 放到一条 non-blocking `copy_stream`。先把 batch 0 拷上去；之后每轮 **先等当前批拷完再 forward，同时发出下一批 H2D**：

```text
copy:    [H2D0][H2D1][H2D2]
default:       [Fwd0][Fwd1][Fwd2]
                    ↑
              Fwd0 ∥ H2D1
```

| 步骤 | 为何合法 |
|------|----------|
| `Fwd_i` ∥ `H2D_{i+1}` | forward 只用已经 wait 过的当前批；下一批写的是另一块 device 内存 |
| `Fwd_i` 前 `current_stream().wait_stream(copy)` | 不能读尚未拷完的 `x` |
| `DataLoader(pin_memory=True)` | 只有 pinned 源才能真正异步 H2D |
| `.to(device, non_blocking=True)` | 否则 CPU 会卡在这次拷贝上，发不出「下一批 H2D + 当前 forward」 |
| `x.record_stream(default)` | `x` 在 copy stream 上分配/填数，却在 default 上被 model 读；不 record 的话缓存分配器可能过早回收 |

核心循环（prev-handle，和 D2H / MoE 同一套）：

```python
copy_stream = torch.cuda.Stream()
prev_x, prev_y, prev_done = None, None, None

for x_cpu, y_cpu in loader:
    with torch.cuda.stream(copy_stream):
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        done = torch.cuda.Event()
        done.record(copy_stream)          # 当前批 H2D 的完工旗

    if prev_x is not None:
        default.wait_event(prev_done)     # 只等上一批，不等当前刚发出的 H2D
        prev_x.record_stream(default)
        pred = model(prev_x)              # Fwd_{i-1} ∥ H2D_i
    prev_x, prev_y, prev_done = x, y, done

default.wait_event(prev_done)             # 收尾最后一批
pred = model(prev_x)
```

这里必须用 `wait_event(prev_done)`，不能 `wait_stream(copy_stream)`：循环里已经把当前批 H2D 排进 copy stream 了，`wait_stream` 会把当前批也等完，`Fwd_{i-1}` 就叠不上 `H2D_i`。

Prefetcher 版是把「先 wait 再发下一批」藏进 `next()`，语义一样，只是拆开了。

和 D2H 例子的对称关系：

| | D2H（算完再拷回） | H2D prefetch（先拷再算） |
|--|-------------------|--------------------------|
| 先发生的 | FFN_i | H2D_i |
| 重叠的 | FFN_{i+1} ∥ D2H_i | Fwd_i ∥ H2D_{i+1} |
| wait 的位置 | copy 等 compute 的 event | **compute 等 copy 的 stream** |
| 收尾 | 读 host 前 `copy_stream.synchronize()` | 下一轮 `wait_stream` 会拦住；epoch 结束不必再 sync 才能做 GPU loss |

`num_workers>0` 叠的是 **CPU 读盘/解码** 和 GPU，跟这条 PCIe H2D 流水是两层。本例用 `num_workers=0`，只看 DMA 重叠。

### 和 D2H 一样：copy 不要走 default

`.to(device)` 若也排进 default，就会和 `model(x)` 排成一队。必须 side stream + `non_blocking=True` + pinned。

也可以计算新开一条、H2D 走 default，但不推荐：DataLoader 之后几乎所有 PyTorch op 都在当前/default 流上，H2D 放 default 很容易被后续 kernel 插队打串。

---

## 运行

需要 NVIDIA GPU + CUDA 版 PyTorch：

```bash
cd /Users/saboxu/Downloads/codes/overlap_optimize
export PYTHONPATH=.
python -m cuda_stream_overlap.d2h_overlap
python -m cuda_stream_overlap.h2d_overlap_loop
```

D2H 可调 chunk 数和计算量；H2D 可调 batch / dim，让拷贝时间和 forward 接近时加速最明显：

```bash
python -m cuda_stream_overlap.d2h_overlap --batch 32768 --dim 4096 --n-chunks 8 --n-layers 6
python -m cuda_stream_overlap.h2d_overlap_loop --n-batches 32 --batch 8192 --dim 4096 --n-layers 6
```

用 Nsight Systems 看两条 stream 是否交叠：

```bash
nsys profile -t cuda --stats=true python -m cuda_stream_overlap.h2d_overlap_loop
```
