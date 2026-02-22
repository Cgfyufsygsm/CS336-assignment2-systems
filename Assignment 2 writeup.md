# Assignment 2 (systems) Writeup

## Profiling and Benchmarking

### `benchmarking_script`

#### (a) 脚本实现

我实现了一个可参数化的 benchmark 脚本，支持按给定超参数初始化模型、生成随机输入 batch、执行 warm-up 步骤后再进行正式计时，并在每步结束后调用 `torch.cuda.synchronize()` 以避免 CUDA 异步导致的计时偏差。脚本支持 `forward`、`forward-backward` 和 `train-step` 三种模式，其中本题使用 `forward-backward` 来分别统计前向与反向耗时，并将结果输出为 JSON 便于后续汇总。

#### (b) benchmarking

> 实验环境：`NVIDIA H800 PCIe`，`batch_size=4`，`warm-up=5`，`measure=10`
>
> 单位：ms（均值 ± 标准差）

| Model Size | Context |       Forward |       Backward |
| ---------- | ------: | ------------: | -------------: |
| small      |     128 |  21.57 ± 0.13 |   23.02 ± 1.06 |
| medium     |     128 |  45.03 ± 1.10 |   46.51 ± 0.47 |
| large      |     128 |  64.98 ± 0.69 |   93.93 ± 0.05 |
| xl         |     128 |  87.18 ± 1.23 |  175.88 ± 0.18 |
| 2.7b       |     128 | 119.78 ± 0.71 | 264.10 ± 0.75 |

对于 context 变化的补充（同样 `warm-up=5`, `measure=10`）：

| Model Size | Context=256 (F/B) | Context=512 (F/B) |
| ---------- | ----------------: | ----------------: |
| small      |     21.86 / 26.00 |     27.24 / 54.23 |
| medium     |     44.37 / 81.12 |    84.28 / 164.97 |
| large      |    86.51 / 170.95 |   161.45 / 342.85 |
| xl         |   152.54 / 321.49 |   326.48 / 665.57 |
| 2.7b       |   232.57 / 487.16 |  490.07 / 1003.19 |

结论：随着模型规模和 context length 增大，forward/backward 耗时都显著上升；在小模型上 backward 仅略高于 forward，但在大模型（尤其 `xl`、`2.7b`）上 backward 明显更重。整体波动较小，大多数配置的标准差维持在低毫秒量级。

#### (c) warm-up

> 单位：ms（均值 ± 标准差）
> 实验环境：5090

| Model Size | Warm-up | Forward | Backward |
|---|---:|---:|---:|
| small | 0 | 71.46 ± 138.07 | 30.60 ± 37.73 |
| small | 1 | 33.54 ± 43.82 | 22.82 ± 3.18 |
| small | 2 | 32.45 ± 39.19 | 25.24 ± 6.83 |
| medium | 0 | 104.75 ± 206.12 | 57.04 ± 48.73 |
| medium | 1 | 47.29 ± 9.32 | 47.87 ± 6.53 |
| medium | 2 | 32.89 ± 1.52 | 34.60 ± 1.13 |
| large | 0 | 113.16 ± 189.20 | 77.75 ± 48.34 |
| large | 1 | 52.67 ± 5.29 | 64.52 ± 3.04 |
| large | 2 | 53.73 ± 1.56 | 65.99 ± 0.82 |
| xl | 0 | 122.53 ± 182.62 | 128.62 ± 35.63 |
| xl | 1 | 63.73 ± 8.23 | 118.83 ± 0.23 |
| xl | 2 | 65.76 ± 4.05 | 121.82 ± 0.08 |
| 2.7b | 0 | 137.96 ± 181.38 | 179.41 ± 49.02 |
| 2.7b | 1 | 81.74 ± 0.73 | 165.86 ± 0.90 |
| 2.7b | 2 | 81.58 ± 0.09 | 165.02 ± 0.69 |

结论：不做 warm-up（warm-up=0）时，forward/backward 的平均耗时明显偏高，而且标准差显著变大，结果不稳定。主要原因是初始迭代会包含额外开销（如 CUDA 上下文初始化、kernel 选择与缓存、内存分配器预热等），导致首批 step 不能代表稳态性能。即使做 1-2 步 warm-up，结果仍可能不同，因为有些配置需要更多步才能完全进入稳态，且系统噪声也会带来残余波动。

### `nsys_profile`

#### (a) forward pass

| Model Size | Context | Forward (Python) | Forward (nsight) |
| ---------- | ------: | ---------------: | ---------------: |
| small      |     128 |     15.57 ± 0.41 |           18.382 |
| medium     |     128 |     29.02 ± 0.09 |           38.213 |
| large      |     128 |     43.23 ± 0.21 |           54.855 |
| xl         |     128 |     59.66 ± 0.03 |           74.066 |
| 2.7b       |     128 |     81.04 ± 0.07 |           53.136 |

还是比较不一样的，推测是因为 kernel 的问题。

#### (b) kernel

| Model Size | Context |                                                       Kernel |  Time | Invoked |
| ---------- | ------: | -----------------------------------------------------------: | ----: | ------- |
| small      |     128 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` | 55.0% | 60      |
| medium     |     128 | ` void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_tn_align1>(T1::Params)` | 45.2% | 120     |
| large      |     128 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)` | 52.6% | 107     |
| xl         |     128 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)` | 56.5% | 143     |
| 2.7b       |     128 | `void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)` | 90.2% | 148     |

确实还是同一个 kernel 在 backward 中耗最长时间，但是比例明显降低了。

#### (c) other kernel

比如 element-wise 乘法的 kernel。

#### (d) training

矩阵乘法的比例明显下降了，从之前的将近一半下降到只有 10-20%。处理 element-wise 的乘法的 kernel 比例上升。

#### (e) softmax vs. mm

可以发现 softmax 还是需要相当时间的，略低于计算 attention score 的时间，略低于 final matmul 的 二倍。

### `mixed_precision_accumulation`

运行结果如下：

```
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

可以发现，如果全程使用 `torch.float32`，则舍入误差最小；若全程使用 `torch.float16`，有效位数少导致误差累积增大；后面两种为混合精度，只要 accumulation 在 `float32` 中进行则误差仍然在可接受范围内。

### `benchmarking_mixed_precision`

#### (a) data type

- 模型参数仍是 FP32
- `ToyModel.fc1` 的输出：FP16
- `ToyModel.ln`  输出：FP32
- logits：FP32
- loss：FP32
- 梯度：FP32

#### (b) LayerNorm presicion analysis

在 FP16 mixed precision 下，LayerNorm 最敏感的是均值/方差的归约与归一化计算（减均值、平方、求和、rsqrt/除法），因为这些操作会累积舍入误差，且容易受 FP16 有效精度与动态范围限制影响。LayerNorm 的统计量若用低精度算，数值误差会被放大并影响训练稳定性，所以通常会在 FP32 中完成这些计算。换成 BF16 后由于其指数位与 FP32 相同，溢出/下溢问题明显缓解，但 BF16 尾数仍较短，精度仍不如 FP32，因此实践中通常仍让 LayerNorm 的统计与归约保持 FP32 更稳妥。

#### (c) mixed precision benchmarking

> 实验环境：`NVIDIA H800 PCIe`，`context_length=128`，`batch_size=4`，`warm-up=5`，`measure=10`
>
> 单位：ms（均值）；Speedup = FP32 / BF16

| Model Size | Fwd FP32 | Fwd BF16 | Fwd Speedup | Bwd FP32 | Bwd BF16 | Bwd Speedup |
| ---------- | -------: | -------: | ----------: | -------: | -------: | ----------: |
| small      |    21.57 |    23.54 |       0.92x |    23.02 |    24.97 |       0.92x |
| medium     |    45.03 |    46.80 |       0.96x |    46.51 |    49.65 |       0.94x |
| large      |    64.98 |    68.18 |       0.95x |    93.93 |    73.93 |       1.27x |
| xl         |    87.18 |    92.29 |       0.94x |   175.88 |    98.59 |       1.78x |
| 2.7b       |   119.78 |    62.11 |       1.93x |   264.10 |    86.81 |       3.04x |

在当前 GPU 上，BF16 对小模型（`small`/`medium`）的 forward 和 backward 都没有提速，甚至略慢；但随着模型增大，BF16 的收益迅速增强，尤其是 backward。以 `2.7b` 为例，forward 从 119.78ms 降到 62.11ms（1.93x），backward 从 264.10ms 降到 86.81ms（3.04x）。整体趋势是模型越大，mixed precision（BF16）越能发挥 Tensor Core 的优势，且在 backward 阶段更明显。

### `memory_profiling`

> 实验设置：`NVIDIA H800 PCIe`，`batch_size=4`，`warm-up=5`，`measure=10`，`context_length in {128, 256, 512}`
>
> 峰值内存来自 `outputs/memory_profiles/{fp32,bf16}` 的 memory snapshot（`torch.cuda.memory._dump_snapshot`），单位统一为 GiB（1 GiB = 1024^3 bytes）。

#### (a) active memory timeline analysis

![](https://yangty-pic.oss-cn-beijing.aliyuncs.com/cs336/lab2/memory-profiling-ctx512-forward.png)

![](https://yangty-pic.oss-cn-beijing.aliyuncs.com/cs336/lab2/memory-profiling-ctx512-train.png)

从时间线看，forward 图呈现明显“先上升后下降”的单峰形状，符合前向中激活逐层累积再释放的行为；train-step 图在更高内存区间内先升到峰值、再分阶段回落并最终稳定在较高平台，能和 forward/backward/optimizer 的阶段性行为对应起来。尤其是 train-step 中后段的锯齿与平台，反映了梯度与优化器相关张量的反复分配和保留。

#### (b) peak memory with FP32

FP32 下 2.7B 的峰值内存如下：

| Context Length | Forward Peak (GiB) | Train-step Peak (GiB) |
| --- | ---: | ---: |
| 128 | 18.08 | 51.44 |
| 256 | 24.31 | 51.44 |
| 512 | 39.77 | 65.52 |

结论：forward 峰值随 context 明显上升；train-step 在 128/256 时峰值接近，但到 512 时显著抬升。

#### (c) peak memory with BF16 mixed

BF16 mixed precision 下峰值（同一模型与 context）：

| Context Length | Forward Peak (GiB) | Train-step Peak (GiB) |
| --- | ---: | ---: |
| 128 | 22.45 | 51.44 |
| 256 | 26.50 | 52.09 |
| 512 | 36.86 | 62.69 |

与 FP32 对比：mixed precision 对峰值内存并非在所有配置都显著下降。小 context（128/256）下 forward 甚至更高或接近，train-step 也基本接近；到 context=512 时，BF16 才出现较明确收益（forward 39.77 -> 36.86 GiB，train-step 65.52 -> 62.69 GiB）。这说明峰值内存并不只由激活 dtype 决定，还受优化器状态、缓存与分配行为影响。

#### (d) activation tensor size

Transformer residual stream 的激活张量形状是 `(B, T, d_model)`，单精度字节数：

```
size_bytes = B * T * d_model * 4
```

在本作业参考配置中 `B=4, d_model=2560`，所以：

```
size_MB = 4 * T * 2560 * 4 / 1024^2 = 0.0390625 * T
```

因此 `T=128/256/512` 时分别为 `5/10/20 MB`（MiB）。

#### (e) largest allocation

在 forward 的 memory timeline（例如 `ctx512_forward.pickle`）把 Detail 降到 10% 后，最大的分配块大小是 **128 MiB**。从 stack trace 看，这些大块主要来自 self-attention 的 `scaled_dot_product_attention` 路径，对应的是注意力分数/概率这类 `B x H x T x T` 级别张量的分配。

## Optimizing Attention with FlashAttention-2

### `pytorch_attention`

> 实验环境：`NVIDIA H800 PCIe`

| d_model | seq_len | forward ms (mean±std) | backward ms (mean±std) | mem before bwd (MiB) | peak mem (MiB) |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.147 ± 0.003 | 0.417 ± 0.006 | 68.59 | 76.71 |
| 16 | 1024 | 0.334 ± 0.003 | 0.886 ± 0.009 | 131.09 | 259.59 |
| 16 | 4096 | 4.567 ± 0.010 | 14.395 ± 0.425 | 1112.38 | 3162.38 |
| 16 | 8192 | 17.767 ± 0.024 | 59.494 ± 1.732 | 4240.75 | 12436.75 |
| 16 | 16384 | 70.177 ± 0.062 | 233.282 ± 6.917 | 16737.50 | 49513.50 |
| 32 | 256 | 0.144 ± 0.004 | 0.545 ± 0.012 | 69.09 | 77.34 |
| 32 | 1024 | 0.346 ± 0.002 | 0.913 ± 0.024 | 133.09 | 262.09 |
| 32 | 4096 | 4.701 ± 0.013 | 14.639 ± 0.430 | 1120.38 | 3172.38 |
| 32 | 8192 | 18.430 ± 0.064 | 57.968 ± 1.806 | 4256.75 | 12456.75 |
| 32 | 16384 | 73.361 ± 0.041 | 230.936 ± 7.252 | 16769.50 | 49553.50 |
| 64 | 256 | 0.144 ± 0.004 | 0.551 ± 0.013 | 70.09 | 78.59 |
| 64 | 1024 | 0.379 ± 0.007 | 0.986 ± 0.010 | 137.09 | 267.09 |
| 64 | 4096 | 5.120 ± 0.017 | 15.918 ± 0.470 | 1136.38 | 3192.38 |
| 64 | 8192 | 19.990 ± 0.112 | 62.409 ± 1.945 | 4288.75 | 12496.75 |
| 64 | 16384 | 79.836 ± 0.119 | 250.914 ± 7.913 | 16833.50 | 49633.50 |
| 128 | 256 | 0.144 ± 0.003 | 0.420 ± 0.006 | 72.09 | 81.09 |
| 128 | 1024 | 0.469 ± 0.018 | 1.246 ± 0.019 | 145.09 | 277.09 |
| 128 | 4096 | 5.986 ± 0.013 | 18.652 ± 0.569 | 1168.38 | 3232.38 |
| 128 | 8192 | 24.207 ± 0.247 | 74.089 ± 2.344 | 4352.75 | 12576.75 |
| 128 | 16384 | 94.780 ± 0.472 | 291.347 ± 9.281 | 16961.50 | 49793.50 |

这个 benchmark 显示了朴素注意力的典型特征：时间与显存都被 $L\times L$ 的注意力矩阵主导。对所有 `d_model`，前向和反向耗时随 `seq_len` 增长非常快，而随 `d_model` 增长相对慢很多。例如在 `d_model=128` 下，前向耗时从 `L=4096` 的 `5.99 ms` 增至 `L=8192` 的 `24.21 ms`，再到 `L=16384` 的 `94.78 ms`；反向也表现出接近“ $L$ 翻倍、耗时约 4 倍”的趋势。

当前 GPU 下未出现 OOM。但观察显存消耗也可以发现其增长趋势也是 $O(L^2)$，要消除这部分内存成本，核心做法是避免显式物化并保存完整 $L\times L$ 注意力矩阵，例如采用 FlashAttention 风格的分块计算（或重计算/checkpoint 思路），以更多计算换更低激活显存。

### `torch_compile`

#### (a) attention

| d_model | seq_len | forward eager(ms) | forward compile(ms) | forward speedup | backward eager(ms) | backward compile(ms) | backward speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.147 | 0.127 | 1.154x | 0.417 | 0.351 | 1.188x |
| 16 | 1024 | 0.334 | 0.195 | 1.712x | 0.886 | 0.484 | 1.832x |
| 16 | 4096 | 4.567 | 2.240 | 2.038x | 14.395 | 6.302 | 2.284x |
| 16 | 8192 | 17.767 | 8.721 | 2.037x | 59.494 | 27.866 | 2.135x |
| 16 | 16384 | 70.177 | 25.981 | 2.701x | 233.282 | 98.637 | 2.365x |
| 32 | 256 | 0.144 | 0.280 | 0.516x | 0.545 | 0.735 | 0.741x |
| 32 | 1024 | 0.346 | 0.412 | 0.839x | 0.913 | 0.817 | 1.117x |
| 32 | 4096 | 4.701 | 1.961 | 2.398x | 14.639 | 5.887 | 2.487x |
| 32 | 8192 | 18.430 | 7.396 | 2.492x | 57.968 | 23.731 | 2.443x |
| 32 | 16384 | 73.361 | 29.497 | 2.487x | 230.936 | 97.754 | 2.362x |
| 64 | 256 | 0.144 | 0.283 | 0.510x | 0.551 | 0.733 | 0.752x |
| 64 | 1024 | 0.379 | 0.435 | 0.872x | 0.986 | 0.771 | 1.279x |
| 64 | 4096 | 5.120 | 2.430 | 2.107x | 15.918 | 7.224 | 2.204x |
| 64 | 8192 | 19.990 | 9.340 | 2.140x | 62.409 | 28.902 | 2.159x |
| 64 | 16384 | 79.836 | 36.924 | 2.162x | 250.914 | 119.268 | 2.104x |
| 128 | 256 | 0.144 | 0.138 | 1.046x | 0.420 | 0.336 | 1.250x |
| 128 | 1024 | 0.469 | 0.415 | 1.131x | 1.246 | 0.763 | 1.634x |
| 128 | 4096 | 5.986 | 3.365 | 1.779x | 18.652 | 10.599 | 1.760x |
| 128 | 8192 | 24.207 | 13.864 | 1.746x | 74.089 | 42.101 | 1.760x |
| 128 | 16384 | 94.780 | 53.214 | 1.781x | 291.347 | 163.079 | 1.787x |

Attention 子模块在长序列下加速明显，`seq_len>=4096` 时前后向普遍接近 `~1.7x-2.7x`。

#### (b) Transformer

| model size | forward eager (ms) | forward compile (ms) | forward speedup | backward eager (ms) | backward compile (ms) | backward speedup |
|:---|---:|---:|---:|---:|---:|---:|
| small | 27.239 | 19.828 | 1.374x | 54.233 | 40.509 | 1.339x |
| medium | 84.276 | 68.714 | 1.226x | 164.967 | 133.447 | 1.236x |
| large | 161.453 | 136.453 | 1.183x | 342.850 | 281.020 | 1.220x |
| xl | 326.482 | 279.752 | 1.167x | 665.574 | 566.427 | 1.175x |
| 2.7b | 490.072 | 448.186 | 1.093x | 1003.195 | 916.464 | 1.095x |

端到端整模型也有稳定收益，但随模型变大加速比有所下降（`small` 约 `1.3x`，`2.7b` 约 `1.09x`）。

### `flash_forward`

### `flash_benchmarking`

