# 第一阶段路线图：算子深度（W1–W16）

每周节奏：周六上午读资料（3h）→ 周六下午写代码（3h）→ 周日实验 + 周报（3h）

---

## 月份 1：CUDA GEMM 优化之路

**目标**：从 naive GEMM 写到 Tensor Core 版本，建立性能分析肌肉记忆。

### W1：基线与分析工具
- [ ] 读：Simon Boehm《How to Optimize a CUDA Matmul Kernel》前言 + Kernel 1–2
- [ ] 读：CUDA Programming Guide 内存层级章节
- [ ] 代码：naive GEMM + global memory coalescing 版（起点代码见 `operators/cuda/gemm/`）
- [ ] 工具：跑通 Nsight Compute，看 SM throughput / memory throughput / occupancy
- [ ] 周报：记录 naive 版本的 bottleneck 判断（为什么访存 bound）

### W2：Shared Memory Tiling
- [ ] 读：Simon Boehm Kernel 3–4；理解 bank conflict
- [ ] 代码：shared memory tiling 版本 + bank conflict 前后对比
- [ ] 实验：扫 tile size（32/64/128），画性能曲线
- [ ] 预期：达 cuBLAS 15–25%

### W3：Register Blocking 与向量化
- [ ] 读：Simon Boehm Kernel 5–8
- [ ] 代码：2D register blocking + float4 向量化访存
- [ ] 实验：Nsight Compute 看 long scoreboard stall 下降
- [ ] 预期：达 cuBLAS 40–60%

### W4：Tensor Core 入门
- [ ] 读：wmma API 文档 + mma PTX 简介；CUTLASS GEMM 概念图（只读概念）
- [ ] 代码：wmma Tensor Core GEMM（FP16 输入 / FP32 累加）
- [ ] 实验：FP32 SIMT vs Tensor Core 的 TFLOPS 对比
- [ ] **月末验收**：SGEMM 达 cuBLAS 70%+ 或 HGEMM 达 50%+；写《我的 GEMM 优化笔记》

---

## 月份 2：Triton 双修

**目标**：同批算子 CUDA/Triton 各写一遍，理解 Triton 编译器边界。

### W5：Triton 基础
- [ ] 跑通官方 tutorial 01–03（vector add / fused softmax / matmul）
- [ ] 玩 matmul tutorial 的 autotune 配置空间
- [ ] 周报：Triton 帮你做了哪些 CUDA 里手动做的事？

### W6：融合算子三件套
- [ ] Triton 实现 RMSNorm、RoPE、SwiGLU
- [ ] 对比 PyTorch eager 和 torch.compile
- [ ] 用 profiler 验证融合后 HBM 访问次数减少
- [ ] 周报：用 roofline 算这些算子的理论带宽上限

### W7：CUDA 对照组
- [ ] CUDA 手写 RMSNorm（前向+反向），warp shuffle 归约
- [ ] Triton vs 手写 CUDA vs apex 三方对比
- [ ] 周报：Triton 在什么情况下追不上手写 CUDA？

### W8：进阶
- [ ] persistent matmul（TMA 相关可选）
- [ ] FP8 GEMM（硬件不支持则改做 persistent + split-K）
- [ ] **月末验收**：6 个算子三档 benchmark 全部入库

---

## 月份 3：FlashAttention 攻坚

**目标**：吃透 FA 系列，具备开发 Attention 变体的能力。

### W9：理论与数学
- [ ] 精读 FA1 论文；读 Milakov & Gimelshein online softmax
- [ ] 手推 online softmax merge 公式；推导 IO 复杂度 O(N²) vs O(N²/M)
- [ ] 代码：PyTorch 伪代码版 online softmax attention（慢但对）

### W10：Triton 版 FA2
- [ ] 读 FA2 论文 + Triton fused attention tutorial
- [ ] 代码：基于 tutorial 改出自己的 Triton FA2 前向，逐模块注释
- [ ] 实验：扫 seq_len（512→8K）对比 PyTorch SDPA

### W11：CUDA 版简化 FA2
- [ ] 读 Tri Dao FA2 CUDA 源码结构（组织方式，不逐行）
- [ ] 代码：CUDA 简化版（固定 head_dim=64，causal 可选）
- [ ] 目标：达 Triton 版 60–80% 性能即可

### W12：变体开发
- [ ] 给 FA2 加 GQA 支持 + sliding window mask（Triton 版）
- [ ] **月末验收**：长文《FlashAttention 从论文到代码》+ 三实现 benchmark 表

---

## 月份 4：反向传播与工程化

### W13：归约类算子 backward
- [ ] 推导 RMSNorm backward 数学（dx/dw/db）
- [ ] CUDA 实现 + torch gradient check 数值验证

### W14：Attention backward
- [ ] 读 FA2 backward 部分（为什么重算不存中间结果）
- [ ] 跑通 Triton tutorial attention backward 并写注释

### W15：工程化
- [ ] torch.utils.cpp_extension 注册 custom op + autograd Function
- [ ] 仓库补 CI（pytest + benchmark 回归）、README、性能文档

### W16：阶段校准 🎯
- [ ] 不写新代码，全面复盘
- [ ] 能力自测：任挑一个 memory-bound 算子，2h 内从数学定义到可用 Triton 实现？
- [ ] 产出：阶段总结博客
- [ ] 决定第二阶段侧重（不达标则顺延补 M1/M3）

---

## 参考资源

- Simon Boehm: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance
- CUDA C++ Programming Guide（案头查阅）
- PMPP 第 4 版（只查不读）
- FlashAttention v1/v2/v3 论文 + Tri Dao 源码
- OpenAI Triton 官方 tutorials
- GPU MODE Discord（常驻：#matmul #flash-attention 频道）
