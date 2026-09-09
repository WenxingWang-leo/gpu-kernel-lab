# GPU Kernel Lab

我的 AI Infra 算子深度修炼仓库 —— 从 naive CUDA kernel 到工业级算子调优的完整记录。

> 目标：给一个陌生 memory-bound / compute-bound 算子，能独立完成
> 「数学定义 → CUDA/Triton 实现 → 性能分析 → 逼近硬件极限」全流程。

## 仓库结构

```
├── operators/          # 算子实现
│   ├── cuda/           # 手写 CUDA kernel（含 PyTorch 扩展）
│   └── triton/         # Triton 实现
├── benchmarks/         # 统一 benchmark 工具与结果
├── notes/              # 学习笔记（L1 碎片层）
│   ├── papers/         # 论文精读
│   └── experiments/    # 实验笔记
├── weekly/             # 周报
├── blog/               # 月度整理（L2）与阶段长文（L3）
└── docs/               # 路线图与长期计划
```

## 第一阶段路线图（16 周 · 算子深度）

| 月份 | 主题 | 验收标准 |
|---|---|---|
| M1 (W1–4) | CUDA GEMM 优化之路 | SGEMM 达 cuBLAS 70%+，能用 roofline 解释每步收益 |
| M2 (W5–8) | Triton 双修 | 6 个算子完成 eager/Triton/CUDA 三档 benchmark |
| M3 (W9–12) | FlashAttention 攻坚 | FA2 三实现（SDPA/Triton/CUDA）对比 + GQA/sliding window 变体 |
| M4 (W13–16) | 反向传播与工程化 | backward kernel + custom op 注册 + CI + 阶段长文 |

详细周计划见 [docs/roadmap.md](docs/roadmap.md)。

## 硬件环境

<!-- 改成你自己的实际环境 -->
- GPU: （填写型号，如 RTX 4090）
- CUDA: （nvcc --version）
- PyTorch / Triton: （版本）

## Benchmark 约定

- 所有性能数据 warmup ≥ 10 次、测量 ≥ 50 次取中位数
- 对比基线：PyTorch eager / torch.compile / cuBLAS（或对应官方库）
- 记录绝对耗时 + 有效带宽（memory-bound）或 TFLOPS（compute-bound）
- 数据写入实验笔记，注明 GPU 型号与驱动版本

## 进度追踪

见 [weekly/](weekly/) 目录，每周日更新。
