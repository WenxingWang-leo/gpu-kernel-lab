# 【实验】
日期：YYYY-MM-DD ｜ 环境：GPU 型号, CUDA x.x, PyTorch/Triton 版本

## 假设
（动手之前写：我认为瓶颈是 X，改了 Y 应该提升 Z。没有假设的实验是在碰运气）

## 基线数据
| 版本 | 输入规模 | 耗时 | 带宽/TFLOPS | 相对基线 |
|---|---|---|---|---|
| PyTorch eager | | | | 1.0x |

## 实验记录
### 尝试 1：（改了什么）
- 结果：
- 分析：（Nsight Compute 关键指标，符合还是违背假设？）
### 尝试 2：
- 结果：
- 分析：

## 结论
- 最终相对 eager 提速 Nx，达到理论上限的 X%
- 剩余 gap 在哪，是否值得继续追

## 坑
（驱动、数值误差、未 warmup 假数据……）

## 代码位置
repo: operators/xxx @ commit <hash>
