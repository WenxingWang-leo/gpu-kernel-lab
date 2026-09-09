# Triton 算子

每个算子一个文件 + 对应测试，M2 开始填充。

## 计划清单

- [ ] W5: tutorial 01–03 跑通与注释版
- [ ] W6: rmsnorm.py / rope.py / swiglu.py
- [ ] W7: 与 CUDA 手写版的三方对比
- [ ] W8: matmul_persistent.py
- [ ] W10–12: flash_attention.py（含 GQA / sliding window 变体）
- [ ] W14: attention backward

## 通用约定

- 每个 kernel 带 autotune 时，记录最终选中的 config 到实验笔记
- 数值校验统一用 `benchmarks/bench.py` 的 `check_close`
