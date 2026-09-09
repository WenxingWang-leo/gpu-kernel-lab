# CUDA 算子

每个算子一个目录，目录内约定：

```
<op_name>/
├── <op>_v1_naive.cu      # 按优化步骤分版本，v1/v2/...
├── <op>_torch_ext.cu     # （可选）PyTorch 扩展封装
├── test_<op>.py          # 数值校验 + benchmark 入口
└── NOTES.md              # 该算子的关键结论（链到 notes/experiments/ 详细笔记）
```

## 进度

| 算子 | 版本 | 状态 | 峰值性能 | 相对基线 |
|---|---|---|---|---|
| gemm | v1 naive | 🚧 W1 | | |

## 编译备忘

- `-arch=sm_XX` 按目标 GPU 调整（4090=sm_89, A100=sm_80, H100=sm_90）
- 调试：`-G -lineinfo`；发布：`-O3 --use_fast_math`（注意 fast_math 的数值影响）
- Nsight Compute：`ncu --set full ./bin`
