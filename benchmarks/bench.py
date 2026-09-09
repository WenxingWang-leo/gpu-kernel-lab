"""统一 benchmark 工具。

约定：
- warmup >= 10 次，测量 >= 50 次，取中位数
- 同时返回毫秒数与（可选的）有效带宽 / TFLOPS

用法示例：
    from benchmarks.bench import bench, bench_op
    ms = bench(lambda: my_kernel(x))
    row = bench_op("rmsnorm_triton", fn, bytes_moved=..., flops=...)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Optional

import torch


def bench(fn: Callable, warmup: int = 10, iters: int = 50) -> float:
    """返回中位耗时（毫秒）。基于 CUDA Event，适用于 GPU kernel。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


@dataclass
class BenchResult:
    name: str
    ms: float
    bandwidth_gbps: Optional[float] = None   # memory-bound 算子填
    tflops: Optional[float] = None           # compute-bound 算子填

    def row(self) -> dict:
        return asdict(self)


def bench_op(
    name: str,
    fn: Callable,
    bytes_moved: Optional[int] = None,
    flops: Optional[int] = None,
    warmup: int = 10,
    iters: int = 50,
) -> BenchResult:
    """benchmark 一个算子并按需换算带宽 / 算力。

    bytes_moved: 一次调用实际读写的显存总字节数（理论最小值）
    flops:       一次调用的浮点运算次数
    """
    ms = bench(fn, warmup, iters)
    bw = bytes_moved / (ms * 1e-3) / 1e9 if bytes_moved else None
    tf = flops / (ms * 1e-3) / 1e12 if flops else None
    return BenchResult(name=name, ms=ms, bandwidth_gbps=bw, tflops=tf)


def print_table(results: list[BenchResult]) -> None:
    """简单表格输出，直接贴进实验笔记。"""
    header = f"{'name':<28} {'ms':>10} {'GB/s':>10} {'TFLOPS':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        bw = f"{r.bandwidth_gbps:.1f}" if r.bandwidth_gbps else "-"
        tf = f"{r.tflops:.2f}" if r.tflops else "-"
        print(f"{r.name:<28} {r.ms:>10.3f} {bw:>10} {tf:>10}")


def check_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-4, atol=1e-4) -> bool:
    """数值正确性校验，不通过时打印最大误差并返回 False。"""
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    if not ok:
        diff = (a - b).abs()
        print(f"[FAIL] max abs diff = {diff.max().item():.3e}, "
              f"mean = {diff.mean().item():.3e}")
    return ok
