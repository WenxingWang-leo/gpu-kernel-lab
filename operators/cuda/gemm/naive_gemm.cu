// W1 起点代码：naive GEMM（C = A @ B，行主序，FP32）
// 每个线程计算 C 的一个元素。
//
// 编译（独立可执行）:
//   nvcc -O2 -arch=sm_89 naive_gemm.cu -o naive_gemm   # sm_89 = RTX 4090，按你的卡改
// 分析:
//   ncu --set full ./naive_gemm
//
// 思考题（写进实验笔记）:
//   1. 这个 kernel 里每个线程对 A、B 的 global memory 访问模式分别是什么？
//   2. 为什么 B 的访问是 coalesced 的而整体性能仍然很差？
//   3. 理论带宽上限下，这个 kernel 的 FLOPs/byte 比值意味着什么？

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                    __LINE__, cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C 的列

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

static void fill_random(float* p, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) p[i] = (float)rand() / RAND_MAX - 0.5f;
}

int main() {
    const int M = 4096, N = 4096, K = 4096;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    float *hA = (float*)malloc(bytesA), *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);
    fill_random(hA, M * K, 42);
    fill_random(hB, K * N, 43);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // warmup
    for (int i = 0; i < 3; ++i)
        gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    const int iters = 10;
    for (int i = 0; i < iters; ++i)
        gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("naive GEMM  M=N=K=%d  %.3f ms  %.2f TFLOPS\n", M, ms, tflops);

    // TODO(W1 后半周): 在这里加 coalesced 版本，并写与 cuBLAS 的数值校验
    // 提示：cublasSgemm 是列主序，传参时注意

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
