// Matrix Multiplication using Tensor Cores (WMMA)
// Principle:
//   WMMA (Warp Matrix Multiply Accumulate) exposes Tensor Core
//   instructions through cooperative warp-level operations.
//   Each warp computes a 16×16×16 tile using half-precision
//   inputs and float accumulation in a single "step".

#include <cassert>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <mma.h>          // WMMA API

using namespace nvcuda;
using namespace nvcuda::wmma;

// Tile dimensions required by WMMA fragment sizes
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Each CTA computes a TILE_M × TILE_N output tile using several warps
#define TILE_M 64    // rows per block  (4 warps in M direction)
#define TILE_N 64    // cols per block  (4 warps in N direction)

//Error-checking macros 
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                    cudaGetErrorString(_e), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Tensor Core kernel
// A : M×K  (half, row-major)
// B : K×N  (half, row-major)
// C : M×N  (float, row-major)
__global__ void tcMatmul(const half*  __restrict__ A,
                         const half*  __restrict__ B,
                         float*       __restrict__ C,
                         int M, int N, int K)
{
    // blockDim.x = 128 (4 warps along X = N direction)
    // blockDim.y = 4   (4 warps along Y = M direction)
    //we use a flat 128-thread block with warpId decoded manually.
    int warpId  = (threadIdx.x + threadIdx.y * blockDim.x) / 32;
    int laneId  = threadIdx.x % 32;        

    // Warp tile origin (row / col in the output)
    int warpRow = (blockIdx.y * TILE_M) + (warpId / (TILE_N / WMMA_N)) * WMMA_M;
    int warpCol = (blockIdx.x * TILE_N) + (warpId % (TILE_N / WMMA_N)) * WMMA_N;

    if (warpRow >= M || warpCol >= N) return;

    // Declare WMMA fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragA;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> fragB;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>         fragC;

    fill_fragment(fragC, 0.0f);

    // Step over K in WMMA_K chunks
    for (int k = 0; k < K; k += WMMA_K) {
        load_matrix_sync(fragA, A + warpRow * K + k,       K);
        load_matrix_sync(fragB, B + k       * N + warpCol, N);
        mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store result
    store_matrix_sync(C + warpRow * N + warpCol, fragC, N, mem_row_major);
    (void)laneId;
}

// Convert float
__global__ void fp32_to_fp16(const float* __restrict__ in,
                              half*        __restrict__ out,
                              int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = __float2half(in[idx]);
}

int main()
{
    const int M = 8192, N = 8192, K = 8192;
    const double gflop = 2.0 * M * N * K / 1e9;

    printf("TensorCore WMMA  %dx%d x %dx%d\n", M, K, K, N);

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 1.0f);
    std::vector<float> h_C(M * N, 0.0f);

    float *d_Af, *d_Bf, *d_C;
    half  *d_Ah, *d_Bh;
    CUDA_CHECK(cudaMalloc(&d_Af, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&d_Bf, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&d_Ah, sizeof(half)  * M * K));
    CUDA_CHECK(cudaMalloc(&d_Bh, sizeof(half)  * K * N));
    CUDA_CHECK(cudaMalloc(&d_C,  sizeof(float) * M * N));

    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2)); CUDA_CHECK(cudaEventCreate(&ev3));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_Af, h_A.data(), sizeof(float)*M*K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Bf, h_B.data(), sizeof(float)*K*N, cudaMemcpyHostToDevice));

    int cvtBlocks = (M * K + 255) / 256;
    fp32_to_fp16<<<cvtBlocks, 256>>>(d_Af, d_Ah, M * K);
    cvtBlocks = (K * N + 255) / 256;
    fp32_to_fp16<<<cvtBlocks, 256>>>(d_Bf, d_Bh, K * N);
    CUDA_CHECK(cudaEventRecord(ev1));

    //  Kernel launch 
    // 128 threads per block, arranged as 4×32: 4 warps each owning one 16×16 tile
    // Each block covers TILE_M × TILE_N output
    dim3 block(128, 4);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    tcMatmul<<<grid, block>>>(d_Ah, d_Bh, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(ev2));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev3));
    CUDA_CHECK(cudaEventSynchronize(ev3));

    float t_h2d, t_ker, t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d,   ev0, ev1));
    CUDA_CHECK(cudaEventElapsedTime(&t_ker,   ev1, ev2));
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev0, ev3));

    double gflops = gflop / (t_ker / 1e3);

    printf("\nTensorCoreResults:\n");
    printf("HD + fp16 cvt: %10.3f ms\n", t_h2d);
    printf("Kernel execution: %10.3f ms\n",  t_ker);
    printf("Total: %10.3f ms\n",  t_total);
    printf("Performance: %10.3f GFLOPS\n", (float)gflops);
    printf("\nCorrectness check:\n");
    // With all-ones A and B, each C entry should be K
    printf("  Expected C[0]  = %g  (should be %d)\n", h_C[0], K);

    CUDA_CHECK(cudaFree(d_Af)); CUDA_CHECK(cudaFree(d_Bf));
    CUDA_CHECK(cudaFree(d_Ah)); CUDA_CHECK(cudaFree(d_Bh));
    CUDA_CHECK(cudaFree(d_C));
    return EXIT_SUCCESS;
}
