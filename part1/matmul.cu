// Two kernels: matmulRowX (row-major X) and matmulRowY (row-major Y)
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using std::cout;
using std::vector;

#ifdef USE_INT
  using dtype = int;
  #define TYPE_NAME "INT"
#else
  using dtype = float;
  #define TYPE_NAME "FLOAT"
#endif

// Kernel A
__global__ void matmulRowX(const dtype* __restrict__ A,
                           const dtype* __restrict__ B,
                           dtype*       __restrict__ C,
                           int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    dtype acc = 0;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];

    C[row * N + col] = acc;
}

// Kernel B with coalesced access pattern for matrix B
__global__ void matmulRowY(const dtype* __restrict__ A,
                           const dtype* __restrict__ B,
                           dtype*       __restrict__ C,
                           int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    dtype acc = 0;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];

    C[row * N + col] = acc;
}

// CPU verification (small N only)
void verify(const vector<dtype>& A, const vector<dtype>& B,
            const vector<dtype>& C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            dtype ref = 0;
            for (int k = 0; k < N; ++k)
                ref += A[i * N + k] * B[k * N + j];
            assert(ref == C[i * N + j]);
        }
}

// run one kernel variant
void runKernel(bool useRowX,
               dtype* d_A, dtype* d_B, dtype* d_C,
               int N, size_t bytes,
               vector<dtype>& h_A, vector<dtype>& h_B, vector<dtype>& h_C)
{
    const char* kname = useRowX ? "matmulRowX" : "matmulRowY";

    cudaEvent_t ev0, ev1, ev2, ev3;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaEventCreate(&ev2); cudaEventCreate(&ev3);

    // H2D
    cudaEventRecord(ev0);
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(ev1);

    // Kernel
    const int THREADS = 32;
    dim3 block(THREADS, THREADS);
    dim3 grid(N / THREADS, N / THREADS);

    if (useRowX)
        matmulRowX<<<grid, block>>>(d_A, d_B, d_C, N);
    else
        matmulRowY<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(ev2);

    // D2H
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(ev3);
    cudaEventSynchronize(ev3);

    float t_h2d, t_ker, t_d2h, t_total;
    cudaEventElapsedTime(&t_h2d,   ev0, ev1);
    cudaEventElapsedTime(&t_ker,   ev1, ev2);
    cudaEventElapsedTime(&t_d2h,   ev2, ev3);
    cudaEventElapsedTime(&t_total, ev0, ev3);

    // GFLOPS = 2*N^3 / (kernel_time_in_s * 1e9)
    float gflops = (2.0f * N / 1e3f * N / 1e3f * N / 1e3f) * 1000.0f / t_ker;

    printf("\n--- Kernel: %s  |  Type: %s ---\n", kname, TYPE_NAME);
    printf("  H->D transfer  : %10.3f ms\n", t_h2d);
    printf("  Kernel exec    : %10.3f ms\n", t_ker);
    printf("  D->H transfer  : %10.3f ms\n", t_d2h);
    printf("  Total          : %10.3f ms\n", t_total);
    printf("  Performance    : %10.3f GFLOPS\n", gflops);

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaEventDestroy(ev2); cudaEventDestroy(ev3);
}

int main()
{
    const int N      = 8192;
    const size_t SZ  = (size_t)N * N * sizeof(dtype);

     cout << "Matrix size: " << N << "x" << N
         << "  Type: " << TYPE_NAME << "\n";
    vector<dtype> h_A(N * N), h_B(N * N), h_C(N * N);
    std::generate(h_A.begin(), h_A.end(), []{ return (dtype)(rand() % 100); });
    std::generate(h_B.begin(), h_B.end(), []{ return (dtype)(rand() % 100); });

    dtype *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, SZ);
    if (err != cudaSuccess) { printf("cudaMalloc A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_B, SZ);
    if (err != cudaSuccess) { printf("cudaMalloc B failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_C, SZ);
    if (err != cudaSuccess) { printf("cudaMalloc C failed: %s\n", cudaGetErrorString(err)); return 1; }


    runKernel(true,  d_A, d_B, d_C, N, SZ, h_A, h_B, h_C);   // matmulRowX
    runKernel(false, d_A, d_B, d_C, N, SZ, h_A, h_B, h_C);   // matmulRowY

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cout << "\nDone.\n";
    return 0;
}