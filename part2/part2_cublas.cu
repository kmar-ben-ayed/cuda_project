#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

//Error checking macros 
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t _s = (call);                                        \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                 \
                    (int)_s, __FILE__, __LINE__);                          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

using real = double;  // change to float for Sgemm

int main()
{
    const int M = 8192, N = 8192, K = 8192;
    const real alpha = 1.0, beta = 0.0;
    const double gflop = 2.0 * M * N * K / 1e9;

    printf("cuBLAS Dgemm  %dx%d x %dx%d\n", M, K, K, N);

    std::vector<real> h_A(M * K, 1.0);   // all-ones for easy verification
    std::vector<real> h_B(K * N, 1.0);
    std::vector<real> h_C(M * N, 0.0);

    // Create cuBLAS handle & stream 
    cublasHandle_t handle;
    cudaStream_t   stream;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    real *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(real) * M * K));
    CUDA_CHECK(cudaMalloc(&d_B, sizeof(real) * K * N));
    CUDA_CHECK(cudaMalloc(&d_C, sizeof(real) * M * N));

    cudaEvent_t ev_start, ev_h2d, ev_ker, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_ker));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), sizeof(real)*M*K, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), sizeof(real)*K*N, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev_h2d, stream));

    // Kernel: C^T = B^T * A^T  (row-major trick) 
    // cuBLAS is column-major; we compute C = A*B in row-major by swapping
    CUDA_CHECK(cudaEventRecord(ev_ker, stream));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,d_B, N,d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));

    CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, sizeof(real)*M*N,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    //  Timing 
    float t_h2d, t_ker, t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d,   ev_start, ev_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&t_ker,   ev_h2d,   ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev_start, ev_stop));

    double gflops = gflop / (t_ker / 1e3);

    printf("cuBLAS Dgemm Results:\n");
    printf("HD transfer: %10.3f ms\n", t_h2d);
    printf("Kernel execution: %10.3f ms\n", t_ker);
    printf("Total: %10.3f ms\n", t_total);
    printf("Performance: %10.3f GFLOPS\n", (float)gflops);
    printf("Correctness check:\n");
    printf("Expected C[0]= %g  (should be %d)\n", h_C[0], K);

    // Cleanup
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
