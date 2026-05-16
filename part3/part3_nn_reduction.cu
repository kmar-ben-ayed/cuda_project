// Steps:
//   CPU  element-wise multiply: xw[i][j] = x[j] * W0[i][j]
//   GPU  row-sum + sigmoid      (Part 3-1: naïve Part 3-2: optimised)
//   GPU final sum of h(1)      (Part 3-2 improvement 3)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Utility macros
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s  at %s:%d\n",                  \
                    cudaGetErrorString(_e), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

static __device__  inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

//  KERNELS 

// Naïve sum-reduction kernel (per row)
//   One block per row of xw.
//   Within the block, threads reduce in a strided tree using shared memory.  Only threads where tid%(2*stride)==0 are
__global__ void naiveRowSumSigmoid(const float* __restrict__ xw,
                                   float*       __restrict__ h,
                                   int rowLen)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int row = blockIdx.x;                       // one block per hidden neuron
    float acc = 0.0f;
    for (int i = tid; i < rowLen; i += blockDim.x)
        acc += xw[row * rowLen + i];
    sdata[tid] = acc;
    __syncthreads();

    // Naive tree
    // Naive tree
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tid % (2 * stride) == 0 && tid + stride < blockDim.x)
        if (tid % (2 * stride) == 0 && tid + stride < blockDim.x)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[row] = sigmoid(sdata[0]);
}

// [3-2 Improvement 1] Interleaved addressing (no divergence)
__global__ void improvedRowSumSigmoid_v1(const float* __restrict__ xw,
                                         float*       __restrict__ h,
                                         int rowLen)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int row = blockIdx.x;
    int base = row * rowLen;

    float acc = 0.0f;

    for (int i = tid; i < rowLen; i += blockDim.x)
        acc += xw[base + i];
    int base = row * rowLen;

    float acc = 0.0f;

    for (int i = tid; i < rowLen; i += blockDim.x)
        acc += xw[base + i];

    sdata[tid] = acc;
    sdata[tid] = acc;
    __syncthreads();

    // Interleaved: active threads are 0..blockDim.x/(2*stride)-1
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[row] = sigmoid(sdata[0]);
}

// [3-2 Improvement 2] Sequential load + interleaved reduction
//   Each thread loads multiple elements before reduction starts
//   ("serial reduction at load time"), increasing arithmetic
//   intensity and halving the number of blocks needed.
//   We use blockDim.x = 256 but process rowLen=8192 elements:
//   each thread loads rowLen/blockDim.x = 32 elements.
__global__ void improvedRowSumSigmoid_v2(const float* __restrict__ xw,
                                         float*       __restrict__ h,
                                         int rowLen)
{
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int row  = blockIdx.x;
    int base = row * rowLen;

    // Each thread accumulates its share of the row
    float acc = 0.0f;
    for (int i = tid; i < rowLen; i += blockDim.x)
        acc += xw[base + i];

    sdata[tid] = acc;
    __syncthreads();

    // Tree reduction on the partial sums in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        h[row] = sigmoid(sdata[0]);
}

// [3-2 Improvement 3] Warp shuffle for the final warp
//   Combines v2's sequential load with __shfl_down_sync for
//   the last 32 elements, avoiding shared memory in that step.
__global__ void improvedRowSumSigmoid_v3(const float* __restrict__ xw,
                                         float*       __restrict__ h,
                                         int rowLen)
{
    extern __shared__ float sdata[];

    int tid  = threadIdx.x;
    int row  = blockIdx.x;
    int base = row * rowLen;

    // Sequential load / partial sum
    float acc = 0.0f;
    for (int i = tid; i < rowLen; i += blockDim.x)
        acc += xw[base + i];

    sdata[tid] = acc;
    __syncthreads();

    // Shared-memory tree down to 32 elements
    for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Final warp: use shuffle to avoid shared-memory latency
    if (tid < 32) {
        float val = sdata[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val,  8);
        val += __shfl_down_sync(0xffffffff, val,  4);
        val += __shfl_down_sync(0xffffffff, val,  2);
        val += __shfl_down_sync(0xffffffff, val,  1);
        if (tid == 0)
            h[row] = sigmoid(val);
    }
}

// Final sum of h(1)  y  (simple reduction, single block)
__global__ void sumHidden(const float* __restrict__ h,
                          float*       __restrict__ y,
                          int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;

    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        acc += h[i];
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *y = sdata[0];
}

// Benchmark helper
struct KernelSpec {
    const char* name;
    void (*launch)(const float*, float*, int, int);
};

static void launchNaive(const float* xw, float* h, int hidden, int rowLen) {
    int bs = rowLen;   // block size = rowLen (8192) … one block per row
    //for the naïve kernel the block size must equal rowLen.
    // 8192 > 1024 (CUDA max), so we clamp and process with gridStride.
    // In practice the lab uses N=8192 and a 1024-thread block:
    bs = 1024;
    naiveRowSumSigmoid<<<hidden, bs, bs * sizeof(float)>>>(xw, h, rowLen);
}
static void launchV1(const float* xw, float* h, int hidden, int rowLen) {
    int bs = 1024;
    improvedRowSumSigmoid_v1<<<hidden, bs, bs * sizeof(float)>>>(xw, h, rowLen);
}
static void launchV2(const float* xw, float* h, int hidden, int rowLen) {
    int bs = 256;
    improvedRowSumSigmoid_v2<<<hidden, bs, bs * sizeof(float)>>>(xw, h, rowLen);
}
static void launchV3(const float* xw, float* h, int hidden, int rowLen) {
    int bs = 256;
    improvedRowSumSigmoid_v3<<<hidden, bs, bs * sizeof(float)>>>(xw, h, rowLen);
}

int main()
{
    const int INPUT   = 8192;
    const int HIDDEN  = 8192;

    printf("=== Part 3: Single-hidden-layer NN  ===\n");
    printf("Input size : %d\n", INPUT);
    printf("Hidden size: %d\n", HIDDEN);
    printf("All weights: 1.0  |  All inputs: 1.0\n\n");

    // Host: x and W0 (all ones) 
    std::vector<float> h_x(INPUT,         1.0f);
    std::vector<float> h_W0(HIDDEN*INPUT, 1.0f);

    // Step 1 (CPU): element-wise product: xw[i][j] = x[j]*W0[i][j]
    std::vector<float> h_xw(HIDDEN * INPUT);
    for (int i = 0; i < HIDDEN; ++i)
        for (int j = 0; j < INPUT; ++j)
            h_xw[i * INPUT + j] = h_x[j] * h_W0[i * INPUT + j];

    printf("Step 1 (CPU element-wise multiply) done.\n\n");

    float *d_xw, *d_h, *d_y;
    CUDA_CHECK(cudaMalloc(&d_xw, sizeof(float) * HIDDEN * INPUT));
    CUDA_CHECK(cudaMalloc(&d_h,  sizeof(float) * HIDDEN));
    CUDA_CHECK(cudaMalloc(&d_y,  sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_xw, h_xw.data(),
                          sizeof(float)*HIDDEN*INPUT, cudaMemcpyHostToDevice));

    // Benchmark kernels
    using LaunchFn = void (*)(const float*, float*, int, int);
    const char*  names[]   = { "Naive (stride/branch)", "V1 (interleaved)",
                                "V2 (seq load+reduce)", "V3 (seq+warp shfl)" };
    LaunchFn     launchers[] = { launchNaive, launchV1, launchV2, launchV3 };

    std::vector<float> h_hidden(HIDDEN);
    float h_y = 0.0f;

    for (int v = 0; v < 4; ++v) {
        launchers[v](d_xw, d_h, HIDDEN, INPUT);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));

        CUDA_CHECK(cudaEventRecord(t0));
        launchers[v](d_xw, d_h, HIDDEN, INPUT);
        // Final sum y = sum(h)
        sumHidden<<<1, 256, 256*sizeof(float)>>>(d_h, d_y, HIDDEN);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

        CUDA_CHECK(cudaMemcpy(h_hidden.data(), d_h, sizeof(float)*HIDDEN, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));

        printf("  %s \n", names[v]);
        printf("  Kernel time : %.4f ms\n", ms);
        printf("  h[0]        : %.6f   (expected: sigmoid(8192) ≈ 1.0)\n", h_hidden[0]);
        printf("  y           : %.2f   (expected: 8192*sigmoid(8192) ≈ 8192.0)\n\n", h_y);

        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    CUDA_CHECK(cudaFree(d_xw));
    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_y));

    printf("Done.\n");
    return 0;
}