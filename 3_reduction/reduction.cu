/*
 * Exercise 3: Parallel Reduction (Sum)
 *
 * This program sums a 1D array on the GPU.
 *
 * High-level idea:
 *   A full sum is too large for one thread block, so the reduction happens
 *   in two stages:
 *
 *     1. Each block reduces its own chunk of the input into one partial sum
 *     2. The host adds those block-level partial sums to get the final result
 *
 * Why shared memory?
 *   Threads in a block need to repeatedly combine values with nearby threads.
 *   Shared memory provides a fast block-local scratchpad for that process.
 *   Each thread first loads one value into shared memory, then the block
 *   repeatedly combines pairs of values until only one sum remains.
 *
 * Tree-reduction pattern:
 *   - Start with one value per thread in shared memory
 *   - On each iteration, only the lower half of the threads stay active
 *   - Thread tid adds the value from tid + stride into its own slot
 *   - The stride halves each round: blockDim/2, blockDim/4, ..., 1
 *   - After the final round, shared[0] holds the sum for that block
 *
 * Output:
 *   partial_sums[blockIdx.x] stores the sum produced by one block.
 *   The CPU then adds all entries of partial_sums to form the final answer.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int kNumElements = 1 << 20;
constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void reduce_sum(const float* input, float* partial_sums, int count) {
    __shared__ float shared[kThreadsPerBlock];

    unsigned int tid = threadIdx.x;
    unsigned int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Three steps:
    //
    // Step 1 — Stage one input value per thread in shared memory, using a
    // zero contribution for threads whose global index is outside the array.
    // Synchronize after the load.
    //
    // Step 2 — Perform an in-place tree reduction in shared memory. On each
    // round, only a shrinking subset of threads should combine values, with
    // a synchronization point between rounds.
    //
    // Step 3 — Have one thread write the block's final sum to the partial
    // output array.

    // TODO: Your code here

    shared[tid] = (global_index < count) ? input[global_index] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) 
    {
        partial_sums[blockIdx.x] = shared[0];
    }
}

}  // namespace

int main() {
    const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);
    const int blocks = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    const size_t partial_bytes = static_cast<size_t>(blocks) * sizeof(float);

    // Host storage for the full input and one partial sum per block.
    float* host_input = static_cast<float*>(std::malloc(bytes));
    float* host_partials = static_cast<float*>(std::malloc(partial_bytes));
    if (!host_input || !host_partials) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < kNumElements; ++i) {
        host_input[i] = 1.0f + static_cast<float>(i % 5) * 0.25f;
    }

    // Compute the full sum on the CPU so the GPU result can be checked.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_sum = 0.0;
    for (int i = 0; i < kNumElements; ++i) {
        cpu_sum += host_input[i];
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    // Copy the input to the GPU and allocate space for block-level partials.
    float* device_input = nullptr;
    float* device_partials = nullptr;
    check_cuda(cudaMalloc(&device_input, bytes), "cudaMalloc(device_input)");
    check_cuda(cudaMalloc(&device_partials, partial_bytes), "cudaMalloc(device_partials)");
    check_cuda(cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Time the reduction kernel itself.
    cudaEvent_t start_event, stop_event;
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate");
    check_cuda(cudaEventRecord(start_event), "cudaEventRecord");

    reduce_sum<<<blocks, kThreadsPerBlock>>>(device_input, device_partials, kNumElements);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(stop_event), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize");

    float gpu_kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    check_cuda(cudaMemcpy(host_partials, device_partials, partial_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Finish the two-stage reduction by summing the block partials on the host.
    double gpu_sum = 0.0;
    for (int i = 0; i < blocks; ++i) {
        gpu_sum += host_partials[i];
    }

    std::printf("Reduction Example\n");
    std::printf("  Elements           : %d\n", kNumElements);
    std::printf("  Blocks             : %d\n", blocks);
    std::printf("  Threads per block  : %d\n", kThreadsPerBlock);
    std::printf("  CPU sum            : %.3f\n", cpu_sum);
    std::printf("  GPU sum            : %.3f\n", gpu_sum);
    std::printf("  CPU time           : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time    : %.3f ms\n", gpu_kernel_ms);
    if (gpu_kernel_ms > 0.0f)
        std::printf("  Speedup (kernel)   : %.2fx\n", cpu_ms / gpu_kernel_ms);
    std::printf("  Validation         : %s\n",
                std::fabs(cpu_sum - gpu_sum) < 1e-2 ? "PASS" : "FAIL");

    // Release all temporary resources before exiting.
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(device_input);
    cudaFree(device_partials);
    std::free(host_input);
    std::free(host_partials);
    return std::fabs(cpu_sum - gpu_sum) < 1e-2 ? 0 : 1;
}
