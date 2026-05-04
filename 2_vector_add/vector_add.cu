/*
 * Exercise 2: Vector Addition
 *
 * Implement a CUDA kernel that adds two vectors element-wise and compare
 * its performance against a CPU implementation.
 *
 * You need to:
 *   1. Write the kernel body (compute global index, bounds check, add).
 *   2. Compute the correct grid size for the kernel launch.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace {

constexpr int kNumElements = 1 << 24;
constexpr int kThreadsPerBlock = 256;

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(1);
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int count) {
    // TODO: Have each thread identify which vector element it owns,
    //       guard against out-of-range threads, and then perform the
    //       elementwise addition for that position.

    // TODO: Your code here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

int main() {
    const size_t bytes = static_cast<size_t>(kNumElements) * sizeof(float);

    // Allocate host buffers for the two inputs and both CPU/GPU results.
    float* host_a = static_cast<float*>(std::malloc(bytes));
    float* host_b = static_cast<float*>(std::malloc(bytes));
    float* host_cpu = static_cast<float*>(std::malloc(bytes));
    float* host_gpu = static_cast<float*>(std::malloc(bytes));
    if (!host_a || !host_b || !host_cpu || !host_gpu) {
        std::fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < kNumElements; ++i) {
        host_a[i] = static_cast<float>(i % 1000) * 0.5f;
        host_b[i] = static_cast<float>((i * 7) % 1000) * 0.25f;
    }

    // Build a reference result on the CPU for later correctness checking.
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kNumElements; ++i) {
        host_cpu[i] = host_a[i] + host_b[i];
    }
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    // Allocate device buffers and timing events for the end-to-end GPU run.
    float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
    check_cuda(cudaMalloc(&dev_a, bytes), "cudaMalloc(dev_a)");
    check_cuda(cudaMalloc(&dev_b, bytes), "cudaMalloc(dev_b)");
    check_cuda(cudaMalloc(&dev_c, bytes), "cudaMalloc(dev_c)");

    cudaEvent_t total_start, kernel_start, kernel_stop, total_stop;
    check_cuda(cudaEventCreate(&total_start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&kernel_stop), "cudaEventCreate");
    check_cuda(cudaEventCreate(&total_stop), "cudaEventCreate");

    check_cuda(cudaEventRecord(total_start), "cudaEventRecord");
    check_cuda(cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D a");
    check_cuda(cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D b");

    // TODO: Choose a grid size that covers the full input vector when each
    //       block contributes kThreadsPerBlock threads. Make sure any final
    //       partial block is still included.

    // TODO: Your code here
    int blocks = (kNumElements + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Time just the kernel launch separately from the full GPU pipeline.
    check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord");
    vector_add_kernel<<<blocks, kThreadsPerBlock>>>(dev_a, dev_b, dev_c, kNumElements);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(kernel_stop), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(kernel_stop), "cudaEventSynchronize");

    check_cuda(cudaMemcpy(host_gpu, dev_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H c");
    check_cuda(cudaEventRecord(total_stop), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(total_stop), "cudaEventSynchronize");

    float gpu_kernel_ms = 0.0f, gpu_total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&gpu_kernel_ms, kernel_start, kernel_stop), "cudaEventElapsedTime");
    check_cuda(cudaEventElapsedTime(&gpu_total_ms, total_start, total_stop), "cudaEventElapsedTime");

    // Compare the GPU output with the CPU reference.
    int mismatches = 0;
    for (int i = 0; i < kNumElements; ++i) {
        if (std::fabs(host_cpu[i] - host_gpu[i]) > 1e-5f) {
            ++mismatches;
            if (mismatches < 5)
                std::fprintf(stderr, "Mismatch at %d: cpu=%f gpu=%f\n", i, host_cpu[i], host_gpu[i]);
        }
    }

    // Query the active GPU so the benchmark output is self-contained.
    int device = 0;
    cudaDeviceProp prop{};
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::printf("Vector Add Benchmark\n");
    std::printf("  Elements          : %d\n", kNumElements);
    std::printf("  GPU               : %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("  CPU time          : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time   : %.3f ms\n", gpu_kernel_ms);
    std::printf("  GPU total time    : %.3f ms (includes copies)\n", gpu_total_ms);
    if (gpu_kernel_ms > 0.0f)
        std::printf("  Speedup (kernel)  : %.2fx\n", cpu_ms / gpu_kernel_ms);
    if (gpu_total_ms > 0.0f)
        std::printf("  Speedup (total)   : %.2fx\n", cpu_ms / gpu_total_ms);
    std::printf("  Validation        : %s\n", mismatches == 0 ? "PASS" : "FAIL");

    // Clean up all host/device allocations and CUDA events.
    cudaEventDestroy(total_start);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(total_stop);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    std::free(host_a);
    std::free(host_b);
    std::free(host_cpu);
    std::free(host_gpu);

    return mismatches == 0 ? 0 : 1;
}
