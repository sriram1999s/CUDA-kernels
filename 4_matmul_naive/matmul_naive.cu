/*
 * Exercise 4: Naive Matrix Multiplication
 *
 * This kernel computes matrix multiplication:
 *
 *     C = A x B
 *
 * where:
 *   - A has shape M x K
 *   - B has shape K x N
 *   - C has shape M x N
 *
 * High-level idea:
 *   Each thread is responsible for exactly one output element C[row][col].
 *   To compute that element, the thread takes:
 *
 *     - row "row" from A
 *     - column "col" from B
 *
 *   and computes their dot product:
 *
 *     C[row][col] = sum over k of A[row][k] * B[k][col]
 *
 * Why this is called "naive":
 *   Every thread reads all K values it needs directly from global memory.
 *   Neighboring threads often reload the same data, so this version is simple
 *   but not very efficient.
 *
 * Mapping:
 *   - threadIdx.y / blockIdx.y choose the output row
 *   - threadIdx.x / blockIdx.x choose the output column
 *   - one 2D grid covers the whole output matrix
 *
 * Matrices are stored in row-major order:
 *   A[i][k] = A[i * K + k]
 *   B[k][j] = B[k * N + j]
 *   C[i][j] = C[i * N + j]
 */
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__ void matmul_naive(int M, int N, int K,
                             const float *A, const float *B, float *C) {
    // TODO: Assign each thread to one output element in C.
    //
    // 1. Convert the 2D block/thread coordinates into the output row and
    // column handled by this thread.
    //
    // 2. Ignore threads that land outside the matrix bounds.
    //
    // 3. Accumulate the dot product between the corresponding row of A and
    // column of B across the shared K dimension.
    //
    // 4. Store the finished value into the correct location in C.

    // TODO: Your code here
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) 
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++)
    {
        sum += A[row * K + k] * B[k * N + col];
    }
    //c[row][col]
    C[row * N + col] = sum;
}

void cpu_matmul(int M, int N, int K, const float *A, const float *B, float *C) {
    // Straightforward triple-loop reference used only for validation/timing.
    for (int x = 0; x < M; ++x)
        for (int y = 0; y < N; ++y) {
            float tmp = 0.0f;
            for (int i = 0; i < K; ++i)
                tmp += A[x * K + i] * B[i * N + y];
            C[x * N + y] = tmp;
        }
}

bool nearly_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int BLOCK_SIZE = 32;

    // Allocate matrices and separate storage for CPU and GPU outputs.
    std::vector<float> A(M * K), B(K * N), C_cpu(M * N, 0.0f), C_gpu(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(i % 13);
    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>((i % 7) - 3);

    // Run the CPU implementation first so we have a correctness baseline.
    std::cout << "Running CPU validation..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(M, N, K, A.data(), B.data(), C_cpu.data());
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    // Allocate device memory and copy the input matrices to the GPU.
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

    CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, C_gpu.size() * sizeof(float)));

    // A 2D grid covers the output matrix tile by tile.
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));

    // Time only the GPU kernel execution.
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matmul_naive<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC);
    CHECK(cudaEventRecord(stop));

    CHECK(cudaGetLastError());
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // Bring the result back and compare against the CPU reference.
    CHECK(cudaMemcpy(C_gpu.data(), dC, C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    for (int i = 0; i < M * N; ++i) {
        if (!nearly_equal(C_cpu[i], C_gpu[i])) {
            std::cerr << "Mismatch at " << i << ": CPU=" << C_cpu[i]
                      << ", GPU=" << C_gpu[i] << std::endl;
            pass = false;
            break;
        }
    }

    // Print timing plus a simple pass/fail correctness summary.
    std::printf("Naive Matrix Multiplication\n");
    std::printf("  Matrix size        : %d x %d x %d\n", M, N, K);
    std::printf("  CPU time           : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time    : %.3f ms\n", ms);
    if (ms > 0.0f)
        std::printf("  Speedup (kernel)   : %.2fx\n", cpu_ms / ms);
    std::printf("  Validation         : %s\n", pass ? "PASS" : "FAIL");

    // Free GPU allocations before returning.
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    return pass ? 0 : 1;
}
