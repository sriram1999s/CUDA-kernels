/*
 * Exercise 5: Shared-Memory Tiled Matrix Multiplication
 *
 * This kernel also computes:
 *
 *     C = A x B
 *
 * but it improves on the naive version by using tiling and shared memory.
 *
 * Why shared memory helps:
 *   In naive matmul, each thread reads an entire row/column pair directly
 *   from global memory, and nearby threads often fetch the same values again.
 *   In a tiled kernel, a block cooperatively loads a small tile of A and a
 *   small tile of B into shared memory once, then many threads reuse those
 *   values while computing different output elements.
 *
 * High-level algorithm:
 *   - Each block is responsible for one TILE_SIZE x TILE_SIZE tile of C
 *   - Each thread computes one output element inside that tile
 *   - The block walks across the K dimension tile by tile
 *   - On each iteration:
 *
 *       1. load one tile of A into shared memory
 *       2. load one tile of B into shared memory
 *       3. synchronize so the whole tile is available
 *       4. accumulate this tile's contribution to the output
 *       5. synchronize before reusing shared memory for the next tile
 *
 * After all K-tiles have been processed, each thread writes its final
 * accumulated value to C.
 *
 * Shared-memory layout:
 *   tileA stores a tile of A.
 *   tileB stores a tile of B and is padded by +1 column to reduce
 *   shared-memory bank conflicts during the inner multiply loop.
 *
 * The shared-memory arrays and loop structure are provided; fill in
 * the bodies marked with TODO.
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
#define TILE_SIZE 32

__global__ void matmul_shared(int M, int N, int K,
                              const float *A, const float *B, float *C) {
    // Shared memory tiles.  tileB is padded by 1 to avoid bank conflicts.
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    int globalRow = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalCol = blockIdx.x * TILE_SIZE + threadIdx.x;

    float partialSum = 0.0f;

    // Loop over tiles along the K dimension
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, TILE_SIZE); tileIdx++) {
        // TODO: Fill in the body of this tile loop.
        //
        // 1. Load this thread's contribution from the current A tile into
        // shared memory, writing zero when the access falls outside A.
        //
        // 2. Load this thread's contribution from the current B tile into
        // shared memory, again handling boundary tiles safely.
        //
        // 3. Synchronize so the full pair of shared-memory tiles is ready.
        //
        // 4. Accumulate this tile's contribution to the output element using
        // the cached A and B values, then synchronize again before advancing
        // to the next tile.

    // TODO: Your code here
        int aCol = tileIdx * TILE_SIZE + threadIdx.x;
        tileA[threadIdx.y][threadIdx.x] = 
            (globalRow < M && aCol < K) ? A[globalRow * K + aCol] : 0.0f;

        int bRow = tileIdx * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] = 
            (bRow < K && globalCol < N) ? B[bRow * N + globalCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            partialSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (globalRow < M && globalCol < N) {
        C[globalRow * N + globalCol] = partialSum;
    }
}

void cpu_matmul(int M, int N, int K, const float *A, const float *B, float *C) {
    // CPU reference used to verify the tiled GPU kernel.
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

    // Allocate inputs plus separate CPU/GPU outputs for correctness checks.
    std::vector<float> A(M * K), B(K * N), C_cpu(M * N, 0.0f), C_gpu(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(i % 13);
    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>((i % 7) - 3);

    // Run the simple CPU implementation as a reference answer.
    std::cout << "Running CPU validation..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matmul(M, N, K, A.data(), B.data(), C_cpu.data());
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

    // Copy the input matrices to the device and reserve space for C.
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, A.size() * sizeof(float)));
    CHECK(cudaMalloc(&dB, B.size() * sizeof(float)));
    CHECK(cudaMalloc(&dC, C_gpu.size() * sizeof(float)));

    CHECK(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC, 0, C_gpu.size() * sizeof(float)));

    // The tiled kernel uses a 2D launch where each block computes one C tile.
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(CEIL_DIV(N, TILE_SIZE), CEIL_DIV(M, TILE_SIZE));

    // Measure the shared-memory kernel execution time.
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matmul_shared<<<gridSize, blockSize>>>(M, N, K, dA, dB, dC);
    CHECK(cudaEventRecord(stop));

    CHECK(cudaGetLastError());
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    // Copy the GPU answer back and compare it with the CPU result.
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

    // Report both performance and correctness in one place.
    std::printf("Shared Memory Matrix Multiplication\n");
    std::printf("  Matrix size        : %d x %d x %d\n", M, N, K);
    std::printf("  CPU time           : %.3f ms\n", cpu_ms);
    std::printf("  GPU kernel time    : %.3f ms\n", ms);
    if (ms > 0.0f)
        std::printf("  Speedup (kernel)   : %.2fx\n", cpu_ms / ms);
    std::printf("  Validation         : %s\n", pass ? "PASS" : "FAIL");

    // Free device-side allocations before exiting.
    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    return pass ? 0 : 1;
}
