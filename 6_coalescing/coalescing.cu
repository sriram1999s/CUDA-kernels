/*
 * Exercise 6: Memory Coalescing
 *
 * This program applies the same operation in two different ways:
 *
 *     out[i][j] = alpha * in[i][j]
 *
 * The math is identical in both kernels. The only difference is how threads
 * are mapped to matrix coordinates, which changes the memory access pattern.
 *
 * Why coalescing matters:
 *   A warp is fastest when neighboring threads access neighboring addresses
 *   in global memory. That pattern is called coalesced access. If neighboring
 *   threads access addresses that are far apart, the hardware needs more
 *   memory transactions and effective bandwidth drops.
 *
 * This matrix is stored in row-major order:
 *
 *     element (row, col) lives at row * N + col
 *
 * So columns are contiguous in memory within a row, while moving from one
 * row to the next jumps by N elements.
 *
 * Part A — uncoalesced:
 *   threadIdx.x is mapped to rows.
 *   That means neighboring threads in a warp touch different rows at the
 *   same column, so their addresses are N floats apart.
 *
 * Part B — coalesced:
 *   threadIdx.x is mapped to columns.
 *   That means neighboring threads touch neighboring columns in the same row,
 *   so their addresses are consecutive in memory.
 *
 * High-level goal:
 *   Compare the runtime of the two kernels and observe that the faster one
 *   is not doing less arithmetic; it is just using a better memory access
 *   pattern.
 *
 * TODO markers:
 *   1. Implement scale_uncoalesced (Part A)
 *   2. Implement scale_coalesced   (Part B)
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(err));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

/* ---------- Part A: Uncoalesced access ----------
 *
 *   threadIdx.x  -> row   (i)
 *   threadIdx.y  -> column (j)
 *
 * Because the matrix is stored in row-major order, element (i, j) is at
 * address  in[i * N + j].  When threadIdx.x changes by 1 the address
 * jumps by N — neighboring threads touch addresses that are far apart.
 */
__global__ void scale_uncoalesced(const float* in, float* out,
                                  int M, int N, float alpha) {
    // TODO: Deliberately map the fastest-changing thread dimension to matrix
    // rows so that neighboring threads in a warp walk down the matrix rather
    // than across it. After computing the coordinates, apply the scaling only
    // for in-bounds elements.

    // TODO: Your code here
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        out[row * N + col] = alpha * in[row * N + col];
    }
}

/* ---------- Part B: Coalesced access ----------
 *
 *   threadIdx.x  -> column (j)
 *   threadIdx.y  -> row    (i)
 *
 * Now when threadIdx.x changes by 1 the address changes by 1 — perfect
 * coalescing.
 */
__global__ void scale_coalesced(const float* in, float* out,
                                int M, int N, float alpha) {
    // TODO: Map the fastest-changing thread dimension to matrix columns so
    // neighboring threads access neighboring memory locations. Once the row
    // and column are known, scale the matching element when the coordinates
    // are within bounds.

    // TODO: Your code here
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        out[row * N + col] = alpha * in[row * N + col];
    } 
}

/* ------------------------------------------------------------------ */

float run_uncoalesced_kernel(dim3 grid, dim3 block,
                             const float* d_in, float* d_out,
                             int M, int N, float alpha,
                             int warmup_iters, int timed_iters) {
    // Warm up the GPU so timing is less affected by one-time startup costs.
    for (int i = 0; i < warmup_iters; ++i) {
        scale_uncoalesced<<<grid, block>>>(d_in, d_out, M, N, alpha);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        scale_uncoalesced<<<grid, block>>>(d_in, d_out, M, N, alpha);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / timed_iters;
}

float run_coalesced_kernel(dim3 grid, dim3 block,
                           const float* d_in, float* d_out,
                           int M, int N, float alpha,
                           int warmup_iters, int timed_iters) {
    // Use the same warmup/timing structure so Part A and Part B are comparable.
    for (int i = 0; i < warmup_iters; ++i) {
        scale_coalesced<<<grid, block>>>(d_in, d_out, M, N, alpha);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        scale_coalesced<<<grid, block>>>(d_in, d_out, M, N, alpha);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / timed_iters;
}

int main() {
    const int M = 4096, N = 4096;
    const float alpha = 2.0f;
    const size_t bytes = static_cast<size_t>(M) * N * sizeof(float);
    const int BLOCK = 32;
    const int warmup = 5, iters = 20;

    // Host data includes the input matrix, a CPU-computed reference, and a
    // reusable buffer for whichever kernel result we copy back.
    float* h_in  = static_cast<float*>(std::malloc(bytes));
    float* h_ref = static_cast<float*>(std::malloc(bytes));
    float* h_out = static_cast<float*>(std::malloc(bytes));
    for (int i = 0; i < M * N; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f;
    for (int i = 0; i < M * N; ++i) h_ref[i] = alpha * h_in[i];

    // Only one device output buffer is needed because the kernels run one at a time.
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- Part A: Uncoalesced ---
    //   gridDim.x covers rows, gridDim.y covers columns
    dim3 block_uncoal(BLOCK, BLOCK);
    dim3 grid_uncoal(CEIL_DIV(M, BLOCK), CEIL_DIV(N, BLOCK));

    float ms_uncoal = run_uncoalesced_kernel(grid_uncoal, block_uncoal,
                                             d_in, d_out, M, N, alpha,
                                             warmup, iters);

    // Validate the slow mapping before moving on to the coalesced version.
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool pass_uncoal = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) { pass_uncoal = false; break; }
    }

    // --- Part B: Coalesced ---
    //   gridDim.x covers columns, gridDim.y covers rows
    dim3 block_coal(BLOCK, BLOCK);
    dim3 grid_coal(CEIL_DIV(N, BLOCK), CEIL_DIV(M, BLOCK));

    float ms_coal = run_coalesced_kernel(grid_coal, block_coal,
                                         d_in, d_out, M, N, alpha,
                                         warmup, iters);

    // Validate the fast mapping with the same reference matrix.
    CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    bool pass_coal = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) { pass_coal = false; break; }
    }

    // Summarize the measured effect of changing only the access pattern.
    int device = 0;
    cudaDeviceProp prop{};
    CHECK(cudaGetDevice(&device));
    CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("Memory Coalescing Benchmark  (%d x %d matrix, %d iterations)\n", M, N, iters);
    std::printf("  GPU                : %s\n", prop.name);
    std::printf("  Uncoalesced (A)    : %.3f ms   [%s]\n", ms_uncoal, pass_uncoal ? "PASS" : "FAIL");
    std::printf("  Coalesced   (B)    : %.3f ms   [%s]\n", ms_coal,   pass_coal   ? "PASS" : "FAIL");
    if (ms_coal > 0.0f)
        std::printf("  Speedup (B/A)      : %.2fx\n", ms_uncoal / ms_coal);

    // Free device and host buffers.
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    std::free(h_in);
    std::free(h_ref);
    std::free(h_out);
    return (pass_uncoal && pass_coal) ? 0 : 1;
}
