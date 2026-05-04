/*
 * Exercise 7: Shared Memory Bank Conflicts
 *
 * This program transposes a matrix:
 *
 *     output[col][row] = input[row][col]
 *
 * Here:
 *   - input is the original M x N matrix in global memory
 *   - output is the transposed N x M matrix in global memory
 *
 * Why use shared memory if each element is used only once?
 *   The goal is not reuse; the goal is reordering. A direct transpose in
 *   global memory usually makes either the reads or the writes strided.
 *   Shared memory gives each block a fast scratchpad so it can:
 *
 *     1. read a tile from global memory in a coalesced pattern
 *     2. transpose that tile inside shared memory by swapping indices
 *     3. write the tile back to global memory in a coalesced pattern
 *
 * High-level algorithm:
 *   - The grid covers the whole matrix tile by tile.
 *   - Each block handles one TILE x TILE region.
 *   - Each thread loads one element into shared memory:
 *
 *         tile[threadIdx.y][threadIdx.x] = input[in_row][in_col]
 *
 *   - After __syncthreads(), the block reads the same tile with swapped
 *     indices:
 *
 *         tile[threadIdx.x][threadIdx.y]
 *
 *     That swapped read is what performs the transpose.
 *   - Each thread then writes one transposed element into output.
 *
 * Bank conflicts:
 *   Shared memory is split into 32 banks. If threads in a warp access
 *   different addresses that map to the same bank, those accesses are
 *   serialized; this is a bank conflict.
 *
 * Part A — transpose_bank_conflicts:
 *   Uses:
 *
 *       __shared__ float tile[TILE][TILE];
 *
 *   For TILE = 32, the row stride is 32 floats. During the transpose step,
 *   a warp reads down a column of shared memory via tile[threadIdx.x][...].
 *   Those addresses are 32 floats apart, so they map to the same bank and
 *   cause a 32-way bank conflict.
 *
 * Part B — transpose_no_conflicts:
 *   Uses:
 *
 *       __shared__ float tile[TILE][TILE + 1];
 *
 *   The algorithm is identical, but the extra column changes the row stride
 *   from 32 to 33 floats. Now a column read from shared memory lands on
 *   different banks instead of the same one, removing the conflict.
 *
 * Both kernels produce the same transposed matrix, but Part B should
 * be measurably faster. The program times each kernel with CUDA events.
 *
 * TODO markers:
 *   1. Implement transpose_bank_conflicts  (Part A)
 *   2. Implement transpose_no_conflicts    (Part B)
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
#define TILE 32

/* ---------- Part A: With bank conflicts ----------
 *
 *  Shared memory tile is  TILE x TILE  (stride = 32).
 *  When threads write/read a column, addresses 0, 32, 64, …
 *  all fall into bank 0  →  32-way bank conflict.
 *
 *  Steps:
 *    1. Compute (row, col) in the INPUT matrix for this thread.
 *    2. Load  input[row][col]  into  tile[threadIdx.y][threadIdx.x].
 *    3. __syncthreads()
 *    4. Compute (row, col) in the OUTPUT matrix.
 *       The output tile is at the *transposed* block position:
 *       output block row = blockIdx.x, output block col = blockIdx.y.
 *    5. Write  tile[threadIdx.x][threadIdx.y]  to  output[row][col].
 *       (Note the swapped indices — this is the column read that causes
 *        bank conflicts when the tile is not padded.)
 */
__global__ void transpose_bank_conflicts(const float* input, float* output,
                                         int M, int N) {
    __shared__ float tile[TILE][TILE];

    // TODO: Implement the transpose through shared memory for the unpadded
    // tile.
    //
    // 1. Compute this thread's input coordinates and stage that element in
    // shared memory, handling boundary tiles safely.
    //
    // 2. Synchronize so every element of the tile has been loaded.
    //
    // 3. Compute the corresponding output coordinates for the transposed
    // tile location.
    //
    // 4. Write the transposed value by reading the shared tile with swapped
    // indices. In this version, that shared-memory access pattern should
    // still exhibit bank conflicts.

    // TODO: Your code here

    int in_row = blockIdx.y * TILE + threadIdx.y;
    int in_col = blockIdx.x * TILE + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = (in_row < M && in_col < N) ? input[in_row * N + in_col] : 0.0f;

    __syncthreads();

    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    if (out_row < M && out_col < N)
        output[out_row * N + out_col] = tile[threadIdx.x][threadIdx.y];
}

/* ---------- Part B: Without bank conflicts ----------
 *
 *  Same logic as Part A, but the tile is  TILE x (TILE + 1).
 *  The extra column shifts each row's starting bank, so the column
 *  access  tile[threadIdx.x][threadIdx.y]  hits 32 *different* banks.
 */
__global__ void transpose_no_conflicts(const float* input, float* output,
                                       int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // <-- +1 padding

    // TODO: Reuse the same transpose logic as Part A, but apply it to the
    // padded shared-memory tile declared above. The extra column should leave
    // the output unchanged while removing the shared-memory bank conflicts
    // seen in the unpadded version.

    // TODO: Your code here
    int in_row = blockIdx.y * TILE + threadIdx.y;
    int in_col = blockIdx.x * TILE + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = (in_row < M && in_col < N) ? input[in_row * N + in_col] : 0.0f;

    __syncthreads();

    int out_row = blockIdx.x * TILE + threadIdx.y;
    int out_col = blockIdx.y * TILE + threadIdx.x;

    if (out_row < M && out_col < N)
        output[out_row * N + out_col] = tile[threadIdx.x][threadIdx.y];
}

/* ------------------------------------------------------------------ */

void cpu_transpose(const float* in, float* out, int M, int N) {
    // CPU reference transpose used to verify both shared-memory kernels.
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            out[j * M + i] = in[i * N + j];
}

float run_bank_conflicts_kernel(dim3 grid, dim3 block,
                                const float* d_in, float* d_out, int M, int N,
                                int warmup_iters, int timed_iters) {
    // Warm up before timing so the measurement better reflects steady-state behavior.
    for (int i = 0; i < warmup_iters; ++i) {
        transpose_bank_conflicts<<<grid, block>>>(d_in, d_out, M, N);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        transpose_bank_conflicts<<<grid, block>>>(d_in, d_out, M, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / timed_iters;
}

float run_no_conflicts_kernel(dim3 grid, dim3 block,
                              const float* d_in, float* d_out, int M, int N,
                              int warmup_iters, int timed_iters) {
    // Use the same benchmarking pattern as Part A for a fair comparison.
    for (int i = 0; i < warmup_iters; ++i) {
        transpose_no_conflicts<<<grid, block>>>(d_in, d_out, M, N);
    }
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < timed_iters; ++i) {
        transpose_no_conflicts<<<grid, block>>>(d_in, d_out, M, N);
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
    const size_t in_bytes  = static_cast<size_t>(M) * N * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(N) * M * sizeof(float);
    const int warmup = 5, iters = 20;

    // Host buffers hold the input, the CPU reference transpose, and the most
    // recent GPU result copied back for validation.
    float* h_in  = static_cast<float*>(std::malloc(in_bytes));
    float* h_ref = static_cast<float*>(std::malloc(out_bytes));
    float* h_out = static_cast<float*>(std::malloc(out_bytes));
    for (int i = 0; i < M * N; ++i) h_in[i] = static_cast<float>(i % 1000) * 0.001f;
    cpu_transpose(h_in, h_ref, M, N);

    // One input buffer and one reusable output buffer are enough for both kernels.
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  in_bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));
    CHECK(cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice));

    // The grid covers the matrix in 32x32 tiles.
    dim3 block(TILE, TILE);
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));

    // Run and validate the version that still suffers from shared-memory conflicts.
    // --- Part A: With bank conflicts ---
    CHECK(cudaMemset(d_out, 0, out_bytes));
    float ms_conflict = run_bank_conflicts_kernel(grid, block, d_in, d_out,
                                                  M, N, warmup, iters);
    CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    bool pass_a = true;
    for (int i = 0; i < N * M; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) { pass_a = false; break; }
    }

    // Run and validate the padded version that avoids those conflicts.
    // --- Part B: Without bank conflicts ---
    CHECK(cudaMemset(d_out, 0, out_bytes));
    float ms_no_conflict = run_no_conflicts_kernel(grid, block, d_in, d_out,
                                                   M, N, warmup, iters);
    CHECK(cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost));
    bool pass_b = true;
    for (int i = 0; i < N * M; ++i) {
        if (std::fabs(h_out[i] - h_ref[i]) > 1e-4f) { pass_b = false; break; }
    }

    // Report correctness and the performance gap between the two layouts.
    int device = 0;
    cudaDeviceProp prop{};
    CHECK(cudaGetDevice(&device));
    CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("Bank Conflict Benchmark  (%d x %d transpose, %d iterations)\n", M, N, iters);
    std::printf("  GPU                      : %s\n", prop.name);
    std::printf("  With bank conflicts (A)  : %.3f ms   [%s]\n", ms_conflict,    pass_a ? "PASS" : "FAIL");
    std::printf("  No bank conflicts   (B)  : %.3f ms   [%s]\n", ms_no_conflict, pass_b ? "PASS" : "FAIL");
    if (ms_no_conflict > 0.0f)
        std::printf("  Speedup (A/B)            : %.2fx\n", ms_conflict / ms_no_conflict);

    // Release all temporary allocations.
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    std::free(h_in);
    std::free(h_ref);
    std::free(h_out);
    return (pass_a && pass_b) ? 0 : 1;
}
