#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

/**
 * YOUR TASK:
 * 1. Calculate the global thread index.
 * 2. Check if the index is within bounds (n).
 * 3. Read the value from the 'data' array at that index.
 * 4. Increment the corresponding bin in the 'bins' array.
 * * NOTE: Multiple threads will likely see the same value at the same time.
 * Use the appropriate CUDA intrinsic to handle the race condition.
 */
__global__ void histogram_kernel(const unsigned char* data, int* bins, int n) {
    // TODO: Implement the histogram logic here

    int idx = threadIdx.x;
    if (idx < n) {
        atomicAdd(&bins[data[idx]], 1);
    }
}

__global__ void histogram_kernel_race(const unsigned char* data, int* bins, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        bins[data[idx]] += 1;
    }
}

int main() {
    // 1. Setup Data
    const int N = 1024 * 1024; // 1MB of data
    unsigned char *h_data = (unsigned char*)malloc(N);
    int h_bins[NUM_BINS] = {0};

    for (int i = 0; i < N; i++) h_data[i] = (unsigned char)(rand() % NUM_BINS);

    // 2. Allocate Device Memory
    unsigned char *d_data;
    int *d_bins;
    cudaMalloc(&d_data, N);
    cudaMalloc(&d_bins, NUM_BINS * sizeof(int));

    // 3. Initialize and Copy
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, NUM_BINS * sizeof(int));

    // 4. Launch Kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Launching kernel with %d blocks...\n", gridSize);
    histogram_kernel<<<gridSize, blockSize>>>(d_data, d_bins, N);

    // 5. Cleanup and Verify
    cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    int total_counts = 0;
    for (int i = 0; i < NUM_BINS; i++) total_counts += h_bins[i];

    if (total_counts == N) {
        printf("PASS: Total counts match (%d)\n", total_counts);
    } else {
        printf("FAIL: Total counts (%d) do not match N (%d)\n", total_counts, N);
    }

    cudaFree(d_data);
    cudaFree(d_bins);
    free(h_data);
    return 0;
}