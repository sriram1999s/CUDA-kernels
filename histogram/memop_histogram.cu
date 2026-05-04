#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

/**
 * YOUR TASK: Optimized Shared Memory Histogram
 * * 1. Declare a __shared__ array 'temp' of size NUM_BINS.
 * 2. INITIALIZATION: Every thread in the block must help set the 'temp' 
 * bins to 0. (Since shared memory is uninitialized).
 * 3. SYNC: Wait for initialization to finish.
 * 4. LOCAL COUNT: Process global data and use atomicAdd on the SHARED 'temp' bins.
 * 5. SYNC: Wait for all threads in the block to finish their local counts.
 * 6. AGGREGATION: Have each thread take its finished shared bin value and 
 * use atomicAdd to add it to the GLOBAL 'bins' array.
 */
__global__ void histogram_shared_kernel(const unsigned char* data, int* bins, int n) {
    // TODO: 1. Declare shared memory
    __shared__ int temp[NUM_BINS];
    int idx = threadIdx.x;

    // TODO: 2. Initialize shared memory to zero
    temp[idx] = 0;

    // TODO: 3. Synchronize
    __syncthreads();
    
    // TODO: 4. Perform local histogram in shared memory
    if (idx < n) {
        atomicAdd(&temp[data[idx]], 1);
    }

    // TODO: 5. Synchronize
    __syncthreads();
    
    // TODO: 6. Flush shared memory results to global memory
    atomicAdd(&bins[idx], temp[idx]);
}

int main() {
    const int N = 1024 * 1024; // 1MB
    unsigned char *h_data = (unsigned char*)malloc(N);
    int h_bins[NUM_BINS] = {0};

    // Initialize with random data
    for (int i = 0; i < N; i++) h_data[i] = (unsigned char)(rand() % NUM_BINS);

    unsigned char *d_data;
    int *d_bins;
    cudaMalloc(&d_data, N);
    cudaMalloc(&d_bins, NUM_BINS * sizeof(int));

    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
    cudaMemset(d_bins, 0, NUM_BINS * sizeof(int));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Launching Shared Memory Optimized Kernel...\n");
    histogram_shared_kernel<<<gridSize, blockSize>>>(d_data, d_bins, N);

    cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify
    int total_counts = 0;
    for (int i = 0; i < NUM_BINS; i++) total_counts += h_bins[i];

    if (total_counts == N) {
        printf("PASS: Optimized counts match (%d)\n", total_counts);
    } else {
        printf("FAIL: Total counts (%d) do not match N (%d)\n", total_counts, N);
    }

    cudaFree(d_data);
    cudaFree(d_bins);
    free(h_data);
    return 0;
}