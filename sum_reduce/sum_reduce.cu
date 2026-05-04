#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_kernel(int* g_idata, int* g_odata, unsigned int n) {
    // 1. Allocate shared memory for the block
    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Load input from global memory to shared memory
    // Handle bounds check
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // 3. Do reduction in shared memory
    // TODO: Implement the tree-based reduction loop
    // Hint: for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // YOUR CODE HERE: sdata[tid] += ...
            sdata[tid] += sdata[tid + s];
        }
        // YOUR CODE HERE: Wait for all threads to finish this level
        __syncthreads();
    }

    // 4. Write the result for this block back to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 1024; // Keep it simple for now
    size_t size = N * sizeof(int);

    int h_in[N], h_out[N/BLOCK_SIZE];
    for (int i = 0; i < N; i++) h_in[i] = 1; // Sum should be 1024

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, (N / BLOCK_SIZE) * sizeof(int));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch one block to keep this example simple
    reduce_kernel<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);

    cudaMemcpy(h_out, d_out, (N / BLOCK_SIZE) * sizeof(int), cudaMemcpyDeviceToHost);

    // Final sum on CPU (reducing the results from each block)
    int gpu_result = 0;
    for(int i = 0; i < N/BLOCK_SIZE; i++) gpu_result += h_out[i];

    if (gpu_result == N) printf("SUCCESS: Sum is %d\n", gpu_result);
    else printf("FAIL: Expected %d, got %d\n", N, gpu_result);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
