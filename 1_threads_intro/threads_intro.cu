/*
 * Exercise 1: Thread and Block Indexing
 *
 * Learn how CUDA maps threads to a global index using blockIdx, blockDim,
 * and threadIdx.  Complete the kernel so that each thread prints its
 * block id, thread id, and computed global index.
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_threads() {
    // TODO: Determine each thread's global 1D index from its block and
    //       thread coordinates, then print the block id, thread id, and
    //       global index so you can see how CUDA maps threads onto work.
    //
    //       Expected output is one line per thread showing those three
    //       values. The exact ordering of the lines does not matter.

    // TODO: Your code here
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int globalId = blockId * blockDim.x + threadId;

    printf("Block ID: %d | Thread ID: %d | Global Index: %d\n", blockId, threadId, globalId);
}

int main() {
    const int blocks = 3;
    const int threads_per_block = 4;

    // Launch a small grid so the thread-to-index mapping is easy to inspect.
    printf("Launching %d blocks with %d threads each (%d total threads)\n",
           blocks, threads_per_block, blocks * threads_per_block);

    hello_threads<<<blocks, threads_per_block>>>();

    // Wait for the kernel so any device-side printf output is flushed.
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n",
                cudaGetErrorString(error));
        return 1;
    }
    return 0;
}
