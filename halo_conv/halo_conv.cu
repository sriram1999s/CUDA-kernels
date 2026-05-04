#include <stdio.h>

#define TILE_SIZE 16
#define RADIUS 1
#define MASK_WIDTH (2 * RADIUS + 1)
#define SHARED_SIZE (TILE_SIZE + 2 * RADIUS)

// TODO: Use __constant__ memory for the filter
// This is a special cache for data that is read-only and shared by all threads
__constant__ float c_mask[MASK_WIDTH * MASK_WIDTH];

__global__ void convolution_2d_kernel(const float* input, float* output, int width, int height) {
    // 1. Declare shared memory with padding for the Halo
    __shared__ float s_data[SHARED_SIZE][SHARED_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global output coordinates
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // 2. Collaborative Load with Halo
    // Each thread must load its primary pixel AND participate in loading the halo.
    // This is the "Step Up" - mapping the 16x16 threads to an 18x18 shared buffer.
    
    // TODO: Load center, edges, and corners into s_data
    // Hint: You need to offset the shared index by RADIUS
    s_data[ty + RADIUS][tx + RADIUS] = (row < height && col < width) ? input[row * width + col] : 0.0f;

    // Load Halo (Top/Bottom/Left/Right/Corners)
    // Your code here...

    if(tx < RADIUS) {
        int halo_col = col - RADIUS;
        s_data[ty + RADIUS][tx] = (row < height && halo_col >= 0) ? input[row * width + halo_col] : 0.0f;
    }
    if(tx >= TILE_SIZE - RADIUS) {
        int halo_col = col + RADIUS;
        s_data[ty + RADIUS][tx + 2 * RADIUS] = (row < height && halo_col < width) ? input[row * width + halo_col] : 0.0f;
    }
    if(ty < RADIUS) {
        int halo_row = row - RADIUS;
        s_data[ty][tx + RADIUS] = (halo_row >= 0 && col < width) ? input[halo_row * width + col] : 0.0f;
    }
    if(ty >= TILE_SIZE - RADIUS) {
        int halo_row = row + RADIUS;
        s_data[ty + 2 * RADIUS][tx + RADIUS] = (halo_row < height && col < width) ? input[halo_row * width + col] : 0.0f;
    }

    if(tx < RADIUS && ty < RADIUS) {
        int halo_row = row - RADIUS;
        int halo_col = col - RADIUS;
        s_data[tx][ty] = (halo_col >= 0 && halo_row >= 0) ? input[halo_row * width + halo_col] : 0.0f;
    }
    if (tx >= TILE_SIZE - RADIUS && ty < RADIUS) {
        int halo_row = row - RADIUS;
        int halo_col = col + RADIUS;
        s_data[ty][tx + 2 * RADIUS] = (halo_row >= 0 && halo_col < width) ? input[halo_row * width + halo_col] : 0.0f;
    }
    if (tx < RADIUS && ty >= TILE_SIZE - RADIUS) {
        int halo_row = row + RADIUS;
        int halo_col = col - RADIUS;
        s_data[ty + 2 * RADIUS][tx] = (halo_row < height && halo_col >= 0) ? input[halo_row * width + halo_col] : 0.0f;
    }
    if (tx >= TILE_SIZE - RADIUS && ty >= TILE_SIZE - RADIUS) {
        int halo_row = row + RADIUS;
        int halo_col = col + RADIUS;
        s_data[ty + 2 * RADIUS][tx + 2 * RADIUS] = (halo_row < height && halo_col < width) ? input[halo_row * width + halo_col] : 0.0f;
    }
    __syncthreads();

    // 3. Compute Convolution
    if (row < height && col < width) {
        float sum = 0.0f;
        // Loop over the mask
        for (int i = 0; i < MASK_WIDTH; i++) {
            for (int j = 0; j < MASK_WIDTH; j++) {
                // TODO: Pull from s_data and multiply by c_mask
                sum += s_data[ty + i][tx + j] * c_mask[i * MASK_WIDTH + j];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    const int width = 64;
    const int height = 64;
    const int size = width * height * sizeof(float);

    // Host memory allocation
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    float h_mask[MASK_WIDTH * MASK_WIDTH];

    // Initialize input data (simple gradient) and mask (3x3 box blur)
    for (int i = 0; i < width * height; i++) h_input[i] = (float)(i % 255);
    for (int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) h_mask[i] = 1.0f / 9.0f;

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    // Copy mask to constant memory
    cudaMemcpyToSymbol(c_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Define Grid and Block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    convolution_2d_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Simple verification (printing a small 4x4 section of the output)
    printf("Output (Top-Left 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", h_output[i * width + j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}