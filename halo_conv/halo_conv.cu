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

// CPU Reference Implementation for Correctness Checking
void cpu_convolution(const float* input, const float* mask, float* output, int width, int height) {
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float sum = 0.0f;
            for (int i = 0; i < MASK_WIDTH; i++) {
                for (int j = 0; j < MASK_WIDTH; j++) {
                    int cur_r = r + i - RADIUS;
                    int cur_c = c + j - RADIUS;
                    if (cur_r >= 0 && cur_r < height && cur_c >= 0 && cur_c < width) {
                        sum += input[cur_r * width + cur_c] * mask[i * MASK_WIDTH + j];
                    }
                }
            }
            output[r * width + c] = sum;
        }
    }
}

int main() {
    const int width = 128;   // Larger size for comprehensive check
    const int height = 128;
    const int size = width * height * sizeof(float);

    // Host memory allocation
    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float h_mask[MASK_WIDTH * MASK_WIDTH];

    // Initialize input data with random-ish values and mask with a sharpen filter
    for (int i = 0; i < width * height; i++) h_input[i] = (float)(rand() % 255);
    
    // Let's use a Sharpen mask for the check:
    // [ 0 -1  0 ]
    // [-1  5 -1 ]
    // [ 0 -1  0 ]
    h_mask[0] = 0.0f; h_mask[1] = -1.0f; h_mask[2] = 0.0f;
    h_mask[3] = -1.0f; h_mask[4] = 5.0f; h_mask[5] = -1.0f;
    h_mask[6] = 0.0f; h_mask[7] = -1.0f; h_mask[8] = 0.0f;

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Define Grid and Block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch GPU kernel
    printf("Launching GPU Kernel...\n");
    convolution_2d_kernel<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    // Run CPU reference
    printf("Running CPU Reference implementation...\n");
    cpu_convolution(h_input, h_mask, h_output_cpu, width, height);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // COMPREHENSIVE VERIFICATION
    printf("Verifying results...\n");
    double max_error = 0.0;
    int error_count = 0;
    const double epsilon = 1e-4;

    for (int i = 0; i < width * height; i++) {
        double diff = fabs((double)h_output_gpu[i] - (double)h_output_cpu[i]);
        if (diff > max_error) max_error = diff;
        if (diff > epsilon) {
            if (error_count < 5) {
                printf("Error at index %d: GPU=%f, CPU=%f (diff=%f)\n", i, h_output_gpu[i], h_output_cpu[i], diff);
            }
            error_count++;
        }
    }

    if (error_count == 0) {
        printf("PASS! Maximum error: %e\n", max_error);
    } else {
        printf("FAIL! Total errors: %d. Maximum error: %e\n", error_count, max_error);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    return 0;
}