# CUDA-kernels
Just random cuda kernels I am building to learn CUDA

# High-Performance CUDA Kernels

![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)

A collection of optimized CUDA kernels implementing fundamental parallel algorithms with a focus on memory hierarchy management and hardware-level optimizations.

## Kernels & Optimization Techniques

| Kernel | Key Optimizations | Hardware Focus |
| :--- | :--- | :--- |
| **Vector/Matrix Addition** | Grid-stride loops, 2D Indexing | Coalesced Memory Access |
| **Parallel Reduction** | Shared Memory, Tree-based Summation | Warp Synchronization |
| **Tiled MatMul** | Shared Memory Tiling, `+1` Padding | Bank Conflict Avoidance |
| **Tiled Transpose** | SRAM Pivoting | Global Write Coalescing |

## Performance Highlights

* **Memory Coalescing:** All kernels are designed to ensure that threads within a warp access contiguous global memory addresses to maximize bus utilization.
* **Shared Memory Tiling:** High-latency DRAM traffic is minimized by loading data into on-chip SRAM, increasing the arithmetic intensity of the kernels.
* **Bank Conflict Mitigation:** Tiled kernels utilize address padding to ensure 32-way parallel access to shared memory banks.
* **Resource Occupancy:** Optimized block dimensions to maximize the number of active warps per Streaming Multiprocessor (SM).

## Building and Running

### Prerequisites
* NVIDIA GPU (Architecture: Turing `sm_75` or later)
* CUDA Toolkit 12.x+
* `nvcc` compiler
