#include "tournament_tree.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <climits>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            return false; \
        } \
    } while(0)

// Device kernel implementations
__device__ bool computeSetIntersection(Matrix* m1, int row1, Matrix* m2, int row2) {
    // Shared memory for vector data
    __shared__ int8_t vec1[MAX_VECTOR_SIZE];
    __shared__ int8_t vec2[MAX_VECTOR_SIZE];
    
    // Load vectors into shared memory
    int tid = threadIdx.x;
    if (tid < m1->cols) {
        vec1[tid] = m1->data[m1->row_offsets[row1] + tid];
    }
    if (tid < m2->cols) {
        vec2[tid] = m2->data[m2->row_offsets[row2] + tid];
    }
    __syncthreads();
    
    // Check for intersection (each thread checks subset)
    bool localIntersection = false;
    for (int i = tid; i < m1->cols; i += blockDim.x) {
        for (int j = 0; j < m2->cols; j++) {
            if (vec1[i] == vec2[j]) {
                localIntersection = true;
                break;
            }
        }
        if (localIntersection) break;
    }
    
    // Reduce across threads in block
    __shared__ bool hasMatch[MAX_THREADS_PER_BLOCK];
    hasMatch[tid] = localIntersection;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            hasMatch[tid] = hasMatch[tid] || hasMatch[tid + stride];
        }
        __syncthreads();
    }
    
    return hasMatch[0];
}

__device__ void storeGlobalResult(int8_t* unionSet, int unionSize, int battle_idx, TournamentLevel* next_level) {
    // Store result in next level's matrix
    if (next_level && battle_idx < next_level->num_matrices) {
        Matrix* result_matrix = &next_level->matrices[battle_idx];
        
        // Store union size and data
        if (threadIdx.x == 0) {
            result_matrix->rows = 1;
            result_matrix->cols = unionSize;
            result_matrix->row_offsets[0] = 0;
            
            // Copy union data
            for (int i = 0; i < unionSize; i++) {
                result_matrix->data[i] = unionSet[i];
            }
        }
    }
}

__device__ void storeUnionResult(Matrix* m1, int row1, Matrix* m2, int row2, int battle_idx, TournamentLevel* next_level) {
    __shared__ int8_t unionSet[MAX_UNION_SIZE];
    __shared__ int unionSize;
    
    if (threadIdx.x == 0) unionSize = 0;
    __syncthreads();
    
    // Add elements from first vector
    for (int i = threadIdx.x; i < m1->cols; i += blockDim.x) {
        int8_t val = m1->data[m1->row_offsets[row1] + i];
        int pos = atomicAdd(&unionSize, 1);
        if (pos < MAX_UNION_SIZE) {
            unionSet[pos] = val;
        }
    }
    
    // Add elements from second vector (avoiding duplicates)
    for (int i = threadIdx.x; i < m2->cols; i += blockDim.x) {
        int8_t val = m2->data[m2->row_offsets[row2] + i];
        bool isDuplicate = false;
        
        for (int j = 0; j < unionSize && j < MAX_UNION_SIZE; j++) {
            if (unionSet[j] == val) {
                isDuplicate = true;
                break;
            }
        }
        
        if (!isDuplicate) {
            int pos = atomicAdd(&unionSize, 1);
            if (pos < MAX_UNION_SIZE) {
                unionSet[pos] = val;
            }
        }
    }
    
    __syncthreads();
    
    // Store result in global memory for next round
    storeGlobalResult(unionSet, unionSize, battle_idx, next_level);
}

__device__ void processBattle(TournamentLevel level, int battle_idx, int combination_idx, TournamentLevel* next_level) {
    // Get the two matrices for this battle
    if (battle_idx * 2 + 1 >= level.num_matrices) return;
    
    Matrix* m1 = &level.matrices[battle_idx * 2];
    Matrix* m2 = &level.matrices[battle_idx * 2 + 1];
    
    // Calculate which row combination this thread handles
    int total_combinations = m1->rows * m2->rows;
    if (combination_idx >= total_combinations) return;
    
    int row1 = combination_idx / m2->rows;
    int row2 = combination_idx % m2->rows;
    
    // Apply set intersection algorithm
    bool hasIntersection = computeSetIntersection(m1, row1, m2, row2);
    
    if (hasIntersection) {
        // Store union result for next round
        storeUnionResult(m1, row1, m2, row2, battle_idx, next_level);
    }
}

__global__ void tournamentKernel(TournamentLevel* levels, int level, int max_combinations) {
    int battle = blockIdx.y;
    int combination = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (level >= 0 && battle < levels[level].num_matrices / 2) {
        TournamentLevel* next_level = (level > 0) ? &levels[level - 1] : nullptr;
        processBattle(levels[level], battle, combination, next_level);
    }
}

// Host class implementation
TournamentTree::TournamentTree() : d_tournament_levels(nullptr), num_levels(0), num_matrices(0) {
    // Initialize memory pools
    d_memory.matrix_pool = nullptr;
    d_memory.result_pool = nullptr;
    d_memory.metadata_pool = nullptr;
    d_memory.matrix_pool_size = 0;
    d_memory.result_pool_size = 0;
    d_memory.metadata_pool_size = 0;
}

TournamentTree::~TournamentTree() {
    cleanup();
}

bool TournamentTree::initialize(const std::vector<std::vector<int8_t>>& input_matrices) {
    num_matrices = input_matrices.size();
    if (num_matrices == 0) return false;
    
    // Calculate number of tournament levels
    num_levels = static_cast<int>(ceil(log2(num_matrices)));
    
    // Allocate memory
    if (!allocateMemory()) return false;
    
    // Initialize first level with input matrices
    TournamentLevel* h_levels = new TournamentLevel[num_levels];
    
    // Set up level 0 (input level)
    h_levels[num_levels - 1].num_matrices = num_matrices;
    h_levels[num_levels - 1].matrices = new Matrix[num_matrices];
    h_levels[num_levels - 1].result_counts = new int[num_matrices];
    
    // Copy input matrices to first level
    for (int i = 0; i < num_matrices; i++) {
        Matrix& mat = h_levels[num_levels - 1].matrices[i];
        mat.rows = 1;
        mat.cols = input_matrices[i].size();
        
        // Allocate and copy data
        CUDA_CHECK(cudaMalloc(&mat.data, mat.cols * sizeof(int8_t)));
        CUDA_CHECK(cudaMemcpy(mat.data, input_matrices[i].data(), mat.cols * sizeof(int8_t), cudaMemcpyHostToDevice));
        
        // Set up row offsets
        CUDA_CHECK(cudaMalloc(&mat.row_offsets, sizeof(int)));
        int offset = 0;
        CUDA_CHECK(cudaMemcpy(mat.row_offsets, &offset, sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Initialize other levels
    for (int level = num_levels - 2; level >= 0; level--) {
        int matrices_this_level = h_levels[level + 1].num_matrices / 2;
        h_levels[level].num_matrices = matrices_this_level;
        h_levels[level].matrices = new Matrix[matrices_this_level];
        h_levels[level].result_counts = new int[matrices_this_level];
        
        // Allocate matrices for this level
        for (int i = 0; i < matrices_this_level; i++) {
            Matrix& mat = h_levels[level].matrices[i];
            mat.rows = 1;
            mat.cols = MAX_UNION_SIZE;
            
            CUDA_CHECK(cudaMalloc(&mat.data, mat.cols * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&mat.row_offsets, sizeof(int)));
            
            int offset = 0;
            CUDA_CHECK(cudaMemcpy(mat.row_offsets, &offset, sizeof(int), cudaMemcpyHostToDevice));
        }
    }
    
    // Copy levels to device
    CUDA_CHECK(cudaMemcpy(d_tournament_levels, h_levels, num_levels * sizeof(TournamentLevel), cudaMemcpyHostToDevice));
    
    delete[] h_levels;
    return true;
}

bool TournamentTree::runTournament() {
    for (int level = num_levels - 1; level > 0; level--) {
        int battles_this_level = num_matrices / (1 << (num_levels - level));
        
        // Grid configuration
        dim3 grid(
            (MAX_COMBINATIONS + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK,
            battles_this_level,
            1
        );
        dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);
        
        // Launch kernel for this tournament level
        tournamentKernel<<<grid, block>>>(d_tournament_levels, level, MAX_COMBINATIONS);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for kernel errors
        CUDA_CHECK(cudaGetLastError());
    }
    
    return true;
}

std::vector<int8_t> TournamentTree::getFinalResult() {
    std::vector<int8_t> result;
    
    if (num_levels == 0 || !d_tournament_levels) return result;
    
    // Get the final result from level 0
    TournamentLevel final_level;
    CUDA_CHECK(cudaMemcpy(&final_level, d_tournament_levels, sizeof(TournamentLevel), cudaMemcpyDeviceToHost));
    
    if (final_level.num_matrices > 0) {
        Matrix final_matrix;
        CUDA_CHECK(cudaMemcpy(&final_matrix, final_level.matrices, sizeof(Matrix), cudaMemcpyDeviceToHost));
        
        result.resize(final_matrix.cols);
        CUDA_CHECK(cudaMemcpy(result.data(), final_matrix.data, final_matrix.cols * sizeof(int8_t), cudaMemcpyDeviceToHost));
    }
    
    return result;
}

bool TournamentTree::allocateMemory() {
    // Calculate memory requirements
    size_t total_matrices = 0;
    for (int level = 0; level < num_levels; level++) {
        total_matrices += num_matrices / (1 << level);
    }
    
    d_memory.matrix_pool_size = total_matrices * MAX_UNION_SIZE * sizeof(int8_t);
    d_memory.result_pool_size = total_matrices * MAX_UNION_SIZE * sizeof(int8_t);
    d_memory.metadata_pool_size = total_matrices * sizeof(int);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_memory.matrix_pool, d_memory.matrix_pool_size));
    CUDA_CHECK(cudaMalloc(&d_memory.result_pool, d_memory.result_pool_size));
    CUDA_CHECK(cudaMalloc(&d_memory.metadata_pool, d_memory.metadata_pool_size));
    
    // Allocate tournament levels
    CUDA_CHECK(cudaMalloc(&d_tournament_levels, num_levels * sizeof(TournamentLevel)));
    
    return true;
}

bool TournamentTree::prepareNextLevel(int level) {
    // Implementation for preparing next level
    return true;
}

void TournamentTree::recycleLevel(int level) {
    // Implementation for recycling memory
}

void TournamentTree::cleanup() {
    if (d_memory.matrix_pool) {
        cudaFree(d_memory.matrix_pool);
        d_memory.matrix_pool = nullptr;
    }
    if (d_memory.result_pool) {
        cudaFree(d_memory.result_pool);
        d_memory.result_pool = nullptr;
    }
    if (d_memory.metadata_pool) {
        cudaFree(d_memory.metadata_pool);
        d_memory.metadata_pool = nullptr;
    }
    if (d_tournament_levels) {
        cudaFree(d_tournament_levels);
        d_tournament_levels = nullptr;
    }
}

// Utility functions
void printMatrix(const Matrix& matrix) {
    std::cout << "Matrix: " << matrix.rows << "x" << matrix.cols << std::endl;
    // Note: This would need host memory access for full implementation
}

void printTournamentLevel(const TournamentLevel& level) {
    std::cout << "Level with " << level.num_matrices << " matrices" << std::endl;
}

bool verifyResults(const std::vector<int8_t>& result) {
    // Basic verification
    return !result.empty() && result.size() <= MAX_UNION_SIZE;
} 