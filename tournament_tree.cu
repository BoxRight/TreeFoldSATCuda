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
__device__ bool isValidForUnion(Matrix* m1, int row1, Matrix* m2, int row2) {
    bool hasCommonElement = false;
    
    for (int i = 0; i < m1->cols; i++) {
        int8_t val1 = m1->data[m1->row_offsets[row1] + i];
        if (val1 == 0) break;
        
        for (int j = 0; j < m2->cols; j++) {
            int8_t val2 = m2->data[m2->row_offsets[row2] + j];
            if (val2 == 0) break;
            
            if (val1 == val2) {
                hasCommonElement = true;  // Found intersection
            }
            if (val1 == -val2) {
                // Debug output for contradiction detection
                if (threadIdx.x == 0) {
                    printf("CONTRADICTION DETECTED: val1=%d, val2=%d (row1=%d, row2=%d)\n", 
                           (int)val1, (int)val2, row1, row2);
                }
                return false;  // EARLY EXIT - immediate rejection
            }
        }
    }
    
    if (threadIdx.x == 0 && hasCommonElement) {
        printf("VALID UNION: row1=%d, row2=%d (common element found, no contradictions)\n", 
               row1, row2);
    }
    
    return hasCommonElement;
}

__device__ void storeGlobalResult(int8_t* unionSet, int unionSize, int battle_idx, TournamentLevel* next_level) {
    // Store result in next level's matrix
    // This function is called by only one thread (threadIdx.x == 0)
    if (next_level && battle_idx < next_level->num_matrices) {
        // Get the result matrix pointer
        Matrix* result_matrix = &next_level->matrices[battle_idx];
        
        // Read the current matrix structure
        Matrix current_matrix = *result_matrix;
        
        // Copy union data to the matrix's data array
        for (int i = 0; i < unionSize && i < MAX_UNION_SIZE; i++) {
            current_matrix.data[i] = unionSet[i];
        }
        
        // Update matrix properties
        current_matrix.rows = 1;
        current_matrix.cols = unionSize;
        
        // Write back the updated matrix structure
        *result_matrix = current_matrix;
    }
}



__device__ void storeEmptyResult(int battle_idx, TournamentLevel* next_level) {
    // Store empty result when no intersection is found
    if (next_level && battle_idx < next_level->num_matrices) {
        // Get the result matrix pointer
        Matrix* result_matrix = &next_level->matrices[battle_idx];
        
        // Read the current matrix structure
        Matrix current_matrix = *result_matrix;
        
        // Update matrix properties to indicate empty result
        current_matrix.rows = 1;
        current_matrix.cols = 0;  // Empty result
        
        // Write back the updated matrix structure
        *result_matrix = current_matrix;
    }
}





// Device function to process a single battle between two matrices
__device__ void processBattle(TournamentLevel& level, int battle_idx, int combination_idx, 
                            TournamentLevel* next_level, int* global_result_counts) {
    // Get the two matrices for this battle
    if (battle_idx * 2 + 1 >= level.num_matrices) return;
    
    Matrix* m1 = &level.matrices[battle_idx * 2];
    Matrix* m2 = &level.matrices[battle_idx * 2 + 1];
    
    // Use global memory for result counting (thread-safe across all blocks)
    // No shared memory - all synchronization through global atomics
    
    // Calculate which row combination this thread handles
    int total_combinations = m1->rows * m2->rows;
    if (combination_idx < total_combinations) {
        int row1 = combination_idx / m2->rows;
        int row2 = combination_idx % m2->rows;
        
        // Check if this combination is valid for union (has intersection AND no contradictions)
        bool isValid = isValidForUnion(m1, row1, m2, row2);
        
        if (isValid) {
            // Get a slot for this union result using global atomic operation
            int resultSlot = atomicAdd(&global_result_counts[battle_idx], 1);
            
            // Make sure we don't exceed matrix capacity
            if (resultSlot < MAX_RESULTS_PER_BATTLE && next_level) {
                // Build union in private memory first (completely thread-safe)
                int8_t privateUnion[MAX_UNION_SIZE];
                int privateUnionSize = 0;
                
                // Add elements from first vector
                for (int i = 0; i < m1->cols && privateUnionSize < MAX_UNION_SIZE; i++) {
                    int8_t val = m1->data[m1->row_offsets[row1] + i];
                    if (val == 0) break;  // Stop at zero padding
                    privateUnion[privateUnionSize++] = val;
                }
                
                // Add elements from second vector (safe duplicate checking in private memory)
                for (int i = 0; i < m2->cols && privateUnionSize < MAX_UNION_SIZE; i++) {
                    int8_t val = m2->data[m2->row_offsets[row2] + i];
                    if (val == 0) break;  // Stop at zero padding
                    
                    bool isDuplicate = false;
                    for (int j = 0; j < privateUnionSize; j++) {
                        if (privateUnion[j] == val) {
                            isDuplicate = true;
                            if (threadIdx.x == 0) {
                                printf("DUPLICATE detected: val=%d already present\n", (int)val);
                            }
                            break;
                        }
                        if (privateUnion[j] == -val) {
                            isDuplicate = true;
                            if (threadIdx.x == 0) {
                                printf("CONTRADICTION in union: val=%d conflicts with existing %d\n", 
                                       (int)val, (int)privateUnion[j]);
                            }
                            break;
                        }
                    }
                    
                    if (!isDuplicate) {
                        privateUnion[privateUnionSize++] = val;
                    }
                }
                
                // Copy complete union to global memory (atomic write - no race conditions)
                Matrix* result_matrix = &next_level->matrices[battle_idx];
                int rowOffset = resultSlot * MAX_UNION_SIZE;
                
                // Copy union data from private memory to global memory
                for (int i = 0; i < privateUnionSize; i++) {
                    result_matrix->data[rowOffset + i] = privateUnion[i];
                }
                
                // Pad remaining positions with zeros (end marker)
                for (int i = privateUnionSize; i < MAX_UNION_SIZE; i++) {
                    result_matrix->data[rowOffset + i] = 0;
                }
                
                // Set row offset for this union
                result_matrix->row_offsets[resultSlot] = rowOffset;
            }
        }
    }
    
    // Update final row count (only one thread per battle does this)
    if (threadIdx.x == 0 && blockIdx.x == 0 && next_level) {
        Matrix* result_matrix = &next_level->matrices[battle_idx];
        result_matrix->rows = global_result_counts[battle_idx];
        result_matrix->cols = MAX_UNION_SIZE;  // Matrix width (for memory layout)
    }
}

__global__ void tournamentKernel(TournamentLevel* levels, int current_level, int next_level, 
                               int max_combinations, int* global_result_counts) {
    int battle = blockIdx.y;
    int combination = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (current_level >= 0 && battle < levels[current_level].num_matrices / 2) {
        TournamentLevel* next_level_ptr = (next_level >= 0) ? &levels[next_level] : nullptr;
        processBattle(levels[current_level], battle, combination, next_level_ptr, global_result_counts);
    }
}

// Host class implementation
TournamentTree::TournamentTree() : d_tournament_levels(nullptr), num_levels(0), num_matrices(0) {
    // Constructor - no memory pool initialization needed
}

TournamentTree::~TournamentTree() {
    cleanup();
}

bool TournamentTree::initialize(const std::vector<std::vector<std::vector<int8_t>>>& input_matrices) {
    num_matrices = input_matrices.size();
    if (num_matrices == 0) return false;
    
    // Calculate number of tournament levels
    // For tournament tree: we need log2(n) + 1 levels total
    // Level 0: Final result (1 matrix)
    // Level 1: Semi-finals (2 matrices)
    // Level k: Input level (n matrices)
    if (num_matrices == 1) {
        num_levels = 1;  // Special case: single matrix
    } else {
        num_levels = static_cast<int>(ceil(log2(num_matrices))) + 1;
    }
    

    
    // Allocate device memory for tournament levels
    CUDA_CHECK(cudaMalloc(&d_tournament_levels, num_levels * sizeof(TournamentLevel)));
    
    // Initialize levels from bottom to top
    for (int level = num_levels - 1; level >= 0; level--) {
        int matrices_this_level;
        if (level == num_levels - 1) {
            // Input level - use original number of matrices
            matrices_this_level = num_matrices;
        } else {
            // Result levels - each level has half the matrices of the previous level
            TournamentLevel prev_level;
            cudaError_t error = cudaMemcpy(&prev_level, &d_tournament_levels[level + 1], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
            matrices_this_level = (error == cudaSuccess) ? (prev_level.num_matrices / 2) : 1;
        }
        
        // Allocate device memory for matrices array
        Matrix* d_matrices;
        CUDA_CHECK(cudaMalloc(&d_matrices, matrices_this_level * sizeof(Matrix)));
        
        // Allocate device memory for result counts
        int* d_result_counts;
        CUDA_CHECK(cudaMalloc(&d_result_counts, matrices_this_level * sizeof(int)));
        
        // Create host-side level structure
        TournamentLevel h_level;
        h_level.num_matrices = matrices_this_level;
        h_level.matrices = d_matrices;
        h_level.result_counts = d_result_counts;
        h_level.results = nullptr; // Will be set up later if needed
        
        // Copy level structure to device
        CUDA_CHECK(cudaMemcpy(&d_tournament_levels[level], &h_level, sizeof(TournamentLevel), cudaMemcpyHostToDevice));
        

        
        // Initialize matrices for this level
        if (level == num_levels - 1) {
            // Input level - copy input matrices (multi-row format)
            for (int i = 0; i < num_matrices; i++) {
                Matrix h_matrix;
                h_matrix.rows = input_matrices[i].size();  // Number of vectors in this matrix
                h_matrix.cols = MAX_VECTOR_SIZE;  // Max size per vector
                
                // Calculate total data size needed
                int total_data_size = h_matrix.rows * h_matrix.cols;
                
                // Allocate device memory for matrix data
                CUDA_CHECK(cudaMalloc(&h_matrix.data, total_data_size * sizeof(int8_t)));
                
                // Prepare host data with proper row layout
                std::vector<int8_t> host_data(total_data_size, 0);  // Initialize with zeros
                
                // Copy each vector into its row
                for (int row = 0; row < h_matrix.rows; row++) {
                    const auto& vector = input_matrices[i][row];
                    int row_offset = row * h_matrix.cols;
                    
                    // Copy vector data
                    for (int col = 0; col < std::min((int)vector.size(), h_matrix.cols); col++) {
                        host_data[row_offset + col] = vector[col];
                    }
                    // Remaining positions already initialized to 0
                }
                
                // Copy prepared data to device
                CUDA_CHECK(cudaMemcpy(h_matrix.data, host_data.data(), total_data_size * sizeof(int8_t), cudaMemcpyHostToDevice));
                
                // Allocate device memory for row offsets
                CUDA_CHECK(cudaMalloc(&h_matrix.row_offsets, h_matrix.rows * sizeof(int)));
                std::vector<int> row_offsets(h_matrix.rows);
                for (int row = 0; row < h_matrix.rows; row++) {
                    row_offsets[row] = row * h_matrix.cols;
                }
                CUDA_CHECK(cudaMemcpy(h_matrix.row_offsets, row_offsets.data(), h_matrix.rows * sizeof(int), cudaMemcpyHostToDevice));
                
                // Copy matrix to device
                CUDA_CHECK(cudaMemcpy(&d_matrices[i], &h_matrix, sizeof(Matrix), cudaMemcpyHostToDevice));
            }
        } else {
            // Result levels - allocate empty matrices (multi-row capable)
            for (int i = 0; i < matrices_this_level; i++) {
                Matrix h_matrix;
                h_matrix.rows = MAX_RESULTS_PER_BATTLE;  // Allow for multiple result rows
                h_matrix.cols = MAX_UNION_SIZE;
                
                // Allocate device memory for matrix data (enough for multiple rows)
                int total_data_size = h_matrix.rows * h_matrix.cols;
                CUDA_CHECK(cudaMalloc(&h_matrix.data, total_data_size * sizeof(int8_t)));
                CUDA_CHECK(cudaMemset(h_matrix.data, 0, total_data_size * sizeof(int8_t)));
                
                // Allocate device memory for row offsets (for multiple rows)
                CUDA_CHECK(cudaMalloc(&h_matrix.row_offsets, h_matrix.rows * sizeof(int)));
                std::vector<int> row_offsets(h_matrix.rows);
                for (int row = 0; row < h_matrix.rows; row++) {
                    row_offsets[row] = row * h_matrix.cols;
                }
                CUDA_CHECK(cudaMemcpy(h_matrix.row_offsets, row_offsets.data(), h_matrix.rows * sizeof(int), cudaMemcpyHostToDevice));
                
                // Copy matrix to device
                CUDA_CHECK(cudaMemcpy(&d_matrices[i], &h_matrix, sizeof(Matrix), cudaMemcpyHostToDevice));
            }
        }
    }
    
    return true;
}

bool TournamentTree::runTournament() {
    // Special case: single matrix
    if (num_matrices == 1) {
        return true; // No tournament needed
    }
    
    for (int level = num_levels - 1; level > 0; level--) {
        printf("\n=== Processing Tournament Level %d ===\n", level);
        
        // Get the number of matrices at this level
        TournamentLevel current_level;
        cudaError_t error = cudaMemcpy(&current_level, &d_tournament_levels[level], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy level " << level << " from device: " << cudaGetErrorString(error) << std::endl;
            continue;
        }
        
        printf("Level %d: Processing %d matrices in %d battles\n", level, current_level.num_matrices, current_level.num_matrices / 2);
        
        // Show sample matrices from this level
        for (int i = 0; i < std::min(4, current_level.num_matrices); i++) {
            Matrix matrix;
            error = cudaMemcpy(&matrix, &current_level.matrices[i], sizeof(Matrix), cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) {
                printf("  Matrix %d: %d rows, %d cols\n", i, matrix.rows, matrix.cols);
                
                // Show first few rows
                for (int row = 0; row < std::min(3, matrix.rows); row++) {
                    std::vector<int8_t> row_data(matrix.cols);
                    error = cudaMemcpy(row_data.data(), 
                                      matrix.data + row * matrix.cols, 
                                      matrix.cols * sizeof(int8_t), 
                                      cudaMemcpyDeviceToHost);
                    if (error == cudaSuccess) {
                        printf("    Row %d: ", row);
                        for (int j = 0; j < matrix.cols && row_data[j] != 0; j++) {
                            printf("%d ", row_data[j]);
                        }
                        printf("\n");
                    }
                }
                if (matrix.rows > 3) {
                    printf("    ... and %d more rows\n", matrix.rows - 3);
                }
            }
        }
        
        // Validate that next level exists and is properly initialized
        if (level - 1 >= 0) {
            TournamentLevel next_level;
            error = cudaMemcpy(&next_level, &d_tournament_levels[level - 1], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "Failed to copy next level " << (level-1) << " from device: " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            printf("  Next level %d has %d matrices\n", level-1, next_level.num_matrices);
        }
        
        int battles_this_level = current_level.num_matrices / 2; // Number of battles = half the matrices
        
        if (battles_this_level <= 0) continue;
        
        printf("Battles this level: %d\n", battles_this_level);
        printf("Current level has %d matrices\n", current_level.num_matrices);
        
        // Calculate actual maximum combinations needed per battle
        // Note: current_level.matrices is a device pointer, so we need to copy each matrix
        int max_combinations_per_battle = 0;
        for (int battle = 0; battle < battles_this_level; battle++) {
            int m1_idx = battle * 2;
            int m2_idx = battle * 2 + 1;
            printf("Processing battle %d: checking indices %d and %d\n", battle, m1_idx, m2_idx);
            
            if (m1_idx < current_level.num_matrices && m2_idx < current_level.num_matrices) {
                // Copy matrix headers from device to host to get row counts
                Matrix matrix1, matrix2;
                
                printf("Copying matrix %d from device to host...\n", m1_idx);
                error = cudaMemcpy(&matrix1, &current_level.matrices[m1_idx], sizeof(Matrix), cudaMemcpyDeviceToHost);
                if (error != cudaSuccess) {
                    printf("ERROR: Failed to copy matrix %d: %s\n", m1_idx, cudaGetErrorString(error));
                    return false;
                }
                
                printf("Copying matrix %d from device to host...\n", m2_idx);
                error = cudaMemcpy(&matrix2, &current_level.matrices[m2_idx], sizeof(Matrix), cudaMemcpyDeviceToHost);
                if (error != cudaSuccess) {
                    printf("ERROR: Failed to copy matrix %d: %s\n", m2_idx, cudaGetErrorString(error));
                    return false;
                }
                
                int rows1 = matrix1.rows;
                int rows2 = matrix2.rows;
                printf("Matrix %d has %d rows, Matrix %d has %d rows\n", m1_idx, rows1, m2_idx, rows2);
                
                int combinations_needed = rows1 * rows2;
                max_combinations_per_battle = std::max(max_combinations_per_battle, combinations_needed);
                printf("Battle %d: Matrix %d (%d rows) vs Matrix %d (%d rows) = %d combinations\n", 
                       battle, m1_idx, rows1, m2_idx, rows2, combinations_needed);
            } else {
                printf("Battle %d: Invalid matrix indices %d, %d (total matrices: %d)\n", 
                       battle, m1_idx, m2_idx, current_level.num_matrices);
            }
        }
        
        // Apply reasonable limits to prevent excessive memory usage
        const int MAX_COMBINATIONS_LIMIT = 100000; // Reasonable upper limit
        if (max_combinations_per_battle > MAX_COMBINATIONS_LIMIT) {
            printf("WARNING: Combinations per battle (%d) exceeds limit (%d), capping it.\n", 
                   max_combinations_per_battle, MAX_COMBINATIONS_LIMIT);
            max_combinations_per_battle = MAX_COMBINATIONS_LIMIT;
        }
        
        printf("Max combinations per battle: %d\n", max_combinations_per_battle);
        
        // Calculate grid dimensions with safety checks
        int blocks_x = (max_combinations_per_battle + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
        int blocks_y = battles_this_level;
        
        // Check for excessive grid dimensions
        if (blocks_x > 65535 || blocks_y > 65535) {
            printf("ERROR: Grid dimensions too large: (%d, %d). Reducing combinations limit.\n", blocks_x, blocks_y);
            max_combinations_per_battle = std::min(max_combinations_per_battle, 65535 * MAX_THREADS_PER_BLOCK);
            blocks_x = (max_combinations_per_battle + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
            printf("Adjusted max combinations to %d, blocks_x to %d\n", max_combinations_per_battle, blocks_x);
        }
        
        // Grid configuration - calculate based on actual combinations needed
        dim3 grid(blocks_x, blocks_y, 1);
        dim3 block(MAX_THREADS_PER_BLOCK, 1, 1);
        
        printf("Grid: (%d, %d, 1), Block: (%d, 1, 1)\n", blocks_x, blocks_y, MAX_THREADS_PER_BLOCK);
        
        // Allocate global memory for result counts (one per battle)
        int* d_global_result_counts = nullptr;
        error = cudaMalloc(&d_global_result_counts, battles_this_level * sizeof(int));
        if (error != cudaSuccess) {
            printf("ERROR: Failed to allocate global result counts: %s\n", cudaGetErrorString(error));
            return false;
        }
        
        // Initialize result counts to 0
        error = cudaMemset(d_global_result_counts, 0, battles_this_level * sizeof(int));
        if (error != cudaSuccess) {
            printf("ERROR: Failed to initialize global result counts: %s\n", cudaGetErrorString(error));
            cudaFree(d_global_result_counts);
            return false;
        }
        
        // Validate kernel parameters
        if (level - 1 < 0) {
            std::cerr << "ERROR: Invalid next level: " << (level-1) << std::endl;
            cudaFree(d_global_result_counts);
            continue;
        }
        
        // Launch kernel for this tournament level
        tournamentKernel<<<grid, block>>>(d_tournament_levels, level, level - 1, max_combinations_per_battle, d_global_result_counts);
        
        // Check kernel launch error immediately
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(launch_error) << std::endl;
            cudaFree(d_global_result_counts);
            return false;
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check for kernel execution errors
        CUDA_CHECK(cudaGetLastError());
        
        // Clean up global result counts
        cudaFree(d_global_result_counts);
        
        // Debug: Show results after this level
        printf("\n--- Results after Level %d processing ---\n", level);
        if (level - 1 >= 0) {
            TournamentLevel next_level;
            error = cudaMemcpy(&next_level, &d_tournament_levels[level - 1], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) {
                printf("Results in level %d:\n", level - 1);
                
                for (int i = 0; i < next_level.num_matrices; i++) {
                    Matrix result_matrix;
                    error = cudaMemcpy(&result_matrix, &next_level.matrices[i], sizeof(Matrix), cudaMemcpyDeviceToHost);
                    if (error == cudaSuccess) {
                        printf("  Result matrix %d: %d rows, %d cols\n", i, result_matrix.rows, result_matrix.cols);
                        
                        if (result_matrix.rows == 0) {
                            printf("    EMPTY RESULT - No intersections found!\n");
                        } else {
                            // Show first few result rows
                            for (int row = 0; row < std::min(3, result_matrix.rows); row++) {
                                std::vector<int8_t> row_data(result_matrix.cols);
                                error = cudaMemcpy(row_data.data(), 
                                                  result_matrix.data + row * result_matrix.cols, 
                                                  result_matrix.cols * sizeof(int8_t), 
                                                  cudaMemcpyDeviceToHost);
                                if (error == cudaSuccess) {
                                    printf("    Result row %d: ", row);
                                    for (int j = 0; j < result_matrix.cols && row_data[j] != 0; j++) {
                                        printf("%d ", row_data[j]);
                                    }
                                    printf("\n");
                                }
                            }
                            if (result_matrix.rows > 3) {
                                printf("    ... and %d more rows\n", result_matrix.rows - 3);
                            }
                        }
                    }
                }
            }
        }
        printf("=== End Level %d ===\n", level);
    }
    
    return true;
}

std::vector<int8_t> TournamentTree::getFinalResult() {
    std::vector<int8_t> result;
    
    if (num_levels == 0 || !d_tournament_levels) return result;
    
    // For single matrix, return the original matrix from the last level
    int result_level = (num_matrices == 1) ? (num_levels - 1) : 0;
    
    // Get the final result from the appropriate level
    TournamentLevel final_level;
    cudaError_t error = cudaMemcpy(&final_level, &d_tournament_levels[result_level], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error getting final level: " << cudaGetErrorString(error) << std::endl;
        return result;
    }
    
    if (final_level.num_matrices > 0) {
        Matrix final_matrix;
        error = cudaMemcpy(&final_matrix, final_level.matrices, sizeof(Matrix), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error getting final matrix: " << cudaGetErrorString(error) << std::endl;
            return result;
        }
        
        if (final_matrix.cols > 0) {
            // Read all data first
            std::vector<int8_t> full_data(final_matrix.cols);
            error = cudaMemcpy(full_data.data(), final_matrix.data, final_matrix.cols * sizeof(int8_t), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error getting final data: " << cudaGetErrorString(error) << std::endl;
                result.clear();
                return result;
            }
            
            // Extract only non-zero elements (stop at first zero)
            for (int i = 0; i < full_data.size() && full_data[i] != 0; i++) {
                result.push_back(full_data[i]);
            }
        } else {
            // Empty result - no intersections found
            result.clear();
        }
    }
    
    return result;
}

std::vector<std::vector<int8_t>> TournamentTree::getAllFinalResults() {
    std::vector<std::vector<int8_t>> results;
    
    if (num_levels == 0 || !d_tournament_levels) {
        return results;
    }
    
    // Get final level (level 0)
    TournamentLevel final_level;
    cudaError_t error = cudaMemcpy(&final_level, d_tournament_levels, sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "CUDA error getting final level: " << cudaGetErrorString(error) << std::endl;
        return results;
    }
    
    if (final_level.num_matrices > 0) {
        // Get final matrix
        Matrix final_matrix;
        error = cudaMemcpy(&final_matrix, final_level.matrices, sizeof(Matrix), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "CUDA error getting final matrix: " << cudaGetErrorString(error) << std::endl;
            return results;
        }
        
        std::cout << "Final matrix has " << final_matrix.rows << " rows" << std::endl;
        
        // Read each row as a separate union
        for (int row = 0; row < final_matrix.rows; row++) {
            std::vector<int8_t> union_result;
            
            // Read row data
            std::vector<int8_t> row_data(MAX_UNION_SIZE);
            error = cudaMemcpy(row_data.data(), 
                              final_matrix.data + row * MAX_UNION_SIZE, 
                              MAX_UNION_SIZE * sizeof(int8_t), 
                              cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                std::cerr << "CUDA error getting row " << row << " data: " << cudaGetErrorString(error) << std::endl;
                continue;
            }
            
            // Extract non-zero elements (stop at first zero)
            for (int i = 0; i < MAX_UNION_SIZE && row_data[i] != 0; i++) {
                union_result.push_back(row_data[i]);
            }
            
            if (!union_result.empty()) {
                results.push_back(union_result);
                std::cout << "Row " << row << " union: ";
                for (int val : union_result) {
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    
    return results;
}

bool TournamentTree::prepareNextLevel(int level) {
    // Implementation for preparing next level
    return true;
}

void TournamentTree::recycleLevel(int level) {
    // Implementation for recycling memory
}

void TournamentTree::cleanup() {
    if (d_tournament_levels) {
        // Free nested structures
        for (int level = 0; level < num_levels; level++) {
            TournamentLevel h_level;
            cudaError_t error = cudaMemcpy(&h_level, &d_tournament_levels[level], sizeof(TournamentLevel), cudaMemcpyDeviceToHost);
            if (error == cudaSuccess) {
                // Free matrices in this level
                for (int i = 0; i < h_level.num_matrices; i++) {
                    Matrix h_matrix;
                    error = cudaMemcpy(&h_matrix, &h_level.matrices[i], sizeof(Matrix), cudaMemcpyDeviceToHost);
                    if (error == cudaSuccess) {
                        if (h_matrix.data) cudaFree(h_matrix.data);
                        if (h_matrix.row_offsets) cudaFree(h_matrix.row_offsets);
                    }
                }
                // Free matrices array and result counts
                if (h_level.matrices) cudaFree(h_level.matrices);
                if (h_level.result_counts) cudaFree(h_level.result_counts);
                if (h_level.results) cudaFree(h_level.results);
            }
        }
        
        // Free the main tournament levels array
        cudaFree(d_tournament_levels);
        d_tournament_levels = nullptr;
    }
    
    num_levels = 0;
    num_matrices = 0;
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