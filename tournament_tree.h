#ifndef TOURNAMENT_TREE_H
#define TOURNAMENT_TREE_H

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// Constants
#define MAX_VECTOR_SIZE 127
#define MAX_THREADS_PER_BLOCK 256
#define MAX_COMBINATIONS 1024
#define MAX_UNION_SIZE 254  // MAX_VECTOR_SIZE * 2

// Data structures
struct Matrix {
    int8_t* data;         // Flattened matrix data
    int rows;             // Number of vectors
    int cols;             // Vector length
    int* row_offsets;     // Starting indices for each row
};

struct TournamentLevel {
    Matrix* matrices;     // Array of matrices at this level
    int num_matrices;     // Number of matrices
    int* result_counts;   // Number of results per matrix
    int8_t* results;      // Flattened results storage
};

struct TournamentMemory {
    int8_t* matrix_pool;    // Pre-allocated matrix storage
    int8_t* result_pool;    // Pre-allocated result storage
    int* metadata_pool;     // Pre-allocated metadata
    size_t matrix_pool_size;
    size_t result_pool_size;
    size_t metadata_pool_size;
};

// Host functions
class TournamentTree {
public:
    TournamentTree();
    ~TournamentTree();
    
    // Initialize tournament with input matrices
    bool initialize(const std::vector<std::vector<int8_t>>& input_matrices);
    
    // Run the tournament
    bool runTournament();
    
    // Get final results
    std::vector<int8_t> getFinalResult();
    
    // Cleanup
    void cleanup();

private:
    TournamentLevel* d_tournament_levels;
    TournamentMemory d_memory;
    int num_levels;
    int num_matrices;
    
    // Helper functions
    bool allocateMemory();
    bool prepareNextLevel(int level);
    void recycleLevel(int level);
};

// CUDA kernel declarations
__global__ void tournamentKernel(
    TournamentLevel* levels,
    int level,
    int max_combinations
);

__device__ void processBattle(
    TournamentLevel level,
    int battle_idx,
    int combination_idx
);

__device__ bool computeSetIntersection(
    Matrix* m1, int row1,
    Matrix* m2, int row2
);

__device__ void storeUnionResult(
    Matrix* m1, int row1,
    Matrix* m2, int row2,
    int battle_idx,
    TournamentLevel* next_level
);

__device__ void storeGlobalResult(
    int8_t* unionSet, 
    int unionSize, 
    int battle_idx,
    TournamentLevel* next_level
);

// Utility functions
void printMatrix(const Matrix& matrix);
void printTournamentLevel(const TournamentLevel& level);
bool verifyResults(const std::vector<int8_t>& result);

#endif // TOURNAMENT_TREE_H 