# CUDA Tournament Tree Implementation

A high-performance CUDA implementation of a tournament tree algorithm for computing set intersections and unions across multiple matrices of int8_t vectors.

## Overview

This implementation processes multiple matrices in a tournament-style bracket system where:
1. **Intersection Check**: Each "battle" between two matrices checks if any vectors from the first matrix intersect with any vectors from the second matrix
2. **Union Computation**: If an intersection exists, the union of the intersecting vectors is computed and passed to the next tournament level
3. **Tree Structure**: The tournament proceeds in a logarithmic tree structure until a final result is obtained

## Features

- **Optimized CUDA Kernels**: Efficient parallel processing using shared memory and coalesced memory access
- **Memory Management**: Pre-allocated memory pools and recycling between rounds
- **Scalable Design**: Supports up to 127 elements per vector (int8_t) and configurable matrix sizes
- **Comprehensive Testing**: Multiple test cases including edge cases and performance benchmarks

## File Structure

```
tensor_processor/
├── tournament_tree.h          # Header file with data structures and declarations
├── tournament_tree.cu         # Main CUDA implementation
├── test_tournament.cu         # Comprehensive test suite
├── Makefile                   # Build system
└── README.md                  # This file
```

## Requirements

- **NVIDIA CUDA Toolkit** (version 8.0 or higher)
- **Compatible GPU** with compute capability 5.2 or higher
- **C++14 compatible compiler**

## Building the Project

### Using Make (Recommended)

```bash
# Build the project
make

# Build and run tests
make test

# Clean build artifacts
make clean

# Build with debug symbols
make debug

# Build with profiling information
make profile

# View help
make help
```

### Manual Compilation

```bash
# Create directories
mkdir -p build bin

# Compile CUDA files
nvcc -arch=sm_52 -std=c++14 -O3 -c tournament_tree.cu -o build/tournament_tree.o
nvcc -arch=sm_52 -std=c++14 -O3 -c test_tournament.cu -o build/test_tournament.o

# Link executable
nvcc -arch=sm_52 -std=c++14 -O3 -o bin/tournament_test build/tournament_tree.o build/test_tournament.o
```

## Usage

### Basic Example

```cpp
#include "tournament_tree.h"
#include <vector>

int main() {
    // Create input matrices
    std::vector<std::vector<int8_t>> matrices = {
        {1, 2, 3, 4},
        {3, 4, 5, 6},
        {5, 6, 7, 8},
        {7, 8, 9, 10}
    };
    
    // Initialize tournament
    TournamentTree tournament;
    if (!tournament.initialize(matrices)) {
        std::cerr << "Failed to initialize tournament" << std::endl;
        return 1;
    }
    
    // Run tournament
    if (!tournament.runTournament()) {
        std::cerr << "Failed to run tournament" << std::endl;
        return 1;
    }
    
    // Get results
    auto result = tournament.getFinalResult();
    
    // Process results
    std::cout << "Final result: ";
    for (int8_t val : result) {
        std::cout << (int)val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### Running Tests

```bash
# Run all tests
./bin/tournament_test

# Or using make
make test
```

## Test Cases

The test suite includes:

1. **Basic Intersection**: Simple matrices with known intersections
2. **Multiple Intersections**: Matrices with multiple overlapping elements
3. **No Intersections**: Matrices with no common elements
4. **Single Matrix**: Edge case with only one matrix
5. **Large Vectors**: Testing with vectors close to maximum size (127 elements)
6. **Random Data**: Testing with randomly generated matrices
7. **Performance**: Benchmarking with larger datasets

## Algorithm Details

### Tournament Structure

The tournament operates in `log₂(n)` levels where `n` is the number of input matrices:

```
Level 3: [M1] [M2] [M3] [M4] [M5] [M6] [M7] [M8]
Level 2:    [R1]    [R2]    [R3]    [R4]
Level 1:       [R5]           [R6]
Level 0:          [Final Result]
```

### Key Kernels

1. **`tournamentKernel`**: Main orchestrator that processes tournament levels
2. **`processBattle`**: Handles individual battles between matrix pairs
3. **`computeSetIntersection`**: Efficient parallel intersection detection
4. **`storeUnionResult`**: Computes and stores union of intersecting vectors

### Memory Optimization

- **Shared Memory**: Vectors loaded once per thread block
- **Coalesced Access**: Sequential memory patterns for optimal bandwidth
- **Memory Pools**: Pre-allocated storage to avoid runtime allocation
- **Atomic Operations**: Thread-safe union computation

## Performance Considerations

### Optimal Configuration

- **Block Size**: 256 threads per block (configurable via `MAX_THREADS_PER_BLOCK`)
- **Vector Size**: Up to 127 elements per vector (int8_t limitation)
- **Memory Usage**: Scales with number of matrices and vector sizes

### Scalability

The implementation scales well with:
- Number of matrices (logarithmic tournament structure)
- Vector sizes (up to hardware limits)
- GPU compute capability (automatic optimization)

## Configuration

Key constants in `tournament_tree.h`:

```cpp
#define MAX_VECTOR_SIZE 127        // Maximum elements per vector
#define MAX_THREADS_PER_BLOCK 256  // Threads per CUDA block
#define MAX_COMBINATIONS 1024      // Maximum combinations per battle
#define MAX_UNION_SIZE 254         // Maximum union size
```

## Debugging

### Common Issues

1. **CUDA Out of Memory**: Reduce `MAX_VECTOR_SIZE` or number of matrices
2. **No CUDA Device**: Ensure CUDA-capable GPU is available
3. **Compilation Errors**: Check CUDA toolkit installation and GPU compatibility

### Debug Build

```bash
make debug
```

### Profiling

```bash
make profile
# Use with nvprof or Nsight Systems
nvprof ./bin/tournament_test
```

## License

This implementation is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

Based on the tournament tree algorithm design with CUDA optimizations for parallel set operations. 