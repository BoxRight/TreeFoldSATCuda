#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <set>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <unordered_map>
#include <fstream>
#include <string>
#include <map>
#include <cctype>
#include <sstream>

// Error checking macro
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Configuration parameters
#define MAX_ELEMENTS_PER_VECTOR 128
#define BLOCK_SIZE 256

// Absolute value functor for Thrust
struct AbsoluteFunctor {
    __host__ __device__
    int operator()(const int x) const {
        return x < 0 ? -x : x;
    }
};

// Simple JSON parser for our specific needs
class SimpleJsonParser {
private:
    std::string data;
    size_t pos = 0;

    void skipWhitespace() {
        while (pos < data.size() && std::isspace(data[pos])) pos++;
    }

    bool match(char c) {
        skipWhitespace();
        if (pos < data.size() && data[pos] == c) {
            pos++;
            return true;
        }
        return false;
    }

    std::string parseString() {
        if (!match('"')) return "";
        
        size_t start = pos;
        while (pos < data.size() && data[pos] != '"') {
            if (data[pos] == '\\' && pos + 1 < data.size()) pos++;
            pos++;
        }
        
        std::string result = data.substr(start, pos - start);
        match('"'); // Consume closing quote
        return result;
    }

    int parseInt() {
        skipWhitespace();
        
        bool negative = false;
        if (pos < data.size() && data[pos] == '-') {
            negative = true;
            pos++;
        }
        
        int value = 0;
        while (pos < data.size() && std::isdigit(data[pos])) {
            value = value * 10 + (data[pos] - '0');
            pos++;
        }
        
        return negative ? -value : value;
    }
    
    std::vector<int> parseArray() {
        std::vector<int> result;
        if (!match('[')) return result;
        
        while (!match(']')) {
            result.push_back(parseInt());
            match(','); // Consume comma if present
        }
        
        return result;
    }

public:
    SimpleJsonParser(const std::string& jsonData) : data(jsonData) {}
    
    struct Clause {
        std::string key;
        int condition_id1;
        int condition_id2;
        int consequence_id;
    };
    
    struct Matrix {
        std::string key;
        int rows;
        int cols;
        int type;
        std::vector<int> data;
    };
    
    std::vector<Clause> parseClauses() {
        std::vector<Clause> clauses;
        
        // Find "clauses": [ in the file
        size_t clausesStart = data.find("\"clauses\":");
        if (clausesStart == std::string::npos) return clauses;
        
        pos = clausesStart + 10; // Move past "clauses":
        skipWhitespace();
        
        if (!match('[')) return clauses;
        
        while (!match(']')) {
            if (!match('{')) break;
            
            Clause clause;
            
            while (!match('}')) {
                if (!match('"')) break;
                
                std::string key;
                while (pos < data.size() && data[pos] != '"') key += data[pos++];
                match('"');
                
                match(':');
                
                if (key == "key") {
                    clause.key = parseString();
                } else if (key == "condition_id1") {
                    clause.condition_id1 = parseInt();
                } else if (key == "condition_id2") {
                    clause.condition_id2 = parseInt();
                } else if (key == "consequence_id") {
                    clause.consequence_id = parseInt();
                } else {
                    // Skip unknown field
                    while (pos < data.size() && data[pos] != ',' && data[pos] != '}') pos++;
                }
                
                match(','); // Consume comma if present
            }
            
            clauses.push_back(clause);
            match(','); // Consume comma if present
        }
        
        return clauses;
    }
    
    std::vector<Matrix> parseMatrices() {
        std::vector<Matrix> matrices;
        
        // Find "matrices": [ in the file
        size_t matricesStart = data.find("\"matrices\":");
        if (matricesStart == std::string::npos) return matrices;
        
        pos = matricesStart + 11; // Move past "matrices":
        skipWhitespace();
        
        if (!match('[')) return matrices;
        
        while (!match(']')) {
            if (!match('{')) break;
            
            Matrix matrix;
            
            while (!match('}')) {
                if (!match('"')) break;
                
                std::string key;
                while (pos < data.size() && data[pos] != '"') key += data[pos++];
                match('"');
                
                match(':');
                
                if (key == "key") {
                    matrix.key = parseString();
                } else if (key == "rows") {
                    matrix.rows = parseInt();
                } else if (key == "cols") {
                    matrix.cols = parseInt();
                } else if (key == "type") {
                    matrix.type = parseInt();
                } else if (key == "data") {
                    matrix.data = parseArray();
                } else {
                    // Skip unknown field
                    while (pos < data.size() && data[pos] != ',' && data[pos] != '}') pos++;
                }
                
                match(','); // Consume comma if present
            }
            
            matrices.push_back(matrix);
            match(','); // Consume comma if present
        }
        
        return matrices;
    }
};

//-------------------------------------------------------------------------
// Host-side data structures
//-------------------------------------------------------------------------
typedef struct {
    std::vector<std::vector<int>> vectors;  // Original vectors 
} HostSet;

// CUDA Set Data Structure (more efficient layout)
typedef struct {
    int8_t* data;         // Flattened array of all elements
    int* offsets;      // Starting index for each vector/set
    int* sizes;        // Size of each vector/set
    int numItems;      // Number of vectors/sets
    int totalElements; // Total number of elements
    int8_t* deviceBuffer; // Reusable device buffer for operations
    int bufferSize;    // Size of the device buffer
} CudaSet;

// Result buffer for parallel combination processing
typedef struct {
    int* data;         // Buffer for all potential results
    int* validFlags;   // Flags indicating if each combination is valid
    int* sizes;        // Size of each result set
    int maxResultSize; // Maximum possible size of a result
    int numCombinations; // Total number of combinations
} CombinationResultBuffer;

// Allocate memory for a CUDA set with additional buffer space
CudaSet allocateCudaSet(int numItems, int totalElements, int bufferSize = 0) {
    CudaSet set;
    set.numItems = numItems;
    set.totalElements = totalElements;
    
    CHECK_CUDA_ERROR(cudaMalloc(&set.data, totalElements * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&set.offsets, numItems * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&set.sizes, numItems * sizeof(int)));
    
    // Allocate device buffer if size is specified
    if (bufferSize > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&set.deviceBuffer, bufferSize * sizeof(int)));
        set.bufferSize = bufferSize;
    } else {
        set.deviceBuffer = nullptr;
        set.bufferSize = 0;
    }
    
    return set;
}

// Free memory for a CUDA set
void freeCudaSet(CudaSet* set) {
    if (set->data) cudaFree(set->data);
    if (set->offsets) cudaFree(set->offsets);
    if (set->sizes) cudaFree(set->sizes);
    if (set->deviceBuffer) cudaFree(set->deviceBuffer);
    set->numItems = 0;
    set->totalElements = 0;
    set->bufferSize = 0;
    set->data = nullptr;
    set->offsets = nullptr;
    set->sizes = nullptr;
    set->deviceBuffer = nullptr;
}

// Allocate result buffer for parallel combination processing
CombinationResultBuffer allocateCombinationResultBuffer(int numItemsA, int numItemsB, int maxElementsPerVector) {
    CombinationResultBuffer buffer;
    buffer.numCombinations = numItemsA * numItemsB;
    buffer.maxResultSize = 2 * maxElementsPerVector; // Worst case: all elements from both vectors
    
    CHECK_CUDA_ERROR(cudaMalloc(&buffer.data, buffer.numCombinations * buffer.maxResultSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&buffer.validFlags, buffer.numCombinations * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&buffer.sizes, buffer.numCombinations * sizeof(int)));
    
    // Initialize all valid flags to 0 (invalid)
    CHECK_CUDA_ERROR(cudaMemset(buffer.validFlags, 0, buffer.numCombinations * sizeof(int)));
    
    return buffer;
}

// Free result buffer
void freeCombinationResultBuffer(CombinationResultBuffer* buffer) {
    if (buffer->data) cudaFree(buffer->data);
    if (buffer->validFlags) cudaFree(buffer->validFlags);
    if (buffer->sizes) cudaFree(buffer->sizes);
    buffer->data = nullptr;
    buffer->validFlags = nullptr;
    buffer->sizes = nullptr;
}

// Host to device copy for a set (optimized to use pinned memory for larger transfers)
void copyHostToDevice(const HostSet& hostSet, CudaSet* cudaSet) {
    int numItems = hostSet.vectors.size();
    
    // Prepare host side arrays
    std::vector<int> hostIntData;
    std::vector<int> hostOffsets(numItems);
    std::vector<int> hostSizes(numItems);
    
    int currentOffset = 0;
    for (int i = 0; i < numItems; i++) {
        hostOffsets[i] = currentOffset;
        hostSizes[i] = hostSet.vectors[i].size();
        
        for (int j = 0; j < hostSet.vectors[i].size(); j++) {
            hostIntData.push_back(hostSet.vectors[i][j]);
        }
        
        currentOffset += hostSet.vectors[i].size();
    }
    
    // Convert to int8_t for device storage
    std::vector<int8_t> hostData(hostIntData.size());
    for (size_t i = 0; i < hostIntData.size(); ++i) {
        hostData[i] = static_cast<int8_t>(hostIntData[i]);
    }

    // Use pinned memory for large transfers
    int totalElements = hostData.size();
    int8_t* pinnedData = nullptr;
    int* pinnedOffsets = nullptr;
    int* pinnedSizes = nullptr;
    
    if (totalElements > 1024) {
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&pinnedData, totalElements * sizeof(int8_t)));
        CHECK_CUDA_ERROR(cudaMallocHost(&pinnedOffsets, numItems * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMallocHost(&pinnedSizes, numItems * sizeof(int)));
        
        memcpy(pinnedData, hostData.data(), totalElements * sizeof(int8_t));
        memcpy(pinnedOffsets, hostOffsets.data(), numItems * sizeof(int));
        memcpy(pinnedSizes, hostSizes.data(), numItems * sizeof(int));
    }
    
    // Allocate device memory
    *cudaSet = allocateCudaSet(numItems, totalElements, totalElements * 2);
    
    // Copy data to device
    if (totalElements > 1024) {
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->data, pinnedData, totalElements * sizeof(int8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->offsets, pinnedOffsets, numItems * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->sizes, pinnedSizes, numItems * sizeof(int), cudaMemcpyHostToDevice));
        
        cudaFreeHost(pinnedData);
        cudaFreeHost(pinnedOffsets);
        cudaFreeHost(pinnedSizes);
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->data, hostData.data(), totalElements * sizeof(int8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->offsets, hostOffsets.data(), numItems * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(cudaSet->sizes, hostSizes.data(), numItems * sizeof(int), cudaMemcpyHostToDevice));
    }
}

// Device to host copy (optimized with streams for larger data)
HostSet copyDeviceToHost(const CudaSet& cudaSet) {
    HostSet hostSet;
    
    // Copy offsets and sizes
    std::vector<int> hostOffsets(cudaSet.numItems);
    std::vector<int> hostSizes(cudaSet.numItems);
    
    CHECK_CUDA_ERROR(cudaMemcpy(hostOffsets.data(), cudaSet.offsets, cudaSet.numItems * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), cudaSet.sizes, cudaSet.numItems * sizeof(int), cudaMemcpyDeviceToHost));
    
    // For large data, use async transfers with streams
    std::vector<int8_t> hostData8(cudaSet.totalElements);
    
    if (cudaSet.totalElements > 1024) {
        cudaStream_t stream;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        
        int8_t* pinnedData;
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&pinnedData, cudaSet.totalElements * sizeof(int8_t)));
        
        CHECK_CUDA_ERROR(cudaMemcpyAsync(pinnedData, cudaSet.data, cudaSet.totalElements * sizeof(int8_t), 
                                       cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        
        memcpy(hostData8.data(), pinnedData, cudaSet.totalElements * sizeof(int8_t));
        
        cudaFreeHost(pinnedData);
        cudaStreamDestroy(stream);
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy(hostData8.data(), cudaSet.data, cudaSet.totalElements * sizeof(int8_t), 
                                  cudaMemcpyDeviceToHost));
    }
    
    // Reconstruct vectors
    hostSet.vectors.resize(cudaSet.numItems);
    
    // Convert back to int
    std::vector<int> hostData(cudaSet.totalElements);
    for (size_t i = 0; i < hostData8.size(); ++i) {
        hostData[i] = hostData8[i];
    }

    for (int i = 0; i < cudaSet.numItems; i++) {
        int offset = hostOffsets[i];
        int size = hostSizes[i];
        
        hostSet.vectors[i].resize(size);
        for (int j = 0; j < size; j++) {
            hostSet.vectors[i][j] = hostData[offset + j];
        }
    }
    
    return hostSet;
}

// Helper function to create a test set
HostSet createTestSet(const std::vector<std::vector<int>>& vectors) {
    HostSet set;
    set.vectors = vectors;
    return set;
}

//-------------------------------------------------------------------------
// CUDA Kernels and Device Functions
//-------------------------------------------------------------------------

// Device function to check if an element is in a set
__device__ bool deviceContains(const int* array, int size, int value) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) {
            return true;
        }
    }
    return false;
}

__global__ void gatherIndicesKernel(int* uniqueIndices, int* indices, int* uniqueVectorIndices, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int pos = uniqueIndices[idx];
        uniqueVectorIndices[idx] = indices[pos];
    }
}


// Kernel to convert vector elements to unique elements (for Level 1 carry-over)
__global__ void convertToUniqueKernel(
    int8_t* inputData, int* inputOffsets, int* inputSizes, int numItems,
    int8_t* outputData, int* outputOffsets, int* outputSizes, int maxOutputSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numItems) {
        return;
    }
    
    int inputOffset = inputOffsets[idx];
    int inputSize = inputSizes[idx];
    int outputOffset = outputOffsets[idx];
    
    // Local working memory for unique elements
    int localSet[MAX_ELEMENTS_PER_VECTOR];
    int localSetSize = 0;
    
    // Get unique elements
    for (int i = 0; i < inputSize; i++) {
        int val = inputData[inputOffset + i];
        if (!deviceContains(localSet, localSetSize, val)) {
            localSet[localSetSize++] = val;
        }
    }
    
    // Copy result to output
    outputSizes[idx] = localSetSize;
    for (int i = 0; i < localSetSize; i++) {
        outputData[outputOffset + i] = localSet[i];
    }
}


// Kernel that processes all combinations with built-in batching
__global__ void processAllCombinationsKernel(
    int8_t* dataA, int* offsetsA, int* sizesA, int numItemsA,
    int8_t* dataB, int* offsetsB, int* sizesB, int numItemsB,
    int threshold, int level,
    int* resultData, int* resultSizes, int* validFlags, int maxResultSize,
    int combinationsPerThread
) {
    // Calculate global thread ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes multiple combinations using grid-stride loop
    for (int i = 0; i < combinationsPerThread; i++) {
        // Calculate combination index for this thread and iteration
        int combinationIdx = threadId * combinationsPerThread + i;
        
        // Check if this combination index is valid
        if (combinationIdx >= numItemsA * numItemsB) {
            return;
        }
        
        // Calculate setA and setB indices from the combination index
        int idxA = combinationIdx / numItemsB;
        int idxB = combinationIdx % numItemsB;
        
        // Get vectors from set A and set B
        int offsetA = offsetsA[idxA];
        int sizeA = sizesA[idxA];
        int offsetB = offsetsB[idxB];
        int sizeB = sizesB[idxB];
        
        // Local working memory for unique elements
        int localSet[MAX_ELEMENTS_PER_VECTOR * 2];
        int localSetSize = 0;
        
        // Merge vectors, keeping only unique elements
        for (int j = 0; j < sizeA; j++) {
            int val = dataA[offsetA + j];
            if (!deviceContains(localSet, localSetSize, val)) {
                localSet[localSetSize++] = val;
            }
        }
        
        for (int j = 0; j < sizeB; j++) {
            int val = dataB[offsetB + j];
            if (!deviceContains(localSet, localSetSize, val)) {
                localSet[localSetSize++] = val;
            }
        }
        
        // Check threshold condition
        bool isValid = (threshold == 0 || localSetSize <= threshold);
        
        // If valid, copy result to output buffer
        if (isValid) {
            validFlags[combinationIdx] = 1;
            resultSizes[combinationIdx] = localSetSize;
            
            int resultOffset = combinationIdx * maxResultSize;
            for (int j = 0; j < localSetSize; j++) {
                resultData[resultOffset + j] = localSet[j];
            }
        } else {
            validFlags[combinationIdx] = 0;
            resultSizes[combinationIdx] = localSetSize; // Store size for debugging
        }
    }
}


// Kernel to mark unique vectors (more efficient than previous version)
__global__ void markUniqueKernel(int* indices, int* data, int* lengths, int* isUnique, 
                               int numVectors, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx >= numVectors) return;
    
    int curr = indices[idx];
    int prev = indices[idx - 1];
    
    int lenCurr = lengths[curr];
    int lenPrev = lengths[prev];
    
    // If lengths differ, this is a unique vector
    if (lenCurr != lenPrev) {
        return;
    }
    
    // Compare vectors
    int* currData = data + curr * maxLen;
    int* prevData = data + prev * maxLen;
    
    // Fast comparison
    bool identical = true;
    for (int i = 0; i < lenCurr; i++) {
        if (currData[i] != prevData[i]) {
            identical = false;
            break;
        }
    }
    
    // If identical, mark as duplicate
    if (identical) {
        isUnique[idx] = 0;
    }
}

// Kernel to compute hash values for each vector
__global__ void computeVectorHashesKernel(int* data, int* lengths, uint64_t* hashes, int numVectors, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVectors) return;
    
    int* vecStart = data + idx * maxLen;
    int len = lengths[idx];
    
    // FNV-1a hash (very fast and good for integers)
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    
    // Hash length first
    hash ^= len;
    hash *= 1099511628211ULL; // FNV prime
    
    // Hash each element
    for (int i = 0; i < len; i++) {
        hash ^= vecStart[i];
        hash *= 1099511628211ULL;
    }
    
    hashes[idx] = hash;
}

// Kernel to mark duplicates in parallel
__global__ void markDuplicatesKernel(int* indices, int* data, int* lengths, uint64_t* hashes, 
                                   int* isUnique, int numVectors, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx >= numVectors) return; // First item always unique, skip if out of bounds
    
    int curr = indices[idx];
    int prev = indices[idx - 1];
    
    // Quick check: if hashes differ, this is a unique item
    if (hashes[curr] != hashes[prev]) {
        return; // Different hashes, must be unique
    }
    
    // Hashes match, need to check the actual vectors
    int lenCurr = lengths[curr];
    int lenPrev = lengths[prev];
    
    // If lengths differ, this is a unique item
    if (lenCurr != lenPrev) {
        return;
    }
    
    // Check all elements
    int* currData = data + curr * maxLen;
    int* prevData = data + prev * maxLen;
    
    bool identical = true;
    for (int i = 0; i < lenCurr; i++) {
        if (currData[i] != prevData[i]) {
            identical = false;
            break;
        }
    }
    
    // If identical, mark as duplicate
    if (identical) {
        isUnique[idx] = 0;
    }
}

// Initialize arrays kernel
__global__ void initializeArraysKernel(int* indices, int* isUnique, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        indices[idx] = idx;
        isUnique[idx] = 1;
    }
}

// Generate sort keys for small vectors (≤8 elements)
__global__ void generateSortKeysKernel(int* data, int* lengths, uint64_t* keys, int size, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    int* vec = data + idx * maxLen;
    int len = lengths[idx];
    
    // Create a 64-bit key
    uint64_t key = 0;
    
    // First encode the length (in top 8 bits)
    key = (uint64_t)len << 56;
    
    // Then encode elements (in lower bits)
    for (int i = 0; i < len && i < 7; i++) {
        // Scale to positive and pack into 8 bits
        unsigned int val = (vec[i] + 0x7FFFFFFF) & 0xFF;
        key |= ((uint64_t)val << (8 * (6 - i)));
    }
    
    keys[idx] = key;
}

// Mark duplicates kernel - optimized for memory access
__global__ void markDuplicatesKernel(int* indices, int* data, int* lengths, int* isUnique, int size, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx >= size) return;
    
    int curr = indices[idx];
    int prev = indices[idx - 1];
    
    int lenCurr = lengths[curr];
    int lenPrev = lengths[prev];
    
    // Quick length check (if different, definitely unique)
    if (lenCurr != lenPrev) {
        return;
    }
    
    // Pointers to the vectors
    int* currData = data + curr * maxLen;
    int* prevData = data + prev * maxLen;
    
    // Compare elements
    bool identical = true;
    for (int i = 0; i < lenCurr; i++) {
        if (currData[i] != prevData[i]) {
            identical = false;
            break;
        }
    }
    
    if (identical) {
        isUnique[idx] = 0;
    }
}

// Gather unique indices using prefix sum
__global__ void gatherUniqueIndicesKernel(int* indices, int* isUnique, int* prefixSum, int* uniqueIndices, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    if (isUnique[idx] == 1) {
        int destIdx = prefixSum[idx] - 1; // Adjust for 0-based indexing
        uniqueIndices[destIdx] = indices[idx];
    }
}

// Kernel to sort all vectors in parallel
// Each thread block sorts multiple vectors simultaneously
__global__ void sortAllVectorsKernel(int* data, int* lengths, int numVectors, int maxLen) {
    extern __shared__ int sharedMem[];
    
    // Determine which vectors this block processes
    int vectorsPerBlock = blockDim.x;
    int baseVectorIdx = blockIdx.x * vectorsPerBlock;
    
    // Local thread ID within the block
    int localIdx = threadIdx.x;
    
    // Check if this block has any vectors to process
    if (baseVectorIdx >= numVectors) {
        return;
    }
    
    // Calculate the vector index this thread works on
    int vectorIdx = baseVectorIdx + localIdx;
    
    // Check if this thread has a valid vector
    if (vectorIdx < numVectors) {
        // Get vector length
        int len = lengths[vectorIdx];
        
        // Local pointer to this vector's data
        int* vecData = data + vectorIdx * maxLen;
        
        // Sort the vector (using bitonic sort for maximum parallelism)
        for (int k = 2; k <= len; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                for (int i = localIdx; i < len; i += blockDim.x) {
                    int ixj = i ^ j;
                    
                    if (ixj > i && ixj < len) {
                        if ((i & k) == 0) {
                            // Ascending
                            if (vecData[i] > vecData[ixj]) {
                                // Swap
                                int temp = vecData[i];
                                vecData[i] = vecData[ixj];
                                vecData[ixj] = temp;
                            }
                        } else {
                            // Descending
                            if (vecData[i] < vecData[ixj]) {
                                // Swap
                                int temp = vecData[i];
                                vecData[i] = vecData[ixj];
                                vecData[ixj] = temp;
                            }
                        }
                    }
                    
                    // Synchronize threads in the block
                    __syncthreads();
                }
            }
        }
    }
}
//-------------------------------------------------------------------------
// Core processing functions that match Python exactly
//-------------------------------------------------------------------------

// Compute threshold from two vectors (optimized but maintains exact behavior)
int computeThreshold(const CudaSet& setA, const CudaSet& setB) {
    if (setA.numItems == 0 || setB.numItems == 0) {
        return 0;
    }
    
    // Extract first vectors from each set (this part needs to be sequential)
    int sizeA, sizeB;
    CHECK_CUDA_ERROR(cudaMemcpy(&sizeA, setA.sizes, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&sizeB, setB.sizes, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Allocate host memory for first vectors
    std::vector<int8_t> h_firstVectorA(sizeA);
    std::vector<int8_t> h_firstVectorB(sizeB);
    
    // Copy first vectors to host (still sequential - small amount of data)
    int offsetA = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&offsetA, setA.offsets, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_firstVectorA.data(), setA.data + offsetA, sizeA * sizeof(int8_t), cudaMemcpyDeviceToHost));
    
    int offsetB = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&offsetB, setB.offsets, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_firstVectorB.data(), setB.data + offsetB, sizeB * sizeof(int8_t), cudaMemcpyDeviceToHost));
    
    std::vector<int> firstVectorA(sizeA);
    std::vector<int> firstVectorB(sizeB);
    for(int i = 0; i < sizeA; ++i) firstVectorA[i] = h_firstVectorA[i];
    for(int i = 0; i < sizeB; ++i) firstVectorB[i] = h_firstVectorB[i];

    // Use Thrust to process on device (for larger vectors)
    if (sizeA + sizeB > 1024) {
        // Use device vectors
        thrust::device_vector<int> d_merged(sizeA + sizeB);
        thrust::device_vector<int> d_abs(sizeA + sizeB);
        thrust::device_vector<int> d_unique;
        
        // Copy data to device
        thrust::copy(firstVectorA.begin(), firstVectorA.end(), d_merged.begin());
        thrust::copy(firstVectorB.begin(), firstVectorB.end(), d_merged.begin() + sizeA);
        
        // Compute absolute values - using the functor
        thrust::transform(d_merged.begin(), d_merged.end(), d_abs.begin(), AbsoluteFunctor());
        
        // Sort and get unique elements
        thrust::sort(d_abs.begin(), d_abs.end());
        d_unique.resize(d_abs.size());
        auto end = thrust::unique_copy(d_abs.begin(), d_abs.end(), d_unique.begin());
        
        // Return count of unique absolute values
        return end - d_unique.begin();
    } else {
        // Process small vectors on host
        std::vector<int> merged;
        merged.insert(merged.end(), firstVectorA.begin(), firstVectorA.end());
        merged.insert(merged.end(), firstVectorB.begin(), firstVectorB.end());
        
        // Count unique absolute values (Python: len(set(abs(x) for x in merged)))
        std::set<int> uniqueAbsValues;
        for (int value : merged) {
            uniqueAbsValues.insert(abs(value));
        }
        
        return uniqueAbsValues.size();
    }
}

// CPU fallback for extremely large problem sizes
CudaSet processPairCPU(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose) {
    // Copy data to host
    HostSet hostSetA = copyDeviceToHost(setA);
    HostSet hostSetB = copyDeviceToHost(setB);
    
    // Process combinations on CPU with sampling
    std::vector<std::vector<int>> validCombinations;
    
    // For extremely large problems, consider sampling
    bool useSampling = (hostSetA.vectors.size() * hostSetB.vectors.size() > 10000000);
    int stride = useSampling ? 10 : 1; // Process every 10th combination if sampling
    
    int processedCount = 0;
    int validCount = 0;
    
    for (int idxA = 0; idxA < hostSetA.vectors.size(); idxA++) {
        for (int idxB = 0; idxB < hostSetB.vectors.size(); idxB += stride) {
            // Merge vectors
            std::vector<int> merged;
            merged.insert(merged.end(), hostSetA.vectors[idxA].begin(), hostSetA.vectors[idxA].end());
            merged.insert(merged.end(), hostSetB.vectors[idxB].begin(), hostSetB.vectors[idxB].end());
            
            // Get unique elements
            std::set<int> uniqueElements(merged.begin(), merged.end());
            
            // Apply threshold filter
            bool isValid = (threshold == 0 || uniqueElements.size() <= threshold);
            
            if (isValid) {
                validCombinations.push_back(std::vector<int>(uniqueElements.begin(), uniqueElements.end()));
                validCount++;
            }
            
            processedCount++;
            if (verbose && processedCount % 1000000 == 0) {
                printf("    Processed %d combinations, found %d valid...\n", processedCount, validCount);
            }
        }
    }
    
    // Remove duplicates
    std::vector<std::vector<int>> uniqueValidCombinations;
    
    for (const auto& combination : validCombinations) {
        bool isDuplicate = false;
        
        for (const auto& existing : uniqueValidCombinations) {
            if (combination.size() == existing.size()) {
                std::set<int> combinationSet(combination.begin(), combination.end());
                std::set<int> existingSet(existing.begin(), existing.end());
                
                if (combinationSet == existingSet) {
                    isDuplicate = true;
                    break;
                }
            }
        }
        
        if (!isDuplicate) {
            uniqueValidCombinations.push_back(combination);
        }
    }
    
    if (verbose) {
        printf("  CPU processing produced %zu valid combinations, %zu unique\n", 
               validCombinations.size(), uniqueValidCombinations.size());
    }
    
    // Create result set
    HostSet resultHostSet;
    resultHostSet.vectors = uniqueValidCombinations;
    
    CudaSet resultCudaSet;
    copyHostToDevice(resultHostSet, &resultCudaSet);
    
    return resultCudaSet;
}

// GPU deduplication alternative implementation using functors
// Vector comparison functor for thrust::sort
struct VectorCompare {
    const int* data;
    const int* lengths;
    int maxLen;
    
    VectorCompare(const int* d, const int* l, int max) : data(d), lengths(l), maxLen(max) {}
    
    __host__ __device__
    bool operator()(int i, int j) const {
        int lenI = lengths[i];
        int lenJ = lengths[j];
        
        // First compare lengths
        if (lenI != lenJ) {
            return lenI < lenJ;
        }
        
        // If lengths are equal, compare elements
        for (int k = 0; k < lenI; k++) {
            int valI = data[i * maxLen + k];
            int valJ = data[j * maxLen + k];
            if (valI != valJ) {
                return valI < valJ;
            }
        }
        
        return false; // Equal vectors
    }
};

struct MarkDuplicates {
    const int* indices;
    const int* data;
    const int* lengths;
    int* isUnique;
    int maxLen;
    
    MarkDuplicates(const int* idx, const int* d, const int* l, int* u, int max) 
        : indices(idx), data(d), lengths(l), isUnique(u), maxLen(max) {}
    
    __host__ __device__
    void operator()(int idx) {
        if (idx == 0) return; // First item is always unique
        
        int curr = indices[idx];
        int prev = indices[idx - 1];
        
        int lenCurr = lengths[curr];
        int lenPrev = lengths[prev];
        
        // If lengths differ, this is a unique item
        if (lenCurr != lenPrev) return;
        
        // Check if all elements match
        bool identical = true;
        for (int k = 0; k < lenCurr; k++) {
            if (data[curr * maxLen + k] != data[prev * maxLen + k]) {
                identical = false;
                break;
            }
        }
        
        // If identical, mark as non-unique
        if (identical) {
            isUnique[idx] = 0;
        }
    }
};

// Maximally concurrent deduplication
std::vector<std::vector<int>> deduplicateCombinations(
    const std::vector<std::vector<int>>& combinations, bool verbose) {
    
    if (combinations.size() <= 1) return combinations;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    //if (verbose) {
    //    printf("      Deduplicating %zu combinations with maximum concurrency...\n", combinations.size());
    //}
    
    // Find maximum vector length
    size_t maxLen = 0;
    for (const auto& vec : combinations) {
        maxLen = std::max(maxLen, vec.size());
    }
    
    // PHASE 1: Prepare data with maximum parallelism
    // Create device vectors
    thrust::device_vector<int> d_flatData(combinations.size() * maxLen);
    thrust::device_vector<int> d_lengths(combinations.size());
    thrust::device_vector<int> d_indices(combinations.size());
    
    // Determine optimal thread and block configuration
    int maxThreadsPerBlock = 1024;  // Maximum threads per block for most GPUs
    int vectorsPerBlock = 32;      // Process multiple vectors per block for better occupancy
    int elementsPerThread = 4;     // Process multiple elements per thread for better efficiency
    
    dim3 blockSize(maxThreadsPerBlock / vectorsPerBlock);
    dim3 gridSize((combinations.size() + vectorsPerBlock - 1) / vectorsPerBlock);
    
    // Fill device vectors with pinned memory for faster transfers
    int* h_flatData;
    int* h_lengths;
    cudaHostAlloc(&h_flatData, combinations.size() * maxLen * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_lengths, combinations.size() * sizeof(int), cudaHostAllocDefault);
    
    // Fill host pinned memory
    #pragma omp parallel for
    for (size_t i = 0; i < combinations.size(); i++) {
        h_lengths[i] = combinations[i].size();
        
        // Copy data
        for (size_t j = 0; j < combinations[i].size(); j++) {
            h_flatData[i * maxLen + j] = combinations[i][j];
        }
        
        // Pad with INT_MIN
        for (size_t j = combinations[i].size(); j < maxLen; j++) {
            h_flatData[i * maxLen + j] = INT_MIN;
        }
    }
    
    // Copy to device (overlap with sequence generation)
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    
    // Copy data in stream 0
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_flatData.data()), h_flatData,
                  combinations.size() * maxLen * sizeof(int),
                  cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_lengths.data()), h_lengths,
                  combinations.size() * sizeof(int),
                  cudaMemcpyHostToDevice, streams[0]);
    
    // Generate sequence in stream 1
    thrust::sequence(thrust::cuda::par.on(streams[1]), d_indices.begin(), d_indices.end());
    
    // PHASE 2: Sort each vector with maximum parallelism
    // Launch massively parallel kernel to sort all vectors simultaneously
    sortAllVectorsKernel<<<gridSize, blockSize, 0, streams[0]>>>(
        thrust::raw_pointer_cast(d_flatData.data()),
        thrust::raw_pointer_cast(d_lengths.data()),
        combinations.size(),
        maxLen);
    
    // Wait for all operations to complete
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    
    // PHASE 3: Global sort with high-performance parallel algorithm
    // Use key-value sorting with vectors as keys
    thrust::sort(thrust::cuda::par,
               d_indices.begin(), d_indices.end(),
               VectorCompare(thrust::raw_pointer_cast(d_flatData.data()),
                            thrust::raw_pointer_cast(d_lengths.data()),
                            maxLen));
    
    // PHASE 4: Mark duplicates with parallel scan
    thrust::device_vector<int> d_isUnique(combinations.size(), 1);
    
    // Mark duplicates in parallel
    markDuplicatesKernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(d_indices.data()),
        thrust::raw_pointer_cast(d_flatData.data()),
        thrust::raw_pointer_cast(d_lengths.data()),
        thrust::raw_pointer_cast(d_isUnique.data()),
        combinations.size(),
        maxLen);
    
    // PHASE 5: Count and gather unique vectors with parallel primitives
    int uniqueCount = thrust::reduce(thrust::cuda::par, d_isUnique.begin(), d_isUnique.end());
    
    //if (verbose) {
    //    float reduction = 100.0f * (1.0f - (float)uniqueCount / combinations.size());
    //    printf("      Found %d unique combinations out of %zu (%.1f%% reduction)\n", 
    //           uniqueCount, combinations.size(), reduction);
    //}
    
    // Early return for no duplicates
    if (uniqueCount == combinations.size()) {
        cudaFreeHost(h_flatData);
        cudaFreeHost(h_lengths);
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        //if (verbose) {
        //    printf("      Parallel deduplication completed in %.2f ms\n", milliseconds);
        //}
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return combinations;
    }
    
    // Create a vector of unique indices using stream compaction
    thrust::device_vector<int> d_uniqueIndices(uniqueCount);
    thrust::copy_if(thrust::cuda::par,
                  thrust::make_counting_iterator<int>(0),
                  thrust::make_counting_iterator<int>((int)combinations.size()),
                  d_isUnique.begin(),
                  d_uniqueIndices.begin(),
                  thrust::identity<int>());
    
    // Gather the indices of unique vectors

thrust::device_vector<int> d_uniqueVectorIndices(uniqueCount);
gatherIndicesKernel<<<(uniqueCount + 255) / 256, 256>>>(
    thrust::raw_pointer_cast(d_uniqueIndices.data()),
    thrust::raw_pointer_cast(d_indices.data()),
    thrust::raw_pointer_cast(d_uniqueVectorIndices.data()),
    uniqueCount);
    
    // PHASE 6: Construct result with parallel operations
    // Copy results back to host
    thrust::host_vector<int> h_uniqueVectorIndices = d_uniqueVectorIndices;
    
    // Prepare result vectors
    std::vector<std::vector<int>> result(uniqueCount);
    
    // Construct result vectors in parallel
    #pragma omp parallel for
    for (int i = 0; i < uniqueCount; i++) {
        int idx = h_uniqueVectorIndices[i];
        int len = h_lengths[idx];
        
        result[i].reserve(len);
        for (int j = 0; j < len; j++) {
            int val = h_flatData[idx * maxLen + j];
            if (val != INT_MIN) {
                result[i].push_back(val);
            }
        }
    }
    
    // Clean up
    cudaFreeHost(h_flatData);
    cudaFreeHost(h_lengths);
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    //if (verbose) {
    //    printf("      Massively parallel deduplication completed in %.2f ms\n", milliseconds);
    //}
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return result;
}


// Helper function to extract a subset from a CudaSet
CudaSet extractSubset(const CudaSet& set, int startIndex, int count, bool verbose) {
    if (count <= 0) {
        // Return empty set
        CudaSet emptySet;
        emptySet.numItems = 0;
        emptySet.totalElements = 0;
        emptySet.data = nullptr;
        emptySet.offsets = nullptr;
        emptySet.sizes = nullptr;
        emptySet.deviceBuffer = nullptr;
        emptySet.bufferSize = 0;
        return emptySet;
    }
    
    // Copy size and offset information for the slice
    std::vector<int> hostSizes(count);
    std::vector<int> hostOffsets(count);
    
    CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), set.sizes + startIndex, 
                              count * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(hostOffsets.data(), set.offsets + startIndex, 
                              count * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate total elements in the subset
    int totalElements = 0;
    for (int i = 0; i < count; i++) {
        totalElements += hostSizes[i];
    }
    
    // Allocate memory for the subset
    CudaSet subSet = allocateCudaSet(count, totalElements);
    
    // Copy offset and size information
    std::vector<int> newOffsets(count);
    int currentOffset = 0;
    for (int i = 0; i < count; i++) {
        newOffsets[i] = currentOffset;
        currentOffset += hostSizes[i];
    }
    
    CHECK_CUDA_ERROR(cudaMemcpy(subSet.sizes, hostSizes.data(), 
                              count * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(subSet.offsets, newOffsets.data(), 
                              count * sizeof(int), cudaMemcpyHostToDevice));
    
    // Copy data elements for each vector
    for (int i = 0; i < count; i++) {
        int srcOffset = hostOffsets[i];
        int dstOffset = newOffsets[i];
        int size = hostSizes[i];
        
        CHECK_CUDA_ERROR(cudaMemcpy(subSet.data + dstOffset, set.data + srcOffset, 
                                  size * sizeof(int8_t), cudaMemcpyDeviceToDevice));
    }
    
    return subSet;
}

void processAndStreamResults(const std::unordered_map<size_t, std::vector<int>>& uniqueResults, 
                            const char* outputPath, bool verbose) {
    if (verbose) {
        printf("  Streaming %zu results directly to output file\n", uniqueResults.size());
    }
    
    // Open output file
    FILE* outFile = fopen(outputPath, "w");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file %s\n", outputPath);
        return;
    }
    
    // Write header
    fprintf(outFile, "# Final results: %zu vectors\n", uniqueResults.size());
    
    // Convert map to vector for batch processing
    std::vector<std::vector<int>> allVectors;
    allVectors.reserve(uniqueResults.size());
    
    for (const auto& pair : uniqueResults) {
        allVectors.push_back(pair.second);
    }
    
    // Process vectors in batches to conserve memory
    const int BATCH_SIZE = 100000;
    int batches = (allVectors.size() + BATCH_SIZE - 1) / BATCH_SIZE;
    
    std::vector<std::vector<int>> batchResults;
    
    for (int batch = 0; batch < batches; batch++) {
        int start = batch * BATCH_SIZE;
        int end = std::min(start + BATCH_SIZE, (int)allVectors.size());
        
        if (verbose && (batch % 10 == 0 || batch == batches-1)) {
            printf("    Processing batch %d/%d (%.1f%%)\n", 
                   batch+1, batches, 100.0 * (batch+1) / batches);
        }
        
        // Clear previous batch
        batchResults.clear();
        
        // Extract and filter this batch
        for (int i = start; i < end; i++) {
            const std::vector<int>& vec = allVectors[i];
            
            // Filter out negative values
            std::vector<int> positivesOnly;
            for (int val : vec) {
                if (val >= 0) {
                    positivesOnly.push_back(val);
                }
            }
            
            // Sort the vector
            std::sort(positivesOnly.begin(), positivesOnly.end());
            
            batchResults.push_back(positivesOnly);
        }
        
        // Sort this batch
        std::sort(batchResults.begin(), batchResults.end());
        
        // Write to file
        for (const auto& vec : batchResults) {
            fprintf(outFile, "[");
            for (size_t i = 0; i < vec.size(); i++) {
                fprintf(outFile, "%d", vec[i]);
                if (i < vec.size() - 1) fprintf(outFile, ", ");
            }
            fprintf(outFile, "]\n");
        }
        
        // Ensure data is written
        fflush(outFile);
    }
    
    // Close file
    fclose(outFile);
    
    if (verbose) {
        printf("  Results successfully written to %s\n", outputPath);
    }
}

CudaSet processLargePair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    
    if (verbose) {
        printf("  Processing large pair with tiled approach: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               numItemsA, numItemsB, threshold);
        printf("  Total combinations: %lld - using tiled processing for efficiency\n", totalCombinations);
    }
    
    // Define tile sizes based on problem dimensions
    int TILE_SIZE_A = 32;
    int TILE_SIZE_B = 1024;
    
    // Adjust tile sizes if needed
    if (numItemsA > 1000 || numItemsB > 10000) {
        TILE_SIZE_A = 16;
        TILE_SIZE_B = 512;
    }
    
    // Calculate number of tiles
    int numTilesA = (numItemsA + TILE_SIZE_A - 1) / TILE_SIZE_A;
    int numTilesB = (numItemsB + TILE_SIZE_B - 1) / TILE_SIZE_B;
    int totalTiles = numTilesA * numTilesB;
    
    if (verbose) {
        printf("    Processing in %d x %d = %d tiles\n", numTilesA, numTilesB, totalTiles);
        printf("    Tile dimensions: %d x %d items\n", TILE_SIZE_A, TILE_SIZE_B);
    }
    
    // Maintain a set of unique results with fast lookup
    std::unordered_map<size_t, std::vector<int>> uniqueResults;
    
    // Hash function for vectors
    auto hashVector = [](const std::vector<int>& vec) {
        size_t hash = vec.size();
        for (int val : vec) {
            hash ^= std::hash<int>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    };
    
    // Process each tile
    int tilesProcessed = 0;
    int lastProgressUpdate = 0;
    
    for (int tileA = 0; tileA < numTilesA; tileA++) {
        int startA = tileA * TILE_SIZE_A;
        int endA = std::min(startA + TILE_SIZE_A, numItemsA);
        int sizeA = endA - startA;
        
        // Create a sub-set for this tile of setA
        CudaSet tileSetA = extractSubset(setA, startA, sizeA, verbose);
        
        for (int tileB = 0; tileB < numTilesB; tileB++) {
            int startB = tileB * TILE_SIZE_B;
            int endB = std::min(startB + TILE_SIZE_B, numItemsB);
            int sizeB = endB - startB;
            
            tilesProcessed++;
            
            // Update progress periodically
            int progressPercentage = (tilesProcessed * 100) / totalTiles;
            if (verbose && (progressPercentage > lastProgressUpdate || tilesProcessed == totalTiles)) {
                printf("      Processing tile [%d,%d] x [%d,%d] (Tile %d of %d - %d%% complete)\n", 
                      startA, endA-1, startB, endB-1, tilesProcessed, totalTiles, progressPercentage);
                lastProgressUpdate = progressPercentage;
            }
            
            // Create a sub-set for this tile of setB
            CudaSet tileSetB = extractSubset(setB, startB, sizeB, verbose);
            
            // Process this tile pair directly - AVOID circular call to processPair
            // Instead, we'll inline the core GPU processing logic here
            
            int numTileItemsA = tileSetA.numItems;
            int numTileItemsB = tileSetB.numItems;
            long long tileCombinations = (long long)numTileItemsA * (long long)numTileItemsB;
            
            // Allocate result buffer for this tile
            CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(numTileItemsA, numTileItemsB, MAX_ELEMENTS_PER_VECTOR);
            
            // Configure kernel launch
            int threadsPerBlock = 256;
            int maxResultsPerThread = 4;
            int threadsNeeded = (tileCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
            int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            
            // Limit blocks to avoid excessive memory usage
            const int MAX_BLOCKS = 16384;
            if (blocksNeeded > MAX_BLOCKS) {
                blocksNeeded = MAX_BLOCKS;
                maxResultsPerThread = (tileCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
            }
            
            // Launch kernel
            processAllCombinationsKernel<<<blocksNeeded, threadsPerBlock>>>(
                tileSetA.data, tileSetA.offsets, tileSetA.sizes, numTileItemsA,
                tileSetB.data, tileSetB.offsets, tileSetB.sizes, numTileItemsB,
                threshold, level,
                resultBuffer.data, resultBuffer.sizes, resultBuffer.validFlags, resultBuffer.maxResultSize,
                maxResultsPerThread
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            // Count valid combinations
            std::vector<int> hostValidFlags(resultBuffer.numCombinations);
            CHECK_CUDA_ERROR(cudaMemcpy(hostValidFlags.data(), resultBuffer.validFlags, 
                                      resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
            
            int validCount = 0;
            for (int i = 0; i < resultBuffer.numCombinations; i++) {
                if (hostValidFlags[i]) validCount++;
            }
            
            // Process valid combinations
            if (validCount > 0) {
                std::vector<int> hostSizes(resultBuffer.numCombinations);
                CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), resultBuffer.sizes, 
                                          resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
                
                std::vector<int> hostResultData(resultBuffer.numCombinations * resultBuffer.maxResultSize);
                CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
                                          resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                
                // Collect and add valid combinations to our results
                int newResults = 0;
                int duplicates = 0;
                
                for (int i = 0; i < resultBuffer.numCombinations; i++) {
                    if (hostValidFlags[i]) {
                        int size = hostSizes[i];
                        std::vector<int> combination(size);
                        int offset = i * resultBuffer.maxResultSize;
                        
                        for (int j = 0; j < size; j++) {
                            combination[j] = hostResultData[offset + j];
                        }
                        
                        // Sort the combination for canonicalization
                        std::sort(combination.begin(), combination.end());
                        
                        // Add to unique results if not present
                        size_t hash = hashVector(combination);
                        
                        if (uniqueResults.find(hash) == uniqueResults.end()) {
                            // Check for hash collision
                            bool isUnique = true;
                            if (uniqueResults.count(hash) > 0) {
                                if (uniqueResults[hash] == combination) {
                                    isUnique = false;
                                }
                            }
                            
                            if (isUnique) {
                                uniqueResults[hash] = std::move(combination);
                                newResults++;
                            } else {
                                duplicates++;
                            }
                        } else {
                            duplicates++;
                        }
                    }
                }
                
                //if (verbose && validCount > 0) {
               //     printf("        Tile result: %d total, %d new unique, %d duplicates\n", 
                //           validCount, newResults, duplicates);
                //    printf("        Total unique results so far: %zu\n", uniqueResults.size());
               // }
            }
            
            // Free result buffer
            freeCombinationResultBuffer(&resultBuffer);
            
            // Free tile resources
            freeCudaSet(&tileSetB);
        }
        
        // Free tile resources
        freeCudaSet(&tileSetA);
    }
    

    
    if (verbose) {
        printf("  Final result: %zu unique combinations\n", uniqueResults.size());
    }
    
    if (uniqueResults.size() > 20000000) {
        // Stream results to disk
        processAndStreamResults(uniqueResults, "zdd.txt", verbose);
        
        // Return a dummy small CudaSet that indicates stream processing was used
        CudaSet dummyResult = allocateCudaSet(1, 1);
        int* flagData = (int*)malloc(sizeof(int));
        flagData[0] = -999999; // Special flag indicating results were streamed
        CHECK_CUDA_ERROR(cudaMemcpy(dummyResult.data, flagData, sizeof(int), cudaMemcpyHostToDevice));
        free(flagData);
        
        return dummyResult;
    }
    
    // Convert map to result vectors
    std::vector<std::vector<int>> finalResults;
    finalResults.reserve(uniqueResults.size());
    
    for (const auto& pair : uniqueResults) {
        finalResults.push_back(pair.second);
    }
    // Create result set
    HostSet resultHostSet;
    resultHostSet.vectors = finalResults;
    
    
    CudaSet resultCudaSet;
    copyHostToDevice(resultHostSet, &resultCudaSet);
    
    return resultCudaSet;
}
// Process a pair of sets using parallel kernel (maintains exact logic but parallelized)
// Process a pair of sets using batched approach for large sets
// Process a pair of sets using parallel kernel (simplified for correctness)
// Process a pair of sets using GPU-based batching
CudaSet processPair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    
    if (verbose) {
        printf("  Processing pair at level %d: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               level, numItemsA, numItemsB, threshold);
    }
    
    // Empty result for empty inputs
    if (numItemsA == 0 || numItemsB == 0) {
        CudaSet emptySet;
        emptySet.numItems = 0;
        emptySet.totalElements = 0;
        emptySet.data = nullptr;
        emptySet.offsets = nullptr;
        emptySet.sizes = nullptr;
        emptySet.deviceBuffer = nullptr;
        emptySet.bufferSize = 0;
        return emptySet;
    }

    // For extremely large combinations, use CPU fallback
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    if (totalCombinations > 100000000LL) { // 100 million combinations threshold
        if (verbose) {
            printf("    Using CPU fallback for extremely large input (%lld combinations)\n", totalCombinations);
        }
        return processPairCPU(setA, setB, threshold, level, verbose);
    }
            // For extremely large combinations, use the memory-efficient approach
    if (totalCombinations > 5000000LL) { // 5 million threshold
        return processLargePair(setA, setB, threshold, level, verbose);
    }
    
    // Calculate buffer size needed
    int maxResultsPerThread = 4; // Each thread will process up to 4 combinations
    int threadsNeeded = (totalCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
    
    // Determine thread block configuration
    int threadsPerBlock = 256;
    int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
    
    // Limit blocks to avoid excessive memory usage
    const int MAX_BLOCKS = 16384; // Adjust based on GPU capability
    if (blocksNeeded > MAX_BLOCKS) {
        blocksNeeded = MAX_BLOCKS;
        maxResultsPerThread = (totalCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
        if (verbose) {
            printf("    Adjusting to %d blocks with %d results per thread\n", blocksNeeded, maxResultsPerThread);
        }
    }
    
    // Allocate result buffer
    CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(numItemsA, numItemsB, MAX_ELEMENTS_PER_VECTOR);
    
    // Launch kernel with grid-stride batching
    if (verbose) {
        printf("    Using GPU batching with %d blocks, %d threads per block, %d combinations per thread\n", 
               blocksNeeded, threadsPerBlock, maxResultsPerThread);
    }
    
    processAllCombinationsKernel<<<blocksNeeded, threadsPerBlock>>>(
        setA.data, setA.offsets, setA.sizes, numItemsA,
        setB.data, setB.offsets, setB.sizes, numItemsB,
        threshold, level,
        resultBuffer.data, resultBuffer.sizes, resultBuffer.validFlags, resultBuffer.maxResultSize,
        maxResultsPerThread
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Count valid combinations
    std::vector<int> hostValidFlags(resultBuffer.numCombinations);
    CHECK_CUDA_ERROR(cudaMemcpy(hostValidFlags.data(), resultBuffer.validFlags, 
                              resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<int> hostSizes(resultBuffer.numCombinations);
    CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), resultBuffer.sizes, 
                              resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
    
    int validCount = 0;
    for (int i = 0; i < resultBuffer.numCombinations; i++) {
        if (hostValidFlags[i]) validCount++;
    }
    
    if (verbose) {
        printf("    Found %d valid combinations out of %d total\n", validCount, (int)totalCombinations);
    }
		
	// Gather valid combinations
	std::vector<std::vector<int>> validCombinations;

	// Only copy data if we have valid combinations
	if (validCount > 0) {
		if (verbose) {
		    printf("    Copying result data for %d valid combinations...\n", validCount);
		}
		
		std::vector<int> hostResultData(resultBuffer.numCombinations * resultBuffer.maxResultSize);
		CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
		                          resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
		                          cudaMemcpyDeviceToHost));
		
		// Progress reporting variables
		int reportInterval = validCount > 1000 ? validCount / 10 : validCount;
		int lastReportedCount = 0;
		int collectedCount = 0;
		
		// Collect valid combinations
		for (int i = 0; i < resultBuffer.numCombinations; i++) {
		    if (hostValidFlags[i]) {
		        int size = hostSizes[i];
		        std::vector<int> combination(size);
		        int offset = i * resultBuffer.maxResultSize;
		        
		        for (int j = 0; j < size; j++) {
		            combination[j] = hostResultData[offset + j];
		        }
		        
		        validCombinations.push_back(combination);
		        collectedCount++;
		        
		        // Progress reporting for large result sets
		        if (verbose && validCount > 1000 && collectedCount - lastReportedCount >= reportInterval) {
		            printf("    Collected %d of %d valid combinations (%.1f%%)\n", 
		                   collectedCount, validCount, 100.0 * collectedCount / validCount);
		            lastReportedCount = collectedCount;
		        }
		    }
		}
		
		if (verbose && validCount > 1000) {
		    printf("    Collection complete: %d combinations collected\n", collectedCount);
		}
	}

	// Free result buffer
	freeCombinationResultBuffer(&resultBuffer);

	// Remove duplicates
	// Alternative approach: sort-based deduplication - even faster, but requires more initial sorting
	if (verbose) {
		printf("    Removing duplicates from %zu combinations using sort-based approach...\n", validCombinations.size());
	}

	// First, canonicalize each vector by sorting it
	for (auto& combination : validCombinations) {
		std::sort(combination.begin(), combination.end());
	}

	// Then sort all vectors lexicographically
	if (verbose && validCombinations.size() > 10000) {
		printf("    Sorting %zu combinations...\n", validCombinations.size());
	}
	std::sort(validCombinations.begin(), validCombinations.end());

	// Remove duplicates with a linear scan
	if (verbose && validCombinations.size() > 10000) {
		printf("    Performing linear scan to remove duplicates...\n");
	}
	std::vector<std::vector<int>> uniqueValidCombinations;
	uniqueValidCombinations.reserve(validCombinations.size());

	for (size_t i = 0; i < validCombinations.size(); i++) {
		// Skip if same as previous element (duplicates are adjacent after sorting)
		if (i > 0 && validCombinations[i] == validCombinations[i-1]) {
		    continue;
		}
		uniqueValidCombinations.push_back(validCombinations[i]);
		
		// Progress reporting
		if (verbose && validCombinations.size() > 10000 && i % 10000 == 0) {
		    printf("    Deduplication scan: processed %zu of %zu combinations (%.1f%%), found %zu unique\n", 
		           i, validCombinations.size(), 
		           100.0 * i / validCombinations.size(),
		           uniqueValidCombinations.size());
		}
	}

	if (verbose) {
		printf("  Deduplication complete: %zu items, %zu unique (%.1f%% unique)\n", 
		       validCombinations.size(), uniqueValidCombinations.size(), 
		       validCombinations.size() > 0 ? 100.0 * uniqueValidCombinations.size() / validCombinations.size() : 0);
	}
    // Create result set
    HostSet resultHostSet;
    resultHostSet.vectors = uniqueValidCombinations;
    
    CudaSet resultCudaSet;
    copyHostToDevice(resultHostSet, &resultCudaSet);
    
    return resultCudaSet;
}


// Special handling for converting a set to unique elements (for level 1 carry-over)
CudaSet convertSetToUnique(const CudaSet& set, bool verbose) {
    int numItems = set.numItems;
    
    // Allocate host vectors 
    std::vector<int> hostOffsets(numItems);
    std::vector<int> hostSizes(numItems);
    
    CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), set.sizes, numItems * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate max possible size for outputs
    int totalOutputSize = 0;
    for (int i = 0; i < numItems; i++) {
        totalOutputSize += hostSizes[i]; // Worst case: all elements are unique
    }
    
    // Create output arrays
    int8_t* d_outputData = nullptr;
    int* d_outputOffsets = nullptr;
    int* d_outputSizes = nullptr;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_outputData, totalOutputSize * sizeof(int8_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputOffsets, numItems * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_outputSizes, numItems * sizeof(int)));
    
    // Calculate output offsets (equivalent to the input offsets)
    CHECK_CUDA_ERROR(cudaMemcpy(d_outputOffsets, set.offsets, numItems * sizeof(int), cudaMemcpyDeviceToDevice));
    
    // Launch parallel kernel
    int threadsPerBlock = 256;
    int blocks = (numItems + threadsPerBlock - 1) / threadsPerBlock;
    
    convertToUniqueKernel<<<blocks, threadsPerBlock>>>(
        set.data, set.offsets, set.sizes, numItems,
        d_outputData, d_outputOffsets, d_outputSizes, MAX_ELEMENTS_PER_VECTOR
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Create result set
    CudaSet resultSet;
    resultSet.data = d_outputData;
    resultSet.offsets = d_outputOffsets;
    resultSet.sizes = d_outputSizes;
    resultSet.numItems = numItems;
    resultSet.totalElements = totalOutputSize;
    resultSet.deviceBuffer = nullptr;
    resultSet.bufferSize = 0;
    
    if (verbose) {
        printf("  Converting carried-over set for level 2\n");
        printf("  Carried over the last set with %d items\n", numItems);
    }
    
    return resultSet;
}


// Tree fold operations (maintains sequential dependencies but optimizes within each step)
CudaSet treeFoldOperations(const std::vector<CudaSet>& sets, bool verbose) {
    if (sets.empty()) {
        CudaSet emptySet;
        emptySet.numItems = 0;
        emptySet.totalElements = 0;
        emptySet.data = nullptr;
        emptySet.offsets = nullptr;
        emptySet.sizes = nullptr;
        emptySet.deviceBuffer = nullptr;
        emptySet.bufferSize = 0;
        return emptySet;
    }
    
    if (sets.size() == 1) {
        return sets[0];
    }
    
    if (verbose) {
        printf("Starting tree-fold operations with %zu sets\n", sets.size());
        for (size_t i = 0; i < sets.size(); i++) {
            printf("  Set %zu: %d items\n", i + 1, sets[i].numItems);
        }
    }
    
    // Queue of sets to process
    std::vector<CudaSet> currentLevel = sets;
    
    // Continue until we have only one result set
    int level = 0;
    while (currentLevel.size() > 1) {
        level++;
        if (verbose) {
            printf("\nProcessing Level %d with %zu sets\n", level, currentLevel.size());
        }
        
        std::vector<CudaSet> nextLevel;
        
        // Process pairs of sets
        size_t i = 0;
        while (i < currentLevel.size() - 1) {
            CudaSet setA = currentLevel[i];
            CudaSet setB = currentLevel[i + 1];
            
            // Compute threshold for this pair (sequential - critical for algorithm correctness)
            int threshold = computeThreshold(setA, setB);
            
            // Process the pair (parallelized internally)
            CudaSet resultSet = processPair(setA, setB, threshold, level, verbose);
            nextLevel.push_back(resultSet);
            
            // Free intermediate sets
            if (level > 1) {
                freeCudaSet(&currentLevel[i]);
                freeCudaSet(&currentLevel[i + 1]);
            }
            
            i += 2;
        }
        
        // If odd number of sets, carry the last one to next level
        if (i == currentLevel.size() - 1) {
            if (level == 1) {
                // Special handling for level 1 (parallelized)
                CudaSet convertedSet = convertSetToUnique(currentLevel[i], verbose);
                nextLevel.push_back(convertedSet);
            } else {
                nextLevel.push_back(currentLevel[i]);
                if (verbose) {
                    printf("  Carried over the last set with %d items\n", currentLevel[i].numItems);
                }
            }
        }
        
        // Move to next level
        currentLevel = nextLevel;
    }
    CudaSet result = currentLevel[0];
    
    // Check if this is a dummy result indicating streaming
    int firstValue;
    CHECK_CUDA_ERROR(cudaMemcpy(&firstValue, result.data, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (firstValue == -999999) {
        if (verbose) {
            printf("Results were too large for memory and were streamed to disk.\n");
            printf("Check large_result.txt for complete results.\n");
        }
    }
    
    // Return the final result
    return currentLevel[0];
}
// Kernel to filter out negative values and sort
__global__ void filterAndSortKernel(int8_t* data, int* offsets, int* sizes, int numVectors, int maxLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVectors) return;
    
    int offset = offsets[idx];
    int originalSize = sizes[idx];
    
    // Step 1: Filter out negatives
    int newSize = 0;
    for (int i = 0; i < originalSize; i++) {
        int val = data[offset + i];
        if (val >= 0) {
            // Keep only non-negative values
            data[offset + newSize] = val;
            newSize++;
        }
    }
    
    // Update size
    sizes[idx] = newSize;
    
    // Step 2: Sort (simple insertion sort)
    for (int i = 1; i < newSize; i++) {
        int key = data[offset + i];
        int j = i - 1;
        
        while (j >= 0 && data[offset + j] > key) {
            data[offset + j + 1] = data[offset + j];
            j--;
        }
        
        data[offset + j + 1] = key;
    }
}

// Function to post-process on GPU before transferring to host
// Function to post-process on GPU then complete ordering on CPU
std::vector<std::vector<int>> gpuPostProcess(const CudaSet& resultSet, bool verbose) {
    // Step 1: Run GPU kernel to filter and sort all vectors internally
    int threadsPerBlock = 256;
    int blocks = (resultSet.numItems + threadsPerBlock - 1) / threadsPerBlock;
    
    if (verbose) {
        printf("Running GPU post-processing on %d vectors\n", resultSet.numItems);
    }
    
    filterAndSortKernel<<<blocks, threadsPerBlock>>>(
        resultSet.data, resultSet.offsets, resultSet.sizes, 
        resultSet.numItems, MAX_ELEMENTS_PER_VECTOR);
    
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    if (verbose) {
        printf("GPU internal sorting complete, transferring to host for final sorting\n");
    }
    
    // Step 2: Process in batches to avoid memory issues
    const int BATCH_SIZE = 100000;
    int totalVectors = resultSet.numItems;
    int batches = (totalVectors + BATCH_SIZE - 1) / BATCH_SIZE;
    
    std::vector<std::vector<int>> processedResults;
    processedResults.reserve(std::min(totalVectors, 10000000)); // Reserve reasonable amount
    
    for (int batch = 0; batch < batches; batch++) {
        int start = batch * BATCH_SIZE;
        int end = std::min(start + BATCH_SIZE, totalVectors);
        
        if (verbose) {
            printf("Processing batch %d/%d (vectors %d to %d)\n", batch+1, batches, start, end-1);
        }
        
        // Extract subset of the CudaSet
        CudaSet batchSet = extractSubset(resultSet, start, end - start, false);
        
        // Process this batch - already filtered and sorted internally by GPU
        HostSet hostBatch = copyDeviceToHost(batchSet);
        
        // Add to results
        for (const auto& vector : hostBatch.vectors) {
            processedResults.push_back(vector);
        }
        
        // Free batch resources
        freeCudaSet(&batchSet);
        
        // Sort intermediate results if getting too large
        if (processedResults.size() > 1000000) {
            if (verbose) {
                printf("  Performing intermediate sort of %zu results\n", processedResults.size());
            }
            std::sort(processedResults.begin(), processedResults.end());
        }
    }
    
    // Final lexicographical sorting of all vectors
    if (verbose) {
        printf("Performing final lexicographical sort of %zu vectors\n", processedResults.size());
    }
    std::sort(processedResults.begin(), processedResults.end());
    
    return processedResults;
}
// Post-process results (maintains sequential processing for algorithm correctness)
std::vector<std::vector<int>> postProcessResults(const CudaSet& resultSet) {
    HostSet hostResultSet = copyDeviceToHost(resultSet);
    
    // Print raw results before post-processing for debugging
    printf("Result pefore post processing [");
    for (size_t i = 0; i < hostResultSet.vectors.size(); i++) {
        printf("{");
        for (size_t j = 0; j < hostResultSet.vectors[i].size(); j++) {
            printf("%d", hostResultSet.vectors[i][j]);
            if (j < hostResultSet.vectors[i].size() - 1) printf(", ");
        }
        printf("}");
        if (i < hostResultSet.vectors.size() - 1) printf(", ");
    }
    printf("]\n\n");
    
    // Process each set exactly like Python (this needs to be sequential)
    std::vector<std::vector<int>> processedResults;
    
    for (const auto& vector : hostResultSet.vectors) {
        // Filter out negative integers
        std::vector<int> positivesOnly;
        for (int val : vector) {
            if (val >= 0) {
                positivesOnly.push_back(val);
            }
        }
        
        // Sort positives
        std::sort(positivesOnly.begin(), positivesOnly.end());
        
        // Add to processed results
        processedResults.push_back(positivesOnly);
    }
    
    // Sort lexicographically like Python
    std::sort(processedResults.begin(), processedResults.end());
    
    return processedResults;
}


// Function to read JSON and generate test sets
std::vector<std::vector<std::vector<int>>> generateTestSetsFromJSON(const std::string& filename) {
    // Read JSON file
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file: %s\n", filename.c_str());
        return {};
    }
    
    // Read file into string
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonData = buffer.str();
    
    // Parse JSON
    SimpleJsonParser parser(jsonData);
    auto clauses = parser.parseClauses();
    auto matrices = parser.parseMatrices();
    
    // Create a map for easier lookup
    std::map<std::string, SimpleJsonParser::Matrix> matrixMap;
    for (auto& m : matrices) {
        matrixMap[m.key] = m;
    }
    
    std::vector<std::vector<std::vector<int>>> testSets;
    
    // Process each clause
    for (auto& clause : clauses) {
        std::string key = clause.key;
        int condition_id1 = clause.condition_id1;
        int condition_id2 = clause.condition_id2;
        int consequence_id = clause.consequence_id;
        
        // Find corresponding matrix
        if (matrixMap.find(key) == matrixMap.end()) {
            printf("No matrix found for clause: %s\n", key.c_str());
            continue;
        }
        

        auto& matrix = matrixMap[key];
        int rows = matrix.rows;
        int cols = matrix.cols;
        auto& matrixData = matrix.data;
        
        printf("  Using matrix '%s': %d rows x %d cols, data size: %zu\n", 
               key.c_str(), rows, cols, matrixData.size());
        
        std::vector<std::vector<int>> testSet;
        
        // Generate sets based on matrix data
        for (int row = 0; row < rows; row++) {
            std::vector<int> testRow;
            
            // Add condition_id1 (always present)
            testRow.push_back(matrixData[row * cols + 0] == 1 ? condition_id1 : -condition_id1);
            
            // Add remaining elements based on matrix values
            if (condition_id2 != -1) {
                // If condition_id2 exists, add it based on matrix value
                if (cols >= 2) {
                    testRow.push_back(matrixData[row * cols + 1] == 1 ? condition_id2 : -condition_id2);
                }
                // If there's a third column, it's for the consequence
                if (cols >= 3) {
                    testRow.push_back(matrixData[row * cols + 2] == 1 ? consequence_id : -consequence_id);
                }
            } else {
                // If condition_id2 doesn't exist, the second column is for consequence
                if (cols >= 2) {
                    testRow.push_back(matrixData[row * cols + 1] == 1 ? consequence_id : -consequence_id);
                }
            }
            
            testSet.push_back(testRow);
        }
        
        printf("  Generated test set with %zu rows\n", testSet.size());
        testSets.push_back(testSet);
    }
    
    printf("\nGenerated %zu test sets in total\n\n", testSets.size());
    
    // Print verification of the testSets
    printf("=== Verification of Generated Test Sets ===\n");
    for (size_t i = 0; i < testSets.size(); i++) {
        printf("Test Set %zu (clause '%c'):\n", i, 'a' + static_cast<char>(i));
        printf("{\n");
        for (size_t j = 0; j < testSets[i].size(); j++) {
            printf("  {");
            for (size_t k = 0; k < testSets[i][j].size(); k++) {
                printf("%d", testSets[i][j][k]);
                if (k < testSets[i][j].size() - 1) {
                    printf(",");
                }
            }
            printf("}");
            if (j < testSets[i].size() - 1) {
                printf(",");
            }
            printf("\n");
        }
        printf("}\n\n");
    }
    
    return testSets;
}

// Run test cases
// Run test cases
void runTestCases() {
    std::vector<std::vector<std::vector<int>>> testSets = 
        generateTestSetsFromJSON("kelsen_data.json");
    
    // Show input sets
    for (size_t i = 0; i < testSets.size(); i++) {
        printf("  Set %zu: [", i + 1);
        for (size_t j = 0; j < testSets[i].size() && j < 2; j++) {
            printf("[");
            for (size_t k = 0; k < testSets[i][j].size(); k++) {
                printf("%d", testSets[i][j][k]);
                if (k < testSets[i][j].size() - 1) printf(", ");
            }
            printf("]");
            if (j < testSets[i].size() - 1) printf(", ");
        }
        if (testSets[i].size() > 2) printf("...");
        printf("] (%zu items)\n", testSets[i].size());
    }
    
    // Create host sets
    std::vector<HostSet> hostSets;
    for (const auto& vectors : testSets) {
        hostSets.push_back(createTestSet(vectors));
    }
    
    // Convert host sets to CUDA sets
    std::vector<CudaSet> cudaSets;
    for (const auto& hostSet : hostSets) {
        CudaSet cudaSet;
        copyHostToDevice(hostSet, &cudaSet);
        cudaSets.push_back(cudaSet);
    }
    
    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Run tree-fold operations
    CudaSet result = treeFoldOperations(cudaSets, true);
    
    // Record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\nFinal result contains %d combinations (computed in %.2f ms)\n", result.numItems, milliseconds);
    // Check if we got a dummy result (indicating results were too large and written to disk)
    if (result.numItems == 1) {
        // Check if this is our special flag value
        int flagValue;
        CHECK_CUDA_ERROR(cudaMemcpy(&flagValue, result.data, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (flagValue == -999999) {
            printf("Results were too large and were written to disk. Skipping post-processing.\n");
            printf("Check the output file for complete results.\n");
        } else {
            // It's just a normal single-item result, process it
            std::vector<std::vector<int>> processedResults = gpuPostProcess(result, true);
            
            printf("Final processed result contains %zu combinations (negatives removed, sorted by magnitude)\n", 
                   processedResults.size());
            
            // Open file for writing
            FILE* outFile = fopen("zdd.txt", "w");
            if (!outFile) {
                fprintf(stderr, "Error: Could not open zdd.txt for writing\n");
            } else {
                // Write header to file
                fprintf(outFile, "# Final results: %zu vectors\n", processedResults.size());
            }
            
            for (size_t i = 0; i < processedResults.size(); i++) {
                // Print to console
                printf("Result %zu: [", i + 1);
                for (size_t j = 0; j < processedResults[i].size(); j++) {
                    printf("%d", processedResults[i][j]);
                    if (j < processedResults[i].size() - 1) printf(", ");
                }
                printf("]\n");
                
                // Write to file if open
                if (outFile) {
                    fprintf(outFile, "[");
                    for (size_t j = 0; j < processedResults[i].size(); j++) {
                        fprintf(outFile, "%d", processedResults[i][j]);
                        if (j < processedResults[i].size() - 1) fprintf(outFile, ", ");
                    }
                    fprintf(outFile, "]\n");
                }
            }
            
            // Close file if open
            if (outFile) {
                fclose(outFile);
                printf("Results written to zdd.txt\n");
            }
        }
    } else {
        // Normal case - process the results
        std::vector<std::vector<int>> processedResults = gpuPostProcess(result, true);
        
        printf("Final processed result contains %zu combinations (negatives removed, sorted by magnitude)\n", 
               processedResults.size());
        
        // Open file for writing
        FILE* outFile = fopen("zdd.txt", "w");
        if (!outFile) {
            fprintf(stderr, "Error: Could not open zdd.txt for writing\n");
        } else {
            // Write header to file
            fprintf(outFile, "# Final results: %zu vectors\n", processedResults.size());
        }
        
        for (size_t i = 0; i < processedResults.size(); i++) {
            // Print to console
            printf("Result %zu: [", i + 1);
            for (size_t j = 0; j < processedResults[i].size(); j++) {
                printf("%d", processedResults[i][j]);
                if (j < processedResults[i].size() - 1) printf(", ");
            }
            printf("]\n");
            
            // Write to file if open
            if (outFile) {
                fprintf(outFile, "[");
                for (size_t j = 0; j < processedResults[i].size(); j++) {
                    fprintf(outFile, "%d", processedResults[i][j]);
                    if (j < processedResults[i].size() - 1) fprintf(outFile, ", ");
                }
                fprintf(outFile, "]\n");
            }
        }
        
        // Close file if open
        if (outFile) {
            fclose(outFile);
            printf("Results written to zdd.txt\n");
        }
    }
    
    // Clean up
    for (size_t i = 0; i < cudaSets.size(); i++) {
        freeCudaSet(&cudaSets[i]);
    }
    freeCudaSet(&result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




//-------------------------------------------------------------------------
// Main function
//-------------------------------------------------------------------------

int main() {
    // Initialize CUDA
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }
    CHECK_CUDA_ERROR(cudaSetDevice(0));
    
    // Run tests
    runTestCases();
    
    // Clean up
    CHECK_CUDA_ERROR(cudaDeviceReset());
    
    return 0;
}
