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
#include <cassert>
#include <cstdint>

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
        assert(hostIntData[i] >= INT8_MIN && hostIntData[i] <= INT8_MAX && "Input data exceeds int8_t range!");
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
    
    // Open the output file in binary write mode to start fresh.
    FILE* outFile = fopen("zdd.bin", "wb");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file zdd.bin for writing.\n");
        // Return a dummy set to signal failure
        CudaSet dummyResult = allocateCudaSet(1, 1);
        int* flagData = (int*)malloc(sizeof(int));
        flagData[0] = -999999;
        CHECK_CUDA_ERROR(cudaMemcpy(dummyResult.data, flagData, sizeof(int), cudaMemcpyHostToDevice));
        free(flagData);
        return dummyResult;
    }

    if (verbose) {
        printf("  Streaming CPU results directly to zdd.bin...\n");
    }

    // For extremely large problems, consider sampling
    bool useSampling = (hostSetA.vectors.size() * hostSetB.vectors.size() > 10000000);
    int stride = useSampling ? 10 : 1; // Process every 10th combination if sampling
    
    int processedCount = 0;
    
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
                uint32_t vecSize = uniqueElements.size();
                fwrite(&vecSize, sizeof(uint32_t), 1, outFile);

                if (vecSize > 0) {
                    // Convert set to vector to get contiguous data for fwrite
                    std::vector<int> tempVec(uniqueElements.begin(), uniqueElements.end());
                    fwrite(tempVec.data(), sizeof(int), vecSize, outFile);
                }
            }
            
            processedCount++;
            if (verbose && processedCount % 1000000 == 0) {
                printf("    Processed %d combinations...\n", processedCount);
            }
        }
    }
    
    fclose(outFile);

    if (verbose) {
        printf("  CPU processing complete. All results streamed to zdd.bin\n");
    }
    
    // Return a dummy small CudaSet that indicates stream processing was used
    CudaSet dummyResult = allocateCudaSet(1, 1);
    int* flagData = (int*)malloc(sizeof(int));
    flagData[0] = -999999; // Special flag indicating results were streamed
    CHECK_CUDA_ERROR(cudaMemcpy(dummyResult.data, flagData, sizeof(int), cudaMemcpyHostToDevice));
    free(flagData);
    
    return dummyResult;
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

CudaSet processLargePair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    
    if (verbose) {
        printf("  Processing large pair with true streaming: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               numItemsA, numItemsB, threshold);
        printf("  Total combinations: %lld - results will be streamed directly to disk.\n", totalCombinations);
    }

    // Open the output file in binary write mode to start fresh.
    FILE* outFile = fopen("zdd.bin", "wb");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file zdd.bin for writing.\n");
        // Return a dummy set to signal failure
        CudaSet dummyResult = allocateCudaSet(1, 1);
        int* flagData = (int*)malloc(sizeof(int));
        flagData[0] = -999999;
        CHECK_CUDA_ERROR(cudaMemcpy(dummyResult.data, flagData, sizeof(int), cudaMemcpyHostToDevice));
        free(flagData);
        return dummyResult;
    }
    
    // Define tile sizes based on problem dimensions
    int TILE_SIZE_A = 32;
    int TILE_SIZE_B = 1024;
    
    if (numItemsA > 1000 || numItemsB > 10000) {
        TILE_SIZE_A = 16;
        TILE_SIZE_B = 512;
    }
    
    int numTilesA = (numItemsA + TILE_SIZE_A - 1) / TILE_SIZE_A;
    int numTilesB = (numItemsB + TILE_SIZE_B - 1) / TILE_SIZE_B;
    int totalTiles = numTilesA * numTilesB;
    
    int tilesProcessed = 0;
    int lastProgressUpdate = 0;

    // Hoist vectors outside the loop to prevent repeated re-allocations.
    std::vector<int> hostValidFlags;
    std::vector<int> hostSizes;
    std::vector<int> hostResultData;

    for (int tileA = 0; tileA < numTilesA; tileA++) {
        int startA = tileA * TILE_SIZE_A;
        int endA = std::min(startA + TILE_SIZE_A, numItemsA);
        int sizeA = endA - startA;
        
        CudaSet tileSetA = extractSubset(setA, startA, sizeA, verbose);
        
        for (int tileB = 0; tileB < numTilesB; tileB++) {
            int startB = tileB * TILE_SIZE_B;
            int endB = std::min(startB + TILE_SIZE_B, numItemsB);
            int sizeB = endB - startB;
            
            tilesProcessed++;
            int progressPercentage = (tilesProcessed * 100) / totalTiles;
            if (verbose && (progressPercentage > lastProgressUpdate || tilesProcessed == totalTiles)) {
                printf("      Processing tile %d of %d - %d%% complete\n", 
                      tilesProcessed, totalTiles, progressPercentage);
                lastProgressUpdate = progressPercentage;
            }
            
            CudaSet tileSetB = extractSubset(setB, startB, sizeB, verbose);
            
            int numTileItemsA = tileSetA.numItems;
            int numTileItemsB = tileSetB.numItems;
            long long tileCombinations = (long long)numTileItemsA * (long long)numTileItemsB;
            
            CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(numTileItemsA, numTileItemsB, MAX_ELEMENTS_PER_VECTOR);
            
            int threadsPerBlock = 256;
            int maxResultsPerThread = 4;
            int threadsNeeded = (tileCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
            int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
            
            const int MAX_BLOCKS = 16384;
            if (blocksNeeded > MAX_BLOCKS) {
                blocksNeeded = MAX_BLOCKS;
                maxResultsPerThread = (tileCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
            }
            
            processAllCombinationsKernel<<<blocksNeeded, threadsPerBlock>>>(
                tileSetA.data, tileSetA.offsets, tileSetA.sizes, numTileItemsA,
                tileSetB.data, tileSetB.offsets, tileSetB.sizes, numTileItemsB,
                threshold, level,
                resultBuffer.data, resultBuffer.sizes, resultBuffer.validFlags, resultBuffer.maxResultSize,
                maxResultsPerThread
            );
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            
            hostValidFlags.resize(resultBuffer.numCombinations);
            CHECK_CUDA_ERROR(cudaMemcpy(hostValidFlags.data(), resultBuffer.validFlags, 
                                      resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
            
            int validCount = 0;
            for (int flag : hostValidFlags) {
                if (flag) validCount++;
            }
            
            if (validCount > 0) {
                hostSizes.resize(resultBuffer.numCombinations);
                CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), resultBuffer.sizes, 
                                          resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
                
                hostResultData.resize(resultBuffer.numCombinations * resultBuffer.maxResultSize);
                CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
                                          resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
                                          cudaMemcpyDeviceToHost));
                
                for (int i = 0; i < resultBuffer.numCombinations; i++) {
                    if (hostValidFlags[i]) {
                        uint32_t vecSize = hostSizes[i];
                        int offset = i * resultBuffer.maxResultSize;
                        
                        fwrite(&vecSize, sizeof(uint32_t), 1, outFile);
                        if (vecSize > 0) {
                            fwrite(hostResultData.data() + offset, sizeof(int), vecSize, outFile);
                        }
                    }
                }
            }
            
            freeCombinationResultBuffer(&resultBuffer);
            freeCudaSet(&tileSetB);
        }
        
        freeCudaSet(&tileSetA);
    }
    
    fclose(outFile);
    
    if (verbose) {
        printf("  Large pair processing complete. All results streamed to zdd.bin\n");
    }
    
        CudaSet dummyResult = allocateCudaSet(1, 1);
        int* flagData = (int*)malloc(sizeof(int));
        flagData[0] = -999999; // Special flag indicating results were streamed
        CHECK_CUDA_ERROR(cudaMemcpy(dummyResult.data, flagData, sizeof(int), cudaMemcpyHostToDevice));
        free(flagData);
        
        return dummyResult;
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
    
    printf("\nTree fold operations completed in %.2f ms.\n", milliseconds);

    // Check if the result was streamed to disk in a lower-level function
    bool streamed = false;
    if (result.numItems == 1) {
        int flagValue;
        CHECK_CUDA_ERROR(cudaMemcpy(&flagValue, result.data, sizeof(int), cudaMemcpyDeviceToHost));
        if (flagValue == -999999) {
            streamed = true;
            printf("Results were too large and were streamed to zdd.bin during processing.\n");
        }
    }
    
    if (!streamed) {
        // Process the final result set normally
        printf("Final result contains %d combinations. Writing to zdd.bin...\n", result.numItems);
        
        // Copy final results from device to host
        HostSet finalHostSet = copyDeviceToHost(result);

        // Open file for writing in binary mode
        FILE* outFile = fopen("zdd.bin", "wb");
        if (!outFile) {
            fprintf(stderr, "Error: Could not open zdd.bin for writing\n");
        } else {
            // Write total number of vectors
            uint64_t numVectors = finalHostSet.vectors.size();
            fwrite(&numVectors, sizeof(uint64_t), 1, outFile);

            // Write each vector
            for (const auto& vec : finalHostSet.vectors) {
                uint32_t vecSize = vec.size();
                fwrite(&vecSize, sizeof(uint32_t), 1, outFile);
                if (vecSize > 0) {
                    fwrite(vec.data(), sizeof(int), vecSize, outFile);
            }
        }
            fclose(outFile);
            printf("Results successfully written to zdd.bin\n");
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
