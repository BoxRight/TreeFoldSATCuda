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
#include <execution>
#include <functional>
#include <malloc.h>
#include <zstd.h>

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
#define TILE_SIZE_A 256  // Tile size for set A in tiled processing
#define TILE_SIZE_B 3072 // Tile size for set B in tiled processing
#define RESULTS_FLUSH_THRESHOLD 10000 // In-memory result limit before flushing to disk
#define CHUNK_SIZE (1024 * 4) // Number of items to load from a stream at a time

// --- Forward Declarations ---
typedef struct {
    int8_t* data;         // Flattened array of all elements
    int* offsets;      // Starting index for each vector/set
    int* sizes;        // Size of each vector/set
    int numItems;      // Number of vectors/sets
    int totalElements; // Total number of elements
    int8_t* deviceBuffer; // Reusable device buffer for operations
    int bufferSize;    // Size of the device buffer
} CudaSet;
struct LevelItem;
struct ProcessResult;
ProcessResult processPair_inMemory(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose);
void processLargePair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, 
                     std::unordered_map<size_t, std::vector<int>>& uniqueResults,
                     std::function<void()> flushCallback = nullptr);
ProcessResult processPair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, bool allowStreaming);
LevelItem processStreamedPair(LevelItem& itemA, LevelItem& itemB, int threshold, int level, bool verbose);

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



// Result buffer for parallel combination processing
typedef struct {
    int* data;         // Buffer for all potential results
    int* validFlags;   // Flags indicating if each combination is valid
    int* sizes;        // Size of each result set
    int maxResultSize; // Maximum possible size of a result
    int numCombinations; // Total number of combinations
} CombinationResultBuffer;

// Result struct to handle in-memory or streamed results
struct ProcessResult {
    CudaSet set;
    std::string streamPath; // Path to file if results are streamed
    int fromIdA = -1;
    int fromIdB = -1;
    size_t numResultItems = 0;
};


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
        CHECK_CUDA_ERROR(cudaMalloc(&set.deviceBuffer, bufferSize * sizeof(int8_t)));
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

// Represents an item in a processing level of the tree fold
struct LevelItem {
    CudaSet set;
    std::string streamPath;
    int numItems;
    int id;
    bool needsCleanup; // True if this is an intermediate result that should be freed/deleted

    bool isStreamed() const { return !streamPath.empty(); }
};

// Global counter for unique item IDs
static int levelItemCounter = 0;

// Helper to get the first vector from a CudaSet for threshold calculation
std::vector<int> getFirstVectorFromCudaSet(const CudaSet& set) {
    if (set.numItems == 0) return {};
    int size;
    CHECK_CUDA_ERROR(cudaMemcpy(&size, set.sizes, sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<int8_t> h_firstVector8(size);
    int offset = 0; // First vector is always at offset 0
    CHECK_CUDA_ERROR(cudaMemcpy(h_firstVector8.data(), set.data + offset, size * sizeof(int8_t), cudaMemcpyDeviceToHost));
    
    std::vector<int> firstVector(size);
    for(int i = 0; i < size; ++i) firstVector[i] = h_firstVector8[i];
    return firstVector;
}

// Helper to get the first vector from a streamed file
std::vector<int> getFirstVectorFromStream(const std::string& filePath) {
    FILE* inFile = fopen(filePath.c_str(), "rb");
    if (!inFile) return {};

    int vecSize = 0;
    size_t elementsRead = fread(&vecSize, sizeof(int), 1, inFile);
    if (elementsRead == 0) {
        fclose(inFile);
        return {};
    }

    std::vector<int> firstVec(vecSize);
    fread(firstVec.data(), sizeof(int), vecSize, inFile);
    fclose(inFile);
    return firstVec;
}

// Modified threshold computation to handle streamed and in-memory sets
int computeThreshold(const LevelItem& itemA, const LevelItem& itemB) {
    if (itemA.numItems == 0 || itemB.numItems == 0) return 0;

    // Get the first vector from item A
    std::vector<int> firstVectorA = itemA.isStreamed() ? 
        getFirstVectorFromStream(itemA.streamPath) : 
        getFirstVectorFromCudaSet(itemA.set);

    // Get the first vector from item B
    std::vector<int> firstVectorB = itemB.isStreamed() ? 
        getFirstVectorFromStream(itemB.streamPath) : 
        getFirstVectorFromCudaSet(itemB.set);

    if (firstVectorA.empty() || firstVectorB.empty()) return 0;
    
    // The rest of the logic is the same: find unique absolute values
    std::set<int> uniqueAbsValues;
    for (int value : firstVectorA) uniqueAbsValues.insert(abs(value));
    for (int value : firstVectorB) uniqueAbsValues.insert(abs(value));
        
    return uniqueAbsValues.size();
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

// Flushes a batch of results to disk to keep host RAM usage low, using Zstandard compression.
// Each flush operation writes one or more compressed chunks to the file.
size_t flushResultsToDisk(std::unordered_map<size_t, std::vector<int>>& results, 
                        const char* outputPath, bool& isFirstWrite, bool verbose) {
    if (results.empty()) {
        return 0;
    }

    size_t itemsToFlush = results.size();
    if (verbose) {
        printf("    Flushing %zu items to %s (append: %s)\n", itemsToFlush, outputPath, isFirstWrite ? "false" : "true");
    }

    FILE* outFile = fopen(outputPath, isFirstWrite ? "wb" : "ab");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file %s for appending\n", outputPath);
        return 0;
    }

    if (isFirstWrite) {
        isFirstWrite = false;
    }

    const int BATCH_SIZE = 50000;
    std::vector<std::vector<int>> batchVectors;
    batchVectors.reserve(BATCH_SIZE);

    auto it = results.begin();
    while (it != results.end()) {
        // Collect a batch of vectors
        for (int i = 0; i < BATCH_SIZE && it != results.end(); ++i) {
            batchVectors.push_back(std::move(it->second));
            ++it;
        }

        if (batchVectors.empty()) continue;

        // Sort the batch lexicographically before serialization
        std::sort(std::execution::par, batchVectors.begin(), batchVectors.end());

        // Serialize the batch
        std::vector<char> serializedBatch;
        size_t totalElements = 0;
        for(const auto& vec : batchVectors) totalElements += vec.size();
        serializedBatch.reserve(sizeof(size_t) + batchVectors.size() * sizeof(int) + totalElements * sizeof(int));
        
        size_t numVectorsInBatch = batchVectors.size();
        serializedBatch.insert(serializedBatch.end(), (char*)&numVectorsInBatch, (char*)&numVectorsInBatch + sizeof(size_t));

        for (const auto& vec : batchVectors) {
            int vecSize = vec.size();
            serializedBatch.insert(serializedBatch.end(), (char*)&vecSize, (char*)&vecSize + sizeof(int));
            serializedBatch.insert(serializedBatch.end(), (char*)vec.data(), (char*)vec.data() + vecSize * sizeof(int));
        }

        // Compress the serialized batch
        size_t const cBuffSize = ZSTD_compressBound(serializedBatch.size());
        std::vector<char> compressedBatch(cBuffSize);
        size_t const cSize = ZSTD_compress(compressedBatch.data(), cBuffSize, serializedBatch.data(), serializedBatch.size(), 1);
        
        if (ZSTD_isError(cSize)) {
            fprintf(stderr, "ZSTD compression error: %s\n", ZSTD_getErrorName(cSize));
            fclose(outFile);
            return 0;
        }

        // Write the compressed chunk to disk: [size][data]
        uint64_t compressedSize = cSize;
        fwrite(&compressedSize, sizeof(uint64_t), 1, outFile);
        fwrite(compressedBatch.data(), 1, compressedSize, outFile);

        batchVectors.clear();
    }

    fclose(outFile);

    // CRITICAL: Clear the map to free host RAM
    results.clear();
    
    // Force the hash map to release its memory by swapping with an empty map
    std::unordered_map<size_t, std::vector<int>>().swap(results);
    
    // Force allocator to return memory to OS
    malloc_trim(0);
    
    if (verbose) {
        printf("    Flush complete. RAM freed.\n");
    }
    return itemsToFlush;
}

// Loads a compressed chunk of vectors from a binary file into a CudaSet
CudaSet loadCudaSetChunkFromBinary(const char* filePath, long long& fileOffset, int maxVectorsToLoad, bool verbose) {
    // Open file in binary read mode
    FILE* inFile = fopen(filePath, "rb");
    if (!inFile) {
        if (verbose) printf("    Warning: Could not open file for chunk loading: %s\n", filePath);
        CudaSet emptySet = {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
        return emptySet;
    }

    // Seek to the starting offset
    fseek(inFile, fileOffset, SEEK_SET);

    // Read the size of the next compressed chunk
    uint64_t compressedSize = 0;
    size_t elementsRead = fread(&compressedSize, sizeof(uint64_t), 1, inFile);
    if (elementsRead == 0) {
        // End of file
        fclose(inFile);
        CudaSet emptySet = {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
        return emptySet;
    }

    // Read the compressed chunk
    std::vector<char> compressedChunk(compressedSize);
    fread(compressedChunk.data(), 1, compressedSize, inFile);

    // Decompress the chunk
    unsigned long long const rSize = ZSTD_getFrameContentSize(compressedChunk.data(), compressedSize);
    if (rSize == ZSTD_CONTENTSIZE_ERROR || rSize == ZSTD_CONTENTSIZE_UNKNOWN) {
        fprintf(stderr, "Error: ZSTD cannot get decompressed size of the frame\n");
        fclose(inFile);
        return {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
    }
    
    std::vector<char> decompressedChunk(rSize);
    size_t const dSize = ZSTD_decompress(decompressedChunk.data(), rSize, compressedChunk.data(), compressedSize);

    if (ZSTD_isError(dSize) || dSize != rSize) {
        fprintf(stderr, "ZSTD decompression error: %s\n", ZSTD_getErrorName(dSize));
        fclose(inFile);
        return {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
    }

    // Deserialize the vectors from the decompressed chunk
    HostSet hostSet;
    char* bufferPtr = decompressedChunk.data();
    
    size_t numVectorsInChunk = *(size_t*)bufferPtr;
    bufferPtr += sizeof(size_t);
    
    hostSet.vectors.reserve(numVectorsInChunk);

    for (size_t i = 0; i < numVectorsInChunk; ++i) {
        int vecSize = *(int*)bufferPtr;
        bufferPtr += sizeof(int);
        
        std::vector<int> tempVec(vecSize);
        memcpy(tempVec.data(), bufferPtr, vecSize * sizeof(int));
        bufferPtr += vecSize * sizeof(int);

        hostSet.vectors.push_back(std::move(tempVec));
    }

    // Update the file offset for the next call
    fileOffset = ftell(inFile);
    fclose(inFile);

    // If no vectors were loaded, return an empty set
    if (hostSet.vectors.empty()) {
        CudaSet emptySet = {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
        return emptySet;
    }

    if (verbose) {
        printf("    Loaded and decompressed chunk of %zu vectors from %s\n", hostSet.vectors.size(), filePath);
    }

    // Convert the host set to a CudaSet and return
    CudaSet cudaSet;
    copyHostToDevice(hostSet, &cudaSet);
    return cudaSet;
}

// Global counter for unique stream file names
static int streamFileCounter = 0;

// Helper function to get current memory usage
size_t getCurrentMemoryUsage() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label, value, unit;
            iss >> label >> value >> unit;
            return std::stoull(value) * 1024; // Convert KB to bytes
        }
    }
    return 0;
}

// An internal version of processPair that is guaranteed to run on the GPU without triggering another streaming operation.
// It also contains the full deduplication logic.
ProcessResult processPair_inMemory(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose) {
    long long totalCombinations = (long long)setA.numItems * (long long)setB.numItems;
    
    // Calculate buffer size needed
    int maxResultsPerThread = 4;
    int threadsNeeded = (totalCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
    
    // Determine thread block configuration
    int threadsPerBlock = 256;
    int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
    
    // Limit blocks to avoid excessive memory usage
    const int MAX_BLOCKS = 16384;
    if (blocksNeeded > MAX_BLOCKS) {
        blocksNeeded = MAX_BLOCKS;
        maxResultsPerThread = (totalCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
    }
    
    // Allocate result buffer
    CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(setA.numItems, setB.numItems, MAX_ELEMENTS_PER_VECTOR);
    
    // Launch kernel
    processAllCombinationsKernel<<<blocksNeeded, threadsPerBlock>>>(
        setA.data, setA.offsets, setA.sizes, setA.numItems,
        setB.data, setB.offsets, setB.sizes, setB.numItems,
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

    if (verbose) {
        printf("    Found %d valid combinations out of %lld total\n", validCount, totalCombinations);
    }

    std::vector<std::vector<int>> validCombinations;
	if (validCount > 0) {
        if (verbose) {
		    printf("    Copying result data for %d valid combinations...\n", validCount);
		}

        std::vector<int> hostSizes(resultBuffer.numCombinations);
        CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), resultBuffer.sizes, 
                                  resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
		
		std::vector<int> hostResultData(resultBuffer.numCombinations * resultBuffer.maxResultSize);
		CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
		                          resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
		                          cudaMemcpyDeviceToHost));
		
        // Progress reporting variables
		int reportInterval = validCount > 1000 ? validCount / 10 : validCount;
		int lastReportedCount = 0;
		int collectedCount = 0;

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

	freeCombinationResultBuffer(&resultBuffer);

    // Remove duplicates
    if (validCombinations.size() > 1) {
        if (verbose) {
            printf("    Deduplicating %zu combinations...\n", validCombinations.size());
        }
        for (auto& combination : validCombinations) {
            std::sort(combination.begin(), combination.end());
        }
        std::sort(std::execution::par, validCombinations.begin(), validCombinations.end());
        validCombinations.erase(std::unique(validCombinations.begin(), validCombinations.end()), validCombinations.end());
        if (verbose) {
            printf("    Deduplication complete: %zu unique combinations.\n", validCombinations.size());
        }
    }

    // Create result set
    HostSet resultHostSet;
    resultHostSet.vectors = validCombinations;
    
    CudaSet resultCudaSet;
    copyHostToDevice(resultHostSet, &resultCudaSet);
    
    return {resultCudaSet, ""};
}


ProcessResult processPair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, bool allowStreaming = true) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    
    if (verbose) {
        printf("  Processing pair at level %d: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               level, numItemsA, numItemsB, threshold);
    }
    
    // Empty result for empty inputs
    if (numItemsA == 0 || numItemsB == 0) {
        CudaSet emptySet = {nullptr, nullptr, nullptr, 0, 0, nullptr, 0};
        return {emptySet, ""};
    }

    // For extremely large combinations, use the memory-efficient approach (if allowed)
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    if (allowStreaming && totalCombinations > 3000000LL) { // 3 million threshold
        char streamFilePath[256];
        sprintf(streamFilePath, "zdd_stream_level%d_file%d.bin", level, streamFileCounter++);
        
        // This is now a two-step process: compute, then write.
        std::unordered_map<size_t, std::vector<int>> results;
        processLargePair(setA, setB, threshold, level, verbose, results);
        bool isFirstWrite = true;
        flushResultsToDisk(results, streamFilePath, isFirstWrite, verbose);
        
        return { {nullptr, nullptr, nullptr, 0, 0, nullptr, 0}, streamFilePath};
    }
    
    // All other cases are processed in-memory.
    if (verbose) {
        printf("    Using in-memory GPU processing.\n");
    }
    return processPair_inMemory(setA, setB, threshold, level, verbose);
}


void processLargePair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, 
                     std::unordered_map<size_t, std::vector<int>>& uniqueResults,
                     std::function<void()> flushCallback) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    
    if (verbose) {
        printf("  Processing large pair with tiled approach: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               numItemsA, numItemsB, threshold);
        printf("  Total combinations: %lld - using tiled processing\n", totalCombinations);
    }
    
    // Calculate number of tiles
    int numTilesA = (numItemsA + TILE_SIZE_A - 1) / TILE_SIZE_A;
    int numTilesB = (numItemsB + TILE_SIZE_B - 1) / TILE_SIZE_B;
    int totalTiles = numTilesA * numTilesB;
    
    if (verbose) {
        printf("    Processing in %d x %d = %d tiles\n", numTilesA, numTilesB, totalTiles);
        printf("    Tile dimensions: %d x %d items\n", TILE_SIZE_A, TILE_SIZE_B);
    }
    
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
        CudaSet tileSetA = extractSubset(setA, startA, sizeA, false);
        
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
            CudaSet tileSetB = extractSubset(setB, startB, sizeB, false);
            
            // Process this tile pair directly
            int numTileItemsA = tileSetA.numItems;
            int numTileItemsB = tileSetB.numItems;
            
            if (numTileItemsA == 0 || numTileItemsB == 0) {
                freeCudaSet(&tileSetB);
                continue;
            }
            
            ProcessResult chunkResult = processPair_inMemory(tileSetA, tileSetB, threshold, level, false);

            if (chunkResult.set.numItems > 0) {
                HostSet hostResult = copyDeviceToHost(chunkResult.set);
                for (auto& vec : hostResult.vectors) {
                    std::sort(vec.begin(), vec.end());
                    size_t hash = hashVector(vec);
                    if (uniqueResults.find(hash) == uniqueResults.end()) {
                        uniqueResults[hash] = std::move(vec);
                    }
                }
                freeCudaSet(&chunkResult.set);
            }
            
            // Free tile resources
            freeCudaSet(&tileSetB);
            
            // Check if we need to flush results periodically during tiled processing
            if (flushCallback && uniqueResults.size() > RESULTS_FLUSH_THRESHOLD) {
                if (verbose) {
                    printf("      Flushing results during tiled processing (%zu items)...\n", uniqueResults.size());
                }
                flushCallback();
                
                // Force GPU memory cleanup after flush
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                CHECK_CUDA_ERROR(cudaMemGetInfo(nullptr, nullptr)); // Force memory manager update
                
                // Additional aggressive cleanup
                if (verbose) {
                    printf("      Post-flush cleanup: uniqueResults size = %zu\n", uniqueResults.size());
                }
            }
            
            // More frequent flushing for very large tile results
            if (flushCallback && uniqueResults.size() > RESULTS_FLUSH_THRESHOLD / 4) {
                if (verbose) {
                    printf("      Early flush triggered at %zu items\n", uniqueResults.size());
                }
                flushCallback();
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            }
        }
        
        // Free tile resources
        freeCudaSet(&tileSetA);
    }
    
    if (verbose) {
        printf("  Tiled processing complete. Results added to shared collection.\n");
    }
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

// Core function to process a pair where at least one input is streamed to disk
LevelItem processStreamedPair(LevelItem& itemA, LevelItem& itemB, int threshold, int level, bool verbose) {
    char outStreamPath[256];
    sprintf(outStreamPath, "zdd_stream_level%d_file%d.bin", level, streamFileCounter++);
    
    if (verbose) {
        printf("  Processing streamed pair -> %s\n", outStreamPath);
        printf("    Item A (ID %d): %s (%d items)\n", itemA.id, itemA.isStreamed() ? itemA.streamPath.c_str() : "in-memory", itemA.numItems);
        printf("    Item B (ID %d): %s (%d items)\n", itemB.id, itemB.isStreamed() ? itemB.streamPath.c_str() : "in-memory", itemB.numItems);
    }

    std::unordered_map<size_t, std::vector<int>> uniqueResults;
    bool isFirstWrite = true;
    size_t totalItemsWritten = 0;
    
    auto hashVector = [](const std::vector<int>& vec) {
        size_t hash = vec.size();
        for (int val : vec) hash ^= std::hash<int>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        return hash;
    };

    long long offsetA = 0;
    while(offsetA < itemA.numItems) {
        CudaSet chunkA;
        long long currentChunkSizeA = std::min((long long)CHUNK_SIZE, (long long)itemA.numItems - offsetA);

        if (itemA.isStreamed()) {
            chunkA = loadCudaSetChunkFromBinary(itemA.streamPath.c_str(), offsetA, currentChunkSizeA, false);
        } else {
            chunkA = extractSubset(itemA.set, offsetA, currentChunkSizeA, false);
        }

        if(verbose) printf("    Loaded chunk A (items %lld-%lld) with %d items.\n", offsetA, offsetA + chunkA.numItems -1, chunkA.numItems);
        if (chunkA.numItems == 0) {
            offsetA += currentChunkSizeA;
            continue;
        }

        long long offsetB = 0;
        while(offsetB < itemB.numItems) {
            CudaSet chunkB;
            long long currentChunkSizeB = std::min((long long)CHUNK_SIZE, (long long)itemB.numItems - offsetB);
            
            if (itemB.isStreamed()) {
                chunkB = loadCudaSetChunkFromBinary(itemB.streamPath.c_str(), offsetB, currentChunkSizeB, false);
            } else {
                chunkB = extractSubset(itemB.set, offsetB, currentChunkSizeB, false);
            }
            
            if(verbose) printf("      Loaded chunk B (items %lld-%lld) with %d items.\n", offsetB, offsetB + chunkB.numItems - 1, chunkB.numItems);
            if (chunkB.numItems == 0) {
                offsetB += currentChunkSizeB;
                continue;
            }
            
            if(verbose) printf("      Processing chunk pair with tiled approach.\n");
            
            // Create a flush callback that will be called during tiled processing
            auto flushCallback = [&]() {
                size_t memBefore = getCurrentMemoryUsage();
                if(verbose) printf("    Flushing intermediate results to disk (%zu items, RAM: %.1f MB)...\n", 
                                 uniqueResults.size(), memBefore / 1024.0 / 1024.0);
                totalItemsWritten += flushResultsToDisk(uniqueResults, outStreamPath, isFirstWrite, verbose);
                
                std::unordered_map<size_t, std::vector<int>>().swap(uniqueResults);
                malloc_trim(0);
                
                size_t memAfter = getCurrentMemoryUsage();
                long long memDiff = (long long)memBefore - (long long)memAfter;
                if(verbose) printf("    Post-flush RAM: %.1f MB (change: %+.1f MB)\n", 
                                 memAfter / 1024.0 / 1024.0, memDiff / 1024.0 / 1024.0);
            };
            
            processLargePair(chunkA, chunkB, threshold, level, false, uniqueResults, flushCallback);

            if (uniqueResults.size() > RESULTS_FLUSH_THRESHOLD) {
                if(verbose) printf("    Flushing intermediate results to disk (%zu items)...\n", uniqueResults.size());
                totalItemsWritten += flushResultsToDisk(uniqueResults, outStreamPath, isFirstWrite, verbose);
                
                // Force memory cleanup after flush
                CHECK_CUDA_ERROR(cudaDeviceSynchronize());
                CHECK_CUDA_ERROR(cudaMemGetInfo(nullptr, nullptr)); // Force memory manager update
                malloc_trim(0);
            }
            
            freeCudaSet(&chunkB);
            offsetB += currentChunkSizeB;
        }
        
        freeCudaSet(&chunkA);
        offsetA += currentChunkSizeA;
    }
    
    // Final flush for any remaining results
    if (!uniqueResults.empty()) {
        if (verbose) printf("    Flushing final results to disk (%zu items)...\n", uniqueResults.size());
        totalItemsWritten += flushResultsToDisk(uniqueResults, outStreamPath, isFirstWrite, verbose);
    }
    
    if (verbose) printf("  --> Streamed pair processing complete. Result: %zu items in %s\n", totalItemsWritten, outStreamPath);

    return { {}, outStreamPath, (int)totalItemsWritten, levelItemCounter++, true };
}

// Tree fold operations (maintains sequential dependencies but optimizes within each step)
LevelItem treeFoldOperations(const std::vector<CudaSet>& sets, bool verbose) {
    if (sets.empty()) {
        return { {nullptr, nullptr, nullptr, 0, 0, nullptr, 0}, "", 0, -1, false };
    }

    std::vector<std::string> tempFiles; // Keep track of intermediate files to delete

    // Initialize the first level with LevelItems
    std::vector<LevelItem> currentLevel;
    for (const auto& s : sets) {
        currentLevel.push_back({s, "", s.numItems, levelItemCounter++, false});
    }

    if (currentLevel.size() == 1) {
        return currentLevel[0];
    }
    
    if (verbose) {
        printf("Starting tree-fold operations with %zu sets\n", sets.size());
        for (const auto& item : currentLevel) {
            printf("  Set %d: %d items\n", item.id, item.numItems);
        }
    }
    
    int level = 0;
    while (currentLevel.size() > 1) {
        level++;
        if (verbose) {
            printf("\nProcessing Level %d with %zu sets\n", level, currentLevel.size());
        }
        
        std::vector<LevelItem> nextLevel;
        std::vector<bool> processed(currentLevel.size(), false);
        
        // --- Phase 1: Process all possible in-memory pairs ---
        bool inMemoryPairFound;
        do {
            inMemoryPairFound = false;
            int bestI = -1, bestJ = -1;
            int lowestThreshold = INT_MAX;

            // Find the best in-memory pair among all non-processed items
            for (size_t i = 0; i < currentLevel.size(); i++) {
                if (processed[i]) continue;
                for (size_t j = i + 1; j < currentLevel.size(); j++) {
                    if (processed[j]) continue;
                    
                    bool willStream = currentLevel[i].isStreamed() || currentLevel[j].isStreamed() ||
                                      (long long)currentLevel[i].numItems * (long long)currentLevel[j].numItems >= 3000000LL;

                    if (!willStream) {
                        int threshold = computeThreshold(currentLevel[i], currentLevel[j]);
                        if (threshold < lowestThreshold) {
                            lowestThreshold = threshold;
                            bestI = i;
                            bestJ = j;
                        }
                    }
                }
            }

            if (bestI != -1) {
                inMemoryPairFound = true;
                processed[bestI] = true;
                processed[bestJ] = true;

                LevelItem& itemA = currentLevel[bestI];
                LevelItem& itemB = currentLevel[bestJ];

                if (verbose) {
                    printf("  --> Selected in-memory pair: Set %d (%d items) + Set %d (%d items) with threshold %d\n", 
                           itemA.id, itemA.numItems, itemB.id, itemB.numItems, lowestThreshold);
                    printf("      Processing pair in-memory (GPU).\n");
                }

                ProcessResult res = processPair(itemA.set, itemB.set, lowestThreshold, level, verbose, true);
                LevelItem resultItem = { res.set, res.streamPath, res.set.numItems, levelItemCounter++, true };
                if (!res.streamPath.empty()) {
                   tempFiles.push_back(res.streamPath);
                }
                nextLevel.push_back(resultItem);
            }
        } while (inMemoryPairFound);

        // --- Phase 2: Process all remaining items using streaming-optimized strategy ---
        std::vector<int> remainingIndices;
        for(size_t i = 0; i < currentLevel.size(); ++i) {
            if (!processed[i]) {
                remainingIndices.push_back(i);
            }
        }

        if (remainingIndices.size() >= 2) {
            if (verbose) {
                printf("  --> No in-memory pairs left. Applying streaming-optimized pairing to %zu sets.\n", remainingIndices.size());
            }

            std::sort(remainingIndices.begin(), remainingIndices.end(),
                      [&](int a, int b) { return currentLevel[a].numItems < currentLevel[b].numItems; });

            int i = 0;
            int j = remainingIndices.size() - 1;
            while (i < j) {
                int idxA = remainingIndices[i];
                int idxB = remainingIndices[j];
                processed[idxA] = true;
                processed[idxB] = true;

                LevelItem& itemA = currentLevel[idxA];
                LevelItem& itemB = currentLevel[idxB];

                int threshold = computeThreshold(itemA, itemB);
                if (verbose) {
                    printf("  --> Pairing smallest (%s:%d items) with largest (%s:%d items) for streaming.\n",
                           itemA.isStreamed() ? "stream" : "mem", itemA.numItems,
                           itemB.isStreamed() ? "stream" : "mem", itemB.numItems);
                }

                LevelItem resultItem = processStreamedPair(itemA, itemB, threshold, level, verbose);
                tempFiles.push_back(resultItem.streamPath);
                nextLevel.push_back(resultItem);

                i++;
                j--;
            }
        }
        
        // Handle any remaining odd set by carrying it over to the next level
        for(size_t i = 0; i < currentLevel.size(); ++i) {
            if(!processed[i]) {
                LevelItem& carriedItem = currentLevel[i];
                if (verbose) {
                    printf("  --> Carrying over odd set %d (%d items) to next level\n", 
                           carriedItem.id, carriedItem.numItems);
                }

                // For level 1, convert the carried-over set to unique elements, as per original logic.
                // This is a special operation that only happens on the first-level carry-over.
                if (level == 1) {
                   CudaSet convertedSet = convertSetToUnique(carriedItem.set, verbose);
                   // The new item is an intermediate result and will need cleanup
                   nextLevel.push_back({convertedSet, "", convertedSet.numItems, levelItemCounter++, true});
                } else {
                   // For other levels, just move the item to the next level.
                   // It's not a new intermediate result, so it doesn't need cleanup yet.
                   carriedItem.needsCleanup = false;
                   nextLevel.push_back(carriedItem);
                }
            }
        }
        
        // Clean up resources from the completed level that were marked for cleanup.
        for(const auto& item : currentLevel) {
            if(item.needsCleanup) {
                if(item.isStreamed()) {
                     remove(item.streamPath.c_str());
                } else {
                    freeCudaSet(&const_cast<CudaSet&>(item.set));
                }
            }
        }
        
        currentLevel = nextLevel;
    }
    
    LevelItem finalItem = currentLevel.empty() ? LevelItem{} : currentLevel[0];
    
    // Clean up all temporary files except the final result file
    for (const auto& file : tempFiles) {
        if (file != finalItem.streamPath) {
             remove(file.c_str());
        }
    }
    
    return finalItem;
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
    LevelItem finalResult = treeFoldOperations(cudaSets, true);
    
    // Record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("\nTree-fold completed in %.2f ms. Total items: %d\n", milliseconds, finalResult.numItems);
    
    std::vector<std::vector<int>> finalVectors;

    if (finalResult.isStreamed()) {
        printf("Final result is on disk (%s). Loading for post-processing...\n", finalResult.streamPath.c_str());
        // Load all vectors from the final stream file
        long long offset = 0;
        while(true) {
            CudaSet chunk = loadCudaSetChunkFromBinary(finalResult.streamPath.c_str(), offset, 100000, true);
            if (chunk.numItems == 0) break;
            
            HostSet hostChunk = copyDeviceToHost(chunk);
            finalVectors.insert(finalVectors.end(), std::make_move_iterator(hostChunk.vectors.begin()), std::make_move_iterator(hostChunk.vectors.end()));
            freeCudaSet(&chunk);
        }
        if (!finalResult.streamPath.empty()) {
             remove(finalResult.streamPath.c_str()); // Clean up final stream file
        }
        printf("Loaded %zu vectors from final stream file.\n", finalVectors.size());
        
        // Post-process on CPU, then write to file
        std::vector<std::vector<int>> processedResults;
        processedResults.reserve(finalVectors.size());
        for (auto& vec : finalVectors) {
            std::vector<int> positives;
            for (int val : vec) {
                if (val >= 0) positives.push_back(val);
            }
            std::sort(positives.begin(), positives.end());
            processedResults.push_back(std::move(positives));
        }
        std::sort(std::execution::par, processedResults.begin(), processedResults.end());
        finalVectors = std::move(processedResults);

    } else if (finalResult.numItems > 0) {
        printf("Final result is in memory (%d items). Post-processing on GPU...\n", finalResult.numItems);
        // Normal case - process the results from GPU memory
        finalVectors = gpuPostProcess(finalResult.set, true);
        freeCudaSet(&const_cast<CudaSet&>(finalResult.set));
    } else {
        printf("Final result is empty.\n");
    }
    
    printf("Final processed result contains %zu combinations\n", finalVectors.size());
    
    // Open file for writing
    FILE* outFile = fopen("zdd.bin", "wb");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open zdd.bin for writing\n");
    } else {
        for (const auto& vec : finalVectors) {
            int size = vec.size();
            fwrite(&size, sizeof(int), 1, outFile);
            fwrite(vec.data(), sizeof(int), size, outFile);
        }
        fclose(outFile);
        printf("Results written to zdd.bin\n");
    }
    
    // Clean up original sets
    for (size_t i = 0; i < cudaSets.size(); i++) {
        freeCudaSet(&cudaSets[i]);
    }
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
