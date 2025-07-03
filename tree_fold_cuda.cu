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
    // In-memory fields (only valid if backing_file is empty)
    int8_t* data;         // Flattened array of all elements
    int* offsets;      // Starting index for each vector/set
    int* sizes;        // Size of each vector/set
    int numItems;      // Number of vectors/sets
    int totalElements; // Total number of elements
    int8_t* deviceBuffer; // Reusable device buffer for operations
    int bufferSize;    // Size of the device buffer

    // On-disk field. If not empty, this set represents data on disk.
    std::string backing_file;
} CudaSet;

// A reader class to stream vectors from a .bin file into a CudaSet chunk.
class CudaSetReader {
private:
    std::ifstream file;
    std::string filename;
    bool verbose;

public:
    CudaSetReader(const std::string& fname, bool v = false) : filename(fname), verbose(v) {
        file.open(filename, std::ios::binary);
        if (!file) {
            fprintf(stderr, "CudaSetReader Error: Could not open file %s\n", filename.c_str());
        }
        if (verbose) {
            printf("CudaSetReader: Opened %s for reading.\n", filename.c_str());
        }
    }

    ~CudaSetReader() {
        if (file.is_open()) {
            file.close();
            if (verbose) {
                printf("CudaSetReader: Closed %s.\n", filename.c_str());
            }
        }
    }

    // Reads the next batch of vectors into the provided CudaSet buffer.
    // Returns the number of vectors actually read.
    int readNextChunk(CudaSet& chunk_buffer, int max_vectors_to_read) {
        if (!file.is_open() || file.peek() == EOF) {
            return 0; // End of file or file not open
        }

        HostSet hostSet;
        hostSet.vectors.reserve(max_vectors_to_read);
        int totalElementsInChunk = 0;

        for (int i = 0; i < max_vectors_to_read; ++i) {
            uint32_t vecSize;
            file.read(reinterpret_cast<char*>(&vecSize), sizeof(uint32_t));
            if (file.gcount() == 0) { // EOF
                break;
            }
            
            totalElementsInChunk += vecSize;
            std::vector<int> vec(vecSize);
            if (vecSize > 0) {
                file.read(reinterpret_cast<char*>(vec.data()), vecSize * sizeof(int));
            }
            hostSet.vectors.push_back(std::move(vec)); // Use move semantics
        }

        if (hostSet.vectors.empty()) {
            return 0;
        }

        // Now, prepare the data and copy it to the pre-allocated chunk_buffer on the device.
        int numItems = hostSet.vectors.size();
        
        std::vector<int> hostIntData;
        hostIntData.reserve(totalElementsInChunk);
        std::vector<int> hostOffsets(numItems);
        std::vector<int> hostSizes(numItems);
        
        int currentOffset = 0;
        for (int i = 0; i < numItems; i++) {
            hostOffsets[i] = currentOffset;
            hostSizes[i] = hostSet.vectors[i].size();
            hostIntData.insert(hostIntData.end(), hostSet.vectors[i].begin(), hostSet.vectors[i].end());
            currentOffset += hostSet.vectors[i].size();
        }
        
        std::vector<int8_t> hostData(hostIntData.size());
        for (size_t i = 0; i < hostIntData.size(); ++i) {
            assert(hostIntData[i] >= INT8_MIN && hostIntData[i] <= INT8_MAX && "Input data exceeds int8_t range!");
            hostData[i] = static_cast<int8_t>(hostIntData[i]);
        }
        
        int totalElements = hostData.size();
        assert(totalElements == totalElementsInChunk);

        // The caller is responsible for ensuring chunk_buffer is large enough.
        assert(totalElements <= chunk_buffer.bufferSize && "CudaSetReader: chunk_buffer.data not large enough.");
        
        // Update the CudaSet's metadata on the device
        chunk_buffer.numItems = numItems;
        chunk_buffer.totalElements = totalElements;

        // Copy the actual data to the device buffers
        if (totalElements > 0) {
            CHECK_CUDA_ERROR(cudaMemcpy(chunk_buffer.data, hostData.data(), totalElements * sizeof(int8_t), cudaMemcpyHostToDevice));
        }
        CHECK_CUDA_ERROR(cudaMemcpy(chunk_buffer.offsets, hostOffsets.data(), numItems * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(chunk_buffer.sizes, hostSizes.data(), numItems * sizeof(int), cudaMemcpyHostToDevice));

        if (verbose) {
            printf("CudaSetReader: Read chunk of %d vectors (%d total elements) from %s.\n", numItems, totalElements, filename.c_str());
        }

        return numItems;
    }

    bool isOpen() const {
        return file.is_open();
    }
};

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
    set.backing_file = ""; // Default to an in-memory set
    
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
    set->backing_file.clear();
}

// Helper function to generate a unique filename for a stream
std::string generateUniqueFilename(int level, int pair_index) {
    return "zdd_L" + std::to_string(level) + "_P" + std::to_string(pair_index) + ".bin";
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

// Helper function to check if a CudaSet is a dummy placeholder for streamed results
__host__ bool isDummySet(const CudaSet& set) {
    if (set.numItems != 1) {
        return false;
    }
    // Check for the magic number
    int flagValue;
    CHECK_CUDA_ERROR(cudaMemcpy(&flagValue, set.data, sizeof(int), cudaMemcpyDeviceToHost));
    return (flagValue == -999999);
}

// Helper function to append an in-memory CudaSet to the stream file
__host__ void streamCudaSet(const CudaSet& set, FILE* outFile, bool verbose) {
    if (verbose) {
        printf("  Appending in-memory set with %d items to stream...\n", set.numItems);
    }
    
    if (!outFile) {
        fprintf(stderr, "Error: Invalid file handle provided to streamCudaSet.\n");
        return;
    }

    // Copy the entire set to the host to read its contents
    HostSet hostSet = copyDeviceToHost(set);

    // Write each vector to the stream file
    for (const auto& vec : hostSet.vectors) {
        uint32_t vecSize = vec.size();
        fwrite(&vecSize, sizeof(uint32_t), 1, outFile);
        if (vecSize > 0) {
            fwrite(vec.data(), sizeof(int), vecSize, outFile);
        }
    }

    if (verbose) {
        printf("  Append operation complete.\n");
    }
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
CudaSet processPairCPU(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, const std::string& out_filename) {
    // Copy data to host
    HostSet hostSetA = copyDeviceToHost(setA);
    HostSet hostSetB = copyDeviceToHost(setB);
    
    // Open the output file in binary write mode to start fresh for this result set.
    FILE* outFile = fopen(out_filename.c_str(), "wb");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file %s for writing.\n", out_filename.c_str());
        // Return an empty set to signal failure
        return CudaSet{};
    }

    if (verbose) {
        printf("  Streaming CPU results directly to %s...\n", out_filename.c_str());
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
        printf("  CPU processing chunk complete. Results streamed to %s\n", out_filename.c_str());
    }
    
    // Return a lazy CudaSet that represents the file on disk
    CudaSet lazyResult = {};
    lazyResult.backing_file = out_filename;
    return lazyResult;
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

// Helper function to process two in-memory chunks and stream the results to a file
void processInMemoryChunksAndStream(const CudaSet& setA, const CudaSet& setB, int threshold, int level, FILE* outFile, bool verbose) {
    long long totalCombinations = (long long)setA.numItems * (long long)setB.numItems;
    if (totalCombinations == 0) return;

    CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(setA.numItems, setB.numItems, MAX_ELEMENTS_PER_VECTOR);
    
    int threadsPerBlock = 256;
    int maxResultsPerThread = 4;
    int threadsNeeded = (totalCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
    int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
    
    const int MAX_BLOCKS = 16384;
    if (blocksNeeded > MAX_BLOCKS) {
        blocksNeeded = MAX_BLOCKS;
        maxResultsPerThread = (totalCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
    }
    
    processAllCombinationsKernel<<<blocksNeeded, threadsPerBlock>>>(
        setA.data, setA.offsets, setA.sizes, setA.numItems,
        setB.data, setB.offsets, setB.sizes, setB.numItems,
        threshold, level,
        resultBuffer.data, resultBuffer.sizes, resultBuffer.validFlags, resultBuffer.maxResultSize,
        maxResultsPerThread
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<int> hostValidFlags(resultBuffer.numCombinations);
    CHECK_CUDA_ERROR(cudaMemcpy(hostValidFlags.data(), resultBuffer.validFlags, 
                              resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
    
    int validCount = 0;
    for (int flag : hostValidFlags) {
        if (flag) validCount++;
    }

    if (validCount > 0) {
        std::vector<int> hostSizes(resultBuffer.numCombinations);
        CHECK_CUDA_ERROR(cudaMemcpy(hostSizes.data(), resultBuffer.sizes, 
                                  resultBuffer.numCombinations * sizeof(int), cudaMemcpyDeviceToHost));
        
        std::vector<int> hostResultData(resultBuffer.numCombinations * resultBuffer.maxResultSize);
        CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
                                  resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < resultBuffer.numCombinations; i++) {
            if (hostValidFlags[i]) {
                uint32_t vecSize = hostSizes[i];
                int offset = i * resultBuffer.maxResultSize;
                
                // When processing chunks, don't do in-place deduplication.
                // Instead, write directly to the stream.
                fwrite(&vecSize, sizeof(uint32_t), 1, outFile);
                if (vecSize > 0) {
                    fwrite(hostResultData.data() + offset, sizeof(int), vecSize, outFile);
                }
            }
        }
    }
    
    freeCombinationResultBuffer(&resultBuffer);
}

CudaSet processLargePair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, const std::string& out_filename) {
    int numItemsA = setA.numItems;
    int numItemsB = setB.numItems;
    long long totalCombinations = (long long)numItemsA * (long long)numItemsB;
    
    if (verbose) {
        printf("  Processing large pair with true streaming: Set A (%d items) + Set B (%d items), threshold = %d\n", 
               numItemsA, numItemsB, threshold);
        printf("  Total combinations: %lld - results will be streamed to %s.\n", totalCombinations, out_filename.c_str());
    }

    // Open the output file in binary write mode to start fresh.
    FILE* outFile = fopen(out_filename.c_str(), "wb");
    if (!outFile) {
        fprintf(stderr, "Error: Could not open output file %s for writing.\n", out_filename.c_str());
        // Return an empty set to signal failure
        return CudaSet{};
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
            
            // Refactored to use the helper function
            processInMemoryChunksAndStream(tileSetA, tileSetB, threshold, level, outFile, false);
            
            freeCudaSet(&tileSetB);
        }
        
        freeCudaSet(&tileSetA);
    }
    
    fclose(outFile);
    
    if (verbose) {
        printf("  Large pair processing chunk complete. All results streamed to %s\n", out_filename.c_str());
    }
    
    // Return a lazy CudaSet that represents the file on disk
    CudaSet lazyResult = {};
    lazyResult.backing_file = out_filename;
    return lazyResult;
}

// Process a pair of sets using parallel kernel (maintains exact logic but parallelized)
// Process a pair of sets using batched approach for large sets
// Process a pair of sets using parallel kernel (simplified for correctness)
// Process a pair of sets using GPU-based batching
CudaSet processPair(const CudaSet& setA, const CudaSet& setB, int threshold, int level, bool verbose, int pair_index) {
    
    bool setA_is_lazy = !setA.backing_file.empty();
    bool setB_is_lazy = !setB.backing_file.empty();

    // Case 1: Both sets are in-memory
    if (!setA_is_lazy && !setB_is_lazy) {
        if (verbose) {
            printf("  Processing in-memory pair at level %d: Set A (%d items) + Set B (%d items), threshold = %d\n", 
                   level, setA.numItems, setB.numItems, threshold);
        }

        long long totalCombinations = (long long)setA.numItems * (long long)setB.numItems;
        std::string out_filename = generateUniqueFilename(level, pair_index);

        if (totalCombinations > 100000000LL) { // 100 million combinations threshold
            if (verbose) {
                printf("    Using CPU fallback for extremely large input (%lld combinations)\n", totalCombinations);
            }
            return processPairCPU(setA, setB, threshold, level, verbose, out_filename);
        }
        if (totalCombinations > 5000000LL) { // 5 million threshold
            return processLargePair(setA, setB, threshold, level, verbose, out_filename);
        }
        
        // --- Existing in-memory GPU processing logic ---
        int numItemsA = setA.numItems;
        int numItemsB = setB.numItems;
        CombinationResultBuffer resultBuffer = allocateCombinationResultBuffer(numItemsA, numItemsB, MAX_ELEMENTS_PER_VECTOR);
        
        int threadsPerBlock = 256;
        int maxResultsPerThread = 4;
        int threadsNeeded = (totalCombinations + maxResultsPerThread - 1) / maxResultsPerThread;
        int blocksNeeded = (threadsNeeded + threadsPerBlock - 1) / threadsPerBlock;
        
        const int MAX_BLOCKS = 16384;
        if (blocksNeeded > MAX_BLOCKS) {
            blocksNeeded = MAX_BLOCKS;
            maxResultsPerThread = (totalCombinations + (blocksNeeded * threadsPerBlock) - 1) / (blocksNeeded * threadsPerBlock);
            if (verbose) {
                printf("    Adjusting to %d blocks with %d results per thread\n", blocksNeeded, maxResultsPerThread);
            }
        }
        
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
            printf("    Found %d valid combinations out of %lld total\n", validCount, totalCombinations);
        }
		
        std::vector<std::vector<int>> validCombinations;
        if (validCount > 0) {
            if (verbose) {
                printf("    Copying result data for %d valid combinations...\n", validCount);
            }
            
            std::vector<int> hostResultData(resultBuffer.numCombinations * resultBuffer.maxResultSize);
            CHECK_CUDA_ERROR(cudaMemcpy(hostResultData.data(), resultBuffer.data,
                                      resultBuffer.numCombinations * resultBuffer.maxResultSize * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < resultBuffer.numCombinations; i++) {
                if (hostValidFlags[i]) {
                    int size = hostSizes[i];
                    std::vector<int> combination(size);
                    int offset = i * resultBuffer.maxResultSize;
                    
                    for (int j = 0; j < size; j++) {
                        combination[j] = hostResultData[offset + j];
                    }
                    
                    validCombinations.push_back(combination);
                }
            }
        }
        freeCombinationResultBuffer(&resultBuffer);

        if (verbose) {
            printf("    Removing duplicates from %zu combinations using sort-based approach...\n", validCombinations.size());
        }
        for (auto& combination : validCombinations) {
            std::sort(combination.begin(), combination.end());
        }
        if (verbose && validCombinations.size() > 10000) {
            printf("    Sorting %zu combinations...\n", validCombinations.size());
        }
        std::sort(validCombinations.begin(), validCombinations.end());
        
        std::vector<std::vector<int>> uniqueValidCombinations;
        uniqueValidCombinations.reserve(validCombinations.size());
        if (!validCombinations.empty()) {
            uniqueValidCombinations.push_back(validCombinations[0]);
            for (size_t i = 1; i < validCombinations.size(); i++) {
                if (validCombinations[i] != validCombinations[i-1]) {
                    uniqueValidCombinations.push_back(validCombinations[i]);
                }
            }
        }

        if (verbose) {
            printf("  Deduplication complete: %zu items, %zu unique (%.1f%% unique)\n", 
                   validCombinations.size(), uniqueValidCombinations.size(), 
                   validCombinations.size() > 0 ? 100.0 * uniqueValidCombinations.size() / validCombinations.size() : 0);
        }
        
        HostSet resultHostSet;
        resultHostSet.vectors = uniqueValidCombinations;
        
        CudaSet resultCudaSet;
        copyHostToDevice(resultHostSet, &resultCudaSet);
        
        return resultCudaSet;
    }
    
    // Case 2: Lazy vs Lazy
    if (setA_is_lazy && setB_is_lazy) {
        if (verbose) {
            printf("Processing lazy set '%s' vs lazy set '%s'.\n", setA.backing_file.c_str(), setB.backing_file.c_str());
        }

        // Define chunking parameters
        const int CHUNK_VECTORS = 1024;
        const int CHUNK_BUFFER_ELEMENTS = CHUNK_VECTORS * MAX_ELEMENTS_PER_VECTOR;

        // 1. Generate a unique filename for the output of this pair
        std::string out_filename = generateUniqueFilename(level, pair_index);
        FILE* outFile = fopen(out_filename.c_str(), "wb");
        if (!outFile) {
            fprintf(stderr, "Error: Could not open output file %s for writing in lazy-lazy case.\n", out_filename.c_str());
            return CudaSet{};
        }

        if (verbose) {
            printf("  Streaming lazy-vs-lazy results to %s\n", out_filename.c_str());
        }

        // 2. Allocate reusable chunk buffers for reading
        CudaSet chunkA = allocateCudaSet(CHUNK_VECTORS, CHUNK_BUFFER_ELEMENTS);
        CudaSet chunkB = allocateCudaSet(CHUNK_VECTORS, CHUNK_BUFFER_ELEMENTS);

        // 3. Loop through chunks of setA
        CudaSetReader readerA(setA.backing_file, verbose);
        while (readerA.readNextChunk(chunkA, CHUNK_VECTORS) > 0) {
            
            // 4. For each chunk of A, loop through all chunks of setB
            CudaSetReader readerB(setB.backing_file, verbose);
            while (readerB.readNextChunk(chunkB, CHUNK_VECTORS) > 0) {
                
                if (verbose) {
                    printf("    Processing chunk A (%d items) vs chunk B (%d items)\n", chunkA.numItems, chunkB.numItems);
                }
                processInMemoryChunksAndStream(chunkA, chunkB, threshold, level, outFile, false);
            }
        }

        // 6. Clean up
        fclose(outFile);
        freeCudaSet(&chunkA);
        freeCudaSet(&chunkB);

        // 7. Return a new lazy CudaSet pointing to the results file
        CudaSet lazyResult = {};
        lazyResult.backing_file = out_filename;
        if (verbose) {
            printf("  Finished streaming lazy-vs-lazy results to %s\n", out_filename.c_str());
        }
        return lazyResult;
    }

    // Case 3 & 4: Mixed mode (Lazy vs Real or Real vs Lazy)
    if (setA_is_lazy || setB_is_lazy) {
        if (verbose) {
            printf("Processing mixed mode - Lazy:'%s' vs Real:'%s'.\n", setA.backing_file.c_str(), setB.backing_file.c_str());
        }

        const CudaSet& lazySet = setA_is_lazy ? setA : setB;
        const CudaSet& realSet = setA_is_lazy ? setB : setA;

        // Define chunking parameters
        const int CHUNK_VECTORS = 1024;
        const int CHUNK_BUFFER_ELEMENTS = CHUNK_VECTORS * MAX_ELEMENTS_PER_VECTOR;

        // 1. Generate a unique filename for the output
        std::string out_filename = generateUniqueFilename(level, pair_index);
        FILE* outFile = fopen(out_filename.c_str(), "wb");
        if (!outFile) {
            fprintf(stderr, "Error: Could not open output file %s for writing in mixed mode.\n", out_filename.c_str());
            return CudaSet{};
        }
        
        if (verbose) {
            printf("  Streaming mixed-mode results to %s\n", out_filename.c_str());
        }

        // 2. Allocate a reusable chunk buffer for the lazy set
        CudaSet chunk = allocateCudaSet(CHUNK_VECTORS, CHUNK_BUFFER_ELEMENTS);

        // 3. Loop through chunks of the lazy set
        CudaSetReader reader(lazySet.backing_file, verbose);
        while (reader.readNextChunk(chunk, CHUNK_VECTORS) > 0) {
            if (verbose) {
                printf("    Processing chunk (%d items) vs real set (%d items)\n", chunk.numItems, realSet.numItems);
            }
            
            // 4. Process the chunk against the entire real set
            if (setA_is_lazy) {
                processInMemoryChunksAndStream(chunk, realSet, threshold, level, outFile, false);
            } else { // setB is lazy
                processInMemoryChunksAndStream(realSet, chunk, threshold, level, outFile, false);
            }
        }

        // 5. Clean up
        fclose(outFile);
        freeCudaSet(&chunk);

        // 6. Return a new lazy CudaSet pointing to the results file
        CudaSet lazyResult = {};
        lazyResult.backing_file = out_filename;
        if (verbose) {
            printf("  Finished streaming mixed-mode results to %s\n", out_filename.c_str());
        }
        return lazyResult;
    }

    // Should not be reached
    return CudaSet{};
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
            
            // Process the pair (parallelized internally), passing the pair index
            CudaSet resultSet = processPair(setA, setB, threshold, level, verbose, i / 2);
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
