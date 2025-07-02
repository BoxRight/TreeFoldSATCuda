#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Struct used during construction
typedef struct {
    uint16_t variable;
    uint32_t thenNode;
    uint32_t elseNode;
    uint32_t id;
} BuildNode;

// Simple hash table entry
typedef struct {
    uint16_t variable;
    uint32_t thenNode;
    uint32_t elseNode;
    uint32_t nodeId;
    uint8_t valid;
} NodeTableEntry;

// Constants
#define MAX_VARIABLES 32          // Maximum number of variables in vectors
#define MAX_VECTOR_SIZE 32        // Maximum elements per vector
#define TERMINAL_ZERO 0           // ID for terminal zero node
#define TERMINAL_ONE 1            // ID for terminal one node
#define MAX_FILE_LINE 1024        // Maximum length of a line in the input file
#define BATCH_SIZE 5000           // Batch size for processing
#define NODE_TABLE_SIZE 5000000   // Size of node uniqueness table (larger for better sharing)

// Global variables for node table
NodeTableEntry* nodeTable = NULL;
BuildNode* buildNodes = NULL;
uint32_t nodeCount = 0;

// Hash function for node table
uint32_t nodeHash(uint16_t variable, uint32_t thenNode, uint32_t elseNode) {
    return ((uint32_t)variable * 73856093U + 
            (uint32_t)thenNode * 19349663U + 
            (uint32_t)elseNode * 83492791U) % NODE_TABLE_SIZE;
}

// Initialize node table
int initNodeTable() {
    if (nodeTable != NULL) {
        free(nodeTable);
    }
    
    nodeTable = (NodeTableEntry*)calloc(NODE_TABLE_SIZE, sizeof(NodeTableEntry));
    if (!nodeTable) {
        return 0; // Failed to allocate
    }
    
    if (buildNodes != NULL) {
        free(buildNodes);
    }
    
    buildNodes = (BuildNode*)malloc(NODE_TABLE_SIZE * sizeof(BuildNode));
    if (!buildNodes) {
        free(nodeTable);
        nodeTable = NULL;
        return 0; // Failed to allocate
    }
    
    nodeCount = 0;
    return 1; // Success
}

// Find or create a node with deduplication
uint32_t getNode(uint16_t variable, uint32_t thenNode, uint32_t elseNode) {
    // Apply zero-suppression rule
    if (thenNode == TERMINAL_ZERO) {
        return elseNode;
    }
    
    // Hash-based lookup for node reuse
    uint32_t h = nodeHash(variable, thenNode, elseNode);
    uint32_t step = 1;
    uint32_t idx = h;
    
    // Linear probing to find existing or empty slot
    while (nodeTable[idx].valid) {
        if (nodeTable[idx].variable == variable &&
            nodeTable[idx].thenNode == thenNode &&
            nodeTable[idx].elseNode == elseNode) {
            // Node already exists
            return nodeTable[idx].nodeId;
        }
        
        // Try next slot
        idx = (idx + step) % NODE_TABLE_SIZE;
        if (idx == h) {
            // Wrapped around - table is full
            fprintf(stderr, "Node table full, some nodes won't be shared\n");
            // Just create a new node without adding to table
            break;
        }
    }
    
    // Create new node
    uint32_t newId = nodeCount + 2; // +2 for terminal nodes
    
    // Safety check to prevent buffer overflow
    if (nodeCount >= NODE_TABLE_SIZE - 3) {
        fprintf(stderr, "Warning: Node limit reached. Using terminal node.\n");
        return TERMINAL_ONE; // Fallback
    }
    
    // Store the new node
    buildNodes[nodeCount].variable = variable;
    buildNodes[nodeCount].thenNode = thenNode;
    buildNodes[nodeCount].elseNode = elseNode;
    buildNodes[nodeCount].id = newId;
    
    // Add to node table for lookup
    nodeTable[idx].variable = variable;
    nodeTable[idx].thenNode = thenNode;
    nodeTable[idx].elseNode = elseNode;
    nodeTable[idx].nodeId = newId;
    nodeTable[idx].valid = 1;
    
    nodeCount++;
    return newId;
}

// Parse a vector from a string
int parseVector(char* line, uint16_t* vector) {
    int count = 0;
    char* token;
    
    // Skip leading bracket
    if (line[0] == '[') {
        line++;
    }
    
    // Parse comma-separated values
    token = strtok(line, ",]");
    while (token != NULL && count < MAX_VECTOR_SIZE) {
        int val = atoi(token);
        if (val > 0 && val <= UINT16_MAX) {
            vector[count++] = (uint16_t)val;
        }
        token = strtok(NULL, ",]");
    }
    
    return count;
}

// Build a ZDD for a single vector
uint32_t buildZDDForVector(uint16_t* vector, int vectorSize, uint16_t maxVar) {
    // Array showing which variables are in the vector
    uint8_t* inVector = (uint8_t*)calloc(maxVar + 1, sizeof(uint8_t));
    if (!inVector) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return TERMINAL_ONE;
    }
    
    for (int i = 0; i < vectorSize; i++) {
        if (vector[i] >= 1 && vector[i] <= maxVar) {
            inVector[vector[i]] = 1;
        }
    }
    
    // Bottom-up construction (avoids recursion)
    uint32_t* results = (uint32_t*)malloc((maxVar + 2) * sizeof(uint32_t));
    if (!results) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(inVector);
        return TERMINAL_ONE;
    }
    
    results[maxVar + 1] = TERMINAL_ONE; // Base case
    
    for (int var = maxVar; var >= 1; var--) {
        if (inVector[var]) {
            // Variable is in the set
            results[var] = getNode(var, results[var + 1], TERMINAL_ZERO);
        } else {
            // Variable not in the set
            results[var] = results[var + 1];
        }
    }
    
    uint32_t rootNodeId = results[1];
    free(results);
    free(inVector);
    
    return rootNodeId;
}

// Main program
int main() {
    // Hardcoded filenames
    const char* inputFile = "large_result.txt";
    const char* outputFile = "output.json";
    
    printf("Reading vectors from %s\n", inputFile);
    
    FILE* file = fopen(inputFile, "r");
    if (!file) {
        fprintf(stderr, "Error: Could not open input file\n");
        return 1;
    }
    
    // First pass: count vectors and find max variable
    uint16_t maxVar = 0;
    long long totalVectors = 0;
    char line[MAX_FILE_LINE];
    
    printf("First pass: Counting vectors and finding max variable...\n");
    
    // Skip header if present
    if (fgets(line, sizeof(line), file) && strstr(line, "Final results")) {
        if (sscanf(line, "# Final results: %lld vectors", &totalVectors) == 1) {
            printf("Found vector count in header: %lld\n", totalVectors);
        }
    } else {
        rewind(file);
    }
    
    // Count vectors and find max variable
    uint16_t* sampleVector = (uint16_t*)malloc(MAX_VECTOR_SIZE * sizeof(uint16_t));
    
    while (fgets(line, sizeof(line), file)) {
        // Skip comment lines and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        // Count vector
        if (totalVectors == 0 || totalVectors > 1000000000) { // If not from header or unreasonable
            totalVectors++;
        }
        
        // Parse vector to find max variable
        int size = parseVector(line, sampleVector);
        
        for (int i = 0; i < size; i++) {
            if (sampleVector[i] > maxVar) {
                maxVar = sampleVector[i];
            }
        }
        
        // Just sample first 10,000 vectors for max variable
        if (totalVectors >= 10000) {
            break;
        }
    }
    
    free(sampleVector);
    
    printf("Estimated total vectors: %lld, Maximum variable: %d\n", totalVectors, maxVar);
    
    // Disable common prefix optimization
    uint16_t* commonPrefix = (uint16_t*)calloc(maxVar + 1, sizeof(uint16_t));
    int commonPrefixSize = 0;
    
    printf("Common prefix optimization disabled (commonPrefixSize = 0)\n");
    
    // Create output file
    FILE* outFile = fopen(outputFile, "w");
    if (!outFile) {
        fprintf(stderr, "Error: Could not create output file\n");
        fclose(file);
        free(commonPrefix);
        return 1;
    }
    
    // Start JSON output
    fprintf(outFile, "{\n");
    fprintf(outFile, "  \"vectorCount\": %lld,\n", totalVectors);
    fprintf(outFile, "  \"maxVar\": %d,\n", maxVar);
    fprintf(outFile, "  \"commonPrefix\": [");
    
    int prefixCount = 0;
    for (int i = 1; i <= maxVar; i++) {
        if (commonPrefix[i]) {
            if (prefixCount > 0) {
                fprintf(outFile, ", ");
            }
            fprintf(outFile, "%d", i);
            prefixCount++;
        }
    }
    fprintf(outFile, "],\n");
    fprintf(outFile, "  \"batches\": [\n");
    
    // Process file in batches
    printf("Second pass: Building ZDD with global node sharing...\n");
    
    if (!initNodeTable()) {
        fprintf(stderr, "Error: Failed to initialize node table\n");
        fclose(file);
        fclose(outFile);
        free(commonPrefix);
        return 1;
    }
    
    rewind(file);
    
    // Skip header if present
    if (fgets(line, sizeof(line), file) && strstr(line, "Final results")) {
        // Skipped header
    } else {
        rewind(file);
    }
    
    int batchNum = 0;
    int totalBatches = (totalVectors + BATCH_SIZE - 1) / BATCH_SIZE;
    long long processedVectors = 0;
    time_t startTime = time(NULL);
    
    // Allocate vector storage
    uint16_t** vectors = (uint16_t**)malloc(BATCH_SIZE * sizeof(uint16_t*));
    int* vectorSizes = (int*)malloc(BATCH_SIZE * sizeof(int));
    uint32_t* rootNodes = (uint32_t*)malloc(BATCH_SIZE * sizeof(uint32_t));
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        vectors[i] = (uint16_t*)malloc(MAX_VECTOR_SIZE * sizeof(uint16_t));
    }
    
    // Optional: limit the number of batches for testing
    int maxBatchesToProcess = 2; // Set to a large number for full processing
    
    while (!feof(file) && batchNum < maxBatchesToProcess) {
        // Reset for this batch
        int vectorCount = 0;
        nodeCount = 0;
        
        // Read a batch of vectors
        while (vectorCount < BATCH_SIZE && fgets(line, sizeof(line), file)) {
            // Skip comment lines and empty lines
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
                continue;
            }
            
            // Parse vector
            vectorSizes[vectorCount] = parseVector(line, vectors[vectorCount]);
            vectorCount++;
        }
        
        if (vectorCount == 0) {
            break;  // No more vectors
        }
        
        batchNum++;
        printf("Processing batch %d/%d with %d vectors\n", batchNum, totalBatches, vectorCount);
        
        // First build ZDDs for all vectors in this batch
        for (int i = 0; i < vectorCount; i++) {
            rootNodes[i] = buildZDDForVector(vectors[i], vectorSizes[i], maxVar);
            
            // Progress indication for large batches
            if ((i+1) % (BATCH_SIZE / 5) == 0 || i == vectorCount-1) {
                printf("  Processed %d/%d vectors in current batch\n", i+1, vectorCount);
            }
        }
        
        // Write batch as JSON
        fprintf(outFile, "    {\n");
        fprintf(outFile, "      \"vectorCount\": %d,\n", vectorCount);
        fprintf(outFile, "      \"nodeCount\": %u,\n", nodeCount);
        
        // Write root nodes
        fprintf(outFile, "      \"rootNodes\": [");
        for (int i = 0; i < vectorCount; i++) {
            fprintf(outFile, "%s%u", (i > 0 ? ", " : ""), rootNodes[i]);
        }
        fprintf(outFile, "],\n");
        
        // Write nodes
        fprintf(outFile, "      \"nodes\": [\n");
        for (uint32_t i = 0; i < nodeCount; i++) {
            fprintf(outFile, "        {\"id\": %u, \"var\": %u, \"then\": %u, \"else\": %u}%s\n",
                    buildNodes[i].id, buildNodes[i].variable, 
                    buildNodes[i].thenNode, buildNodes[i].elseNode,
                    (i < nodeCount - 1) ? "," : "");
        }
        fprintf(outFile, "      ]\n");
        fprintf(outFile, "    }%s\n", (batchNum < maxBatchesToProcess && !feof(file)) ? "," : "");
        
        // Update progress
        processedVectors += vectorCount;
        
        // Calculate and display progress
        double progress = (double)processedVectors / totalVectors * 100.0;
        time_t currentTime = time(NULL);
        double elapsed = difftime(currentTime, startTime);
        
        if (elapsed > 0) {
            double vectorsPerSecond = processedVectors / elapsed;
            double remainingSeconds = (totalVectors - processedVectors) / vectorsPerSecond;
            
            int remainingHours = (int)(remainingSeconds / 3600);
            int remainingMinutes = (int)((remainingSeconds - remainingHours * 3600) / 60);
            int remainingSecs = (int)(remainingSeconds - remainingHours * 3600 - remainingMinutes * 60);
            
            printf("Processed %lld/%lld vectors (%.2f%%) - %.0f vectors/sec, ETA: %02d:%02d:%02d\n", 
                   processedVectors, totalVectors, progress, vectorsPerSecond, 
                   remainingHours, remainingMinutes, remainingSecs);
        }
    }
    
    // Close JSON structure
    fprintf(outFile, "  ]\n");
    fprintf(outFile, "}\n");
    
    // Clean up
    for (int i = 0; i < BATCH_SIZE; i++) {
        free(vectors[i]);
    }
    free(vectors);
    free(vectorSizes);
    free(rootNodes);
    free(commonPrefix);
    
    if (nodeTable) {
        free(nodeTable);
        nodeTable = NULL;
    }
    
    if (buildNodes) {
        free(buildNodes);
        buildNodes = NULL;
    }
    
    fclose(file);
    fclose(outFile);
    
    // Calculate total time
    time_t totalTime = time(NULL) - startTime;
    printf("ZDD construction complete in %02d:%02d:%02d\n", 
           (int)totalTime / 3600, ((int)totalTime % 3600) / 60, (int)totalTime % 60);
           
    // Calculate file sizes
    FILE* inFileStat = fopen(inputFile, "rb");
    FILE* outFileStat = fopen(outputFile, "rb");
    
    if (inFileStat && outFileStat) {
        fseek(inFileStat, 0, SEEK_END);
        fseek(outFileStat, 0, SEEK_END);
        
        long inSize = ftell(inFileStat);
        long outSize = ftell(outFileStat);
        
        fclose(inFileStat);
        fclose(outFileStat);
        
        printf("Input file size: %.2f MB\n", inSize / (1024.0 * 1024.0));
        printf("Output file size: %.2f MB\n", outSize / (1024.0 * 1024.0));
        printf("Compression ratio: %.2f:1\n", (double)inSize / outSize);
    }
    
    printf("Results saved to %s\n", outputFile);
    
    return 0;
}
