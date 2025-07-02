#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Simple JSON parser state
typedef struct {
    char* buffer;
    size_t pos;
    size_t size;
} Parser;

// Node structure
typedef struct {
    uint32_t id;
    uint16_t variable;
    uint32_t thenNode;
    uint32_t elseNode;
} Node;

// Constants
#define TERMINAL_ZERO 0
#define TERMINAL_ONE 1
#define MAX_NODES_TO_CHECK 1000

// Skip whitespace in parser
void skipWhitespace(Parser* p) {
    while (p->pos < p->size && (p->buffer[p->pos] == ' ' || p->buffer[p->pos] == '\n' || 
           p->buffer[p->pos] == '\r' || p->buffer[p->pos] == '\t')) {
        p->pos++;
    }
}

// Find string in parser
char* findString(Parser* p, const char* str) {
    char* found = strstr(p->buffer + p->pos, str);
    if (found) {
        p->pos = found - p->buffer + strlen(str);
    }
    return found;
}

// Extract a number from parser
uint32_t extractNumber(Parser* p) {
    skipWhitespace(p);
    
    // Find first digit
    while (p->pos < p->size && (p->buffer[p->pos] < '0' || p->buffer[p->pos] > '9')) {
        p->pos++;
    }
    
    // Extract number
    uint32_t num = 0;
    while (p->pos < p->size && p->buffer[p->pos] >= '0' && p->buffer[p->pos] <= '9') {
        num = num * 10 + (p->buffer[p->pos] - '0');
        p->pos++;
    }
    
    return num;
}

// Load file into memory
char* loadFile(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Allocate buffer
    char* buffer = (char*)malloc(*size + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }
    
    // Read file
    fread(buffer, 1, *size, file);
    buffer[*size] = '\0';
    
    fclose(file);
    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <json_zdd_file>\n", argv[0]);
        return 1;
    }
    
    // Load file
    size_t size;
    char* buffer = loadFile(argv[1], &size);
    if (!buffer) {
        printf("Error: Could not load file %s\n", argv[1]);
        return 1;
    }
    
    // Initialize parser
    Parser parser = {buffer, 0, size};
    
    // Find metadata
    uint32_t vectorCount = 0;
    uint32_t maxVar = 0;
    
    if (findString(&parser, "\"vectorCount\":")) {
        vectorCount = extractNumber(&parser);
    }
    
    parser.pos = 0;
    if (findString(&parser, "\"maxVar\":")) {
        maxVar = extractNumber(&parser);
    }
    
    printf("ZDD Metadata: %u vectors, max variable %u\n", vectorCount, maxVar);
    
    // Find first batch
    parser.pos = 0;
    if (!findString(&parser, "\"batches\":")) {
        printf("Error: Could not find batches in file\n");
        free(buffer);
        return 1;
    }
    
    if (!findString(&parser, "\"nodes\":")) {
        printf("Error: Could not find nodes in batch\n");
        free(buffer);
        return 1;
    }
    
    // Count nodes and check their structure
    uint32_t nodesChecked = 0;
    uint32_t validNodes = 0;
    uint32_t suspiciousNodes = 0;
    
    uint32_t nodeId, variable, thenNode, elseNode;
    
    while (findString(&parser, "{\"id\":") && nodesChecked < MAX_NODES_TO_CHECK) {
        // Extract node ID
        nodeId = extractNumber(&parser);
        
        // Extract variable
        if (!findString(&parser, "\"var\":")) break;
        variable = extractNumber(&parser);
        
        // Extract then node
        if (!findString(&parser, "\"then\":")) break;
        thenNode = extractNumber(&parser);
        
        // Extract else node
        if (!findString(&parser, "\"else\":")) break;
        elseNode = extractNumber(&parser);
        
        nodesChecked++;
        
        // Check node validity
        if (variable == 0 || variable > maxVar) {
            printf("Error: Node %u has invalid variable %u\n", nodeId, variable);
            suspiciousNodes++;
        } else if (thenNode != TERMINAL_ZERO && thenNode != TERMINAL_ONE && thenNode < 2) {
            printf("Error: Node %u has invalid then node %u\n", nodeId, thenNode);
            suspiciousNodes++;
        } else if (elseNode != TERMINAL_ZERO && elseNode != TERMINAL_ONE && elseNode < 2) {
            printf("Error: Node %u has invalid else node %u\n", nodeId, elseNode);
            suspiciousNodes++;
        } else if (thenNode == TERMINAL_ZERO) {
            printf("Warning: Node %u violates zero-suppression rule (then=0)\n", nodeId);
            suspiciousNodes++;
        } else {
            validNodes++;
        }
        
        // Print some nodes for inspection
        if (nodesChecked <= 10) {
            printf("Node %u: var=%u, then=%u, else=%u\n", 
                   nodeId, variable, thenNode, elseNode);
        }
    }
    
    printf("\nChecked %u nodes: %u valid, %u suspicious\n", 
           nodesChecked, validNodes, suspiciousNodes);
    
    free(buffer);
    return 0;
}
