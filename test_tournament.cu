#include "tournament_tree.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <set>
#include <algorithm>

void testSATClauses() {
    printf("=== Running SAT Tournament with 51 Clauses ===\n\n");
    
    // Test Set 0 (clause 'a')
    std::vector<std::vector<int8_t>> set0 = {
        {-1,-3,-2}, {-1,-3,2}, {-1,3,-2}, {-1,3,2}, {1,-3,-2}, {1,-3,2}, {1,3,2}
    };
    
    // Test Set 1 (clause 'b')
    std::vector<std::vector<int8_t>> set1 = {
        {-1,-5,-4}, {-1,-5,4}, {-1,5,-4}, {-1,5,4}, {1,-5,-4}, {1,-5,4}, {1,5,4}
    };
    
    // Test Set 2 (clause 'c')
    std::vector<std::vector<int8_t>> set2 = {
        {1,6}, {-1,6}, {-1,-6}
    };
    
    // Test Set 3 (clause 'd')
    std::vector<std::vector<int8_t>> set3 = {
        {-1,-4,-7}, {-1,-4,7}, {-1,4,-7}, {-1,4,7}, {1,-4,-7}, {1,-4,7}, {1,4,7}
    };
    
    // Test Set 4 (clause 'e')
    std::vector<std::vector<int8_t>> set4 = {
        {1,8}, {-1,8}, {-1,-8}
    };
    
    // Test Set 5 (clause 'f')
    std::vector<std::vector<int8_t>> set5 = {
        {-1,-4,-9}, {-1,-4,9}, {-1,4,-9}, {-1,4,9}, {1,-4,-9}, {1,-4,9}, {1,4,9}
    };
    
    // Test Set 6 (clause 'g')
    std::vector<std::vector<int8_t>> set6 = {
        {1,10}, {-1,10}, {-1,-10}
    };
    
    // Test Set 7 (clause 'h')
    std::vector<std::vector<int8_t>> set7 = {
        {1,11}, {-1,11}, {-1,-11}
    };
    
    // Test Set 8 (clause 'i')
    std::vector<std::vector<int8_t>> set8 = {
        {-1,-4,-12}, {-1,-4,12}, {-1,4,-12}, {-1,4,12}, {1,-4,-12}, {1,-4,12}, {1,4,12}
    };
    
    // Test Set 9 (clause 'j')
    std::vector<std::vector<int8_t>> set9 = {
        {1,13}, {-1,13}, {-1,-13}
    };
    
    // Test Set 10 (clause 'k')
    std::vector<std::vector<int8_t>> set10 = {
        {1,14}, {-1,14}, {-1,-14}
    };
    
    // Test Set 11 (clause 'l')
    std::vector<std::vector<int8_t>> set11 = {
        {1,-15}, {-1,15}, {-1,-15}
    };
    
    // Test Set 12 (clause 'm')
    std::vector<std::vector<int8_t>> set12 = {
        {1,16}, {-1,16}, {-1,-16}
    };
    
    // Test Set 13 (clause 'n')
    std::vector<std::vector<int8_t>> set13 = {
        {1,17}, {-1,17}, {-1,-17}
    };
    
    // Test Set 14 (clause 'o')
    std::vector<std::vector<int8_t>> set14 = {
        {1,18}, {-1,18}, {-1,-18}
    };
    
    // Test Set 15 (clause 'p')
    std::vector<std::vector<int8_t>> set15 = {
        {1,19}, {-1,19}, {-1,-19}
    };
    
    // Test Set 16 (clause 'q')
    std::vector<std::vector<int8_t>> set16 = {
        {-1,-21,-20}, {-1,-21,20}, {-1,21,-20}, {-1,21,20}, {1,-21,-20}, {1,-21,20}, {1,21,20}
    };
    
    // Test Set 17 (clause 'r')
    std::vector<std::vector<int8_t>> set17 = {
        {-1,-20,-22}, {-1,-20,22}, {-1,20,-22}, {-1,20,22}, {1,-20,-22}, {1,-20,22}, {1,20,22}
    };
    
    // Test Set 18 (clause 's')
    std::vector<std::vector<int8_t>> set18 = {
        {-1,-24,-23}, {-1,-24,23}, {-1,24,-23}, {-1,24,23}, {1,-24,-23}, {1,-24,23}, {1,24,23}
    };
    
    // Test Set 19 (clause 't')
    std::vector<std::vector<int8_t>> set19 = {
        {1,25}, {-1,25}, {-1,-25}
    };
    
    // Test Set 20 (clause 'u')
    std::vector<std::vector<int8_t>> set20 = {
        {-1,-4,-26}, {-1,-4,26}, {-1,4,-26}, {-1,4,26}, {1,-4,-26}, {1,-4,26}, {1,4,26}
    };
    
    // Test Set 21 (clause 'v')
    std::vector<std::vector<int8_t>> set21 = {
        {1,27}, {-1,27}, {-1,-27}
    };
    
    // Test Set 22 (clause 'w')
    std::vector<std::vector<int8_t>> set22 = {
        {1,27}, {-1,27}, {-1,-27}
    };
    
    // Test Set 23 (clause 'x')
    std::vector<std::vector<int8_t>> set23 = {
        {1,29}, {-1,29}, {-1,-29}
    };
    
    // Test Set 24 (clause 'y')
    std::vector<std::vector<int8_t>> set24 = {
        {1,29}, {-1,29}, {-1,-29}
    };
    
    // Test Set 25 (clause 'z')
    std::vector<std::vector<int8_t>> set25 = {
        {1,31}, {-1,31}, {-1,-31}
    };
    
    // Test Set 26 (clause '{')
    std::vector<std::vector<int8_t>> set26 = {
        {1,31}, {-1,31}, {-1,-31}
    };
    
    // Test Set 27 (clause '|')
    std::vector<std::vector<int8_t>> set27 = {
        {-1,-34,-33}, {-1,-34,33}, {-1,34,-33}, {-1,34,33}, {1,-34,-33}, {1,-34,33}, {1,34,33}
    };
    
    // Test Set 28 (clause '}')
    std::vector<std::vector<int8_t>> set28 = {
        {-1,-36,-35}, {-1,-36,35}, {-1,36,-35}, {-1,36,35}, {1,-36,-35}, {1,-36,35}, {1,36,35}
    };
    
    // Test Set 29 (clause '~')
    std::vector<std::vector<int8_t>> set29 = {
        {-1,-38,-37}, {-1,-38,37}, {-1,38,-37}, {-1,38,37}, {1,-38,-37}, {1,-38,37}, {1,38,37}
    };
    
    // Test Set 30 (clause '')
    std::vector<std::vector<int8_t>> set30 = {
        {-1,-40,-39}, {-1,-40,39}, {-1,40,-39}, {-1,40,39}, {1,-40,-39}, {1,-40,39}, {1,40,39}
    };
    
    // Test Set 31 (clause '')
    std::vector<std::vector<int8_t>> set31 = {
        {-1,-42,-41}, {-1,-42,41}, {-1,42,-41}, {-1,42,41}, {1,-42,-41}, {1,-42,41}, {1,42,41}
    };
    
    // Test Set 32 (clause '')
    std::vector<std::vector<int8_t>> set32 = {
        {-1,-44,-43}, {-1,-44,43}, {-1,44,-43}, {-1,44,43}, {1,-44,-43}, {1,-44,43}, {1,44,43}
    };
    
    // Test Set 33 (clause '')
    std::vector<std::vector<int8_t>> set33 = {
        {-1,-46,-45}, {-1,-46,45}, {-1,46,-45}, {-1,46,45}, {1,-46,-45}, {1,-46,45}, {1,46,45}
    };
    
    // Test Set 34 (clause '')
    std::vector<std::vector<int8_t>> set34 = {
        {-1,-48,-47}, {-1,-48,47}, {-1,48,-47}, {-1,48,47}, {1,-48,-47}, {1,-48,47}, {1,48,47}
    };
    
    // Test Set 35 (clause '')
    std::vector<std::vector<int8_t>> set35 = {
        {-1,-48,-47}, {-1,-48,47}, {-1,48,-47}, {-1,48,47}, {1,-48,-47}, {1,-48,47}, {1,48,47}
    };
    
    // Test Set 36 (clause '')
    std::vector<std::vector<int8_t>> set36 = {
        {1,51}, {-1,51}, {-1,-51}
    };
    
    // Test Set 37 (clause '')
    std::vector<std::vector<int8_t>> set37 = {
        {1,51}, {-1,51}, {-1,-51}
    };
    
    // Test Set 38 (clause '')
    std::vector<std::vector<int8_t>> set38 = {
        {1,53}, {-1,53}, {-1,-53}
    };
    
    // Test Set 39 (clause '')
    std::vector<std::vector<int8_t>> set39 = {
        {1,54}, {-1,54}, {-1,-54}
    };
    
    // Test Set 40 (clause '')
    std::vector<std::vector<int8_t>> set40 = {
        {1,55}, {-1,55}, {-1,-55}
    };
    
    // Test Set 41 (clause '')
    std::vector<std::vector<int8_t>> set41 = {
        {1,56}, {-1,56}, {-1,-56}
    };
    
    // Test Set 42 (clause '')
    std::vector<std::vector<int8_t>> set42 = {
        {1,56}, {-1,56}, {-1,-56}
    };
    
    // Test Set 43 (clause '')
    std::vector<std::vector<int8_t>> set43 = {
        {1,58}, {-1,58}, {-1,-58}
    };
    
    // Test Set 44 (clause '')
    std::vector<std::vector<int8_t>> set44 = {
        {1,58}, {-1,58}, {-1,-58}
    };
    
    // Test Set 45 (clause '')
    std::vector<std::vector<int8_t>> set45 = {
        {-1,-61,-60}, {-1,-61,60}, {-1,61,-60}, {-1,61,60}, {1,-61,-60}, {1,-61,60}, {1,61,60}
    };
    
    // Test Set 46 (clause '')
    std::vector<std::vector<int8_t>> set46 = {
        {1,62}, {-1,62}, {-1,-62}
    };
    
    // Test Set 47 (clause '')
    std::vector<std::vector<int8_t>> set47 = {
        {-1,-64,-63}, {-1,-64,63}, {-1,64,-63}, {-1,64,63}, {1,-64,-63}, {1,-64,63}, {1,64,63}
    };
    
    // Test Set 48 (clause '')
    std::vector<std::vector<int8_t>> set48 = {
        {-1,-66,-65}, {-1,-66,65}, {-1,66,-65}, {-1,66,65}, {1,-66,-65}, {1,-66,65}, {1,66,65}
    };
    
    // Test Set 49 (clause '')
    std::vector<std::vector<int8_t>> set49 = {
        {-1,-68,-67}, {-1,-68,67}, {-1,68,-67}, {-1,68,67}, {1,-68,-67}, {1,-68,67}, {1,68,67}
    };
    
    // Test Set 50 (clause '')
    std::vector<std::vector<int8_t>> set50 = {
        {-1,-70,-69}, {-1,-70,69}, {-1,70,-69}, {-1,70,69}, {1,-70,-69}, {1,-70,69}, {1,70,69}
    };
    
    // Collect all sets
    std::vector<std::vector<std::vector<int8_t>>> allSets = {
        set0, set1, set2, set3, set4, set5, set6, set7, set8, set9, set10,
        set11, set12, set13, set14, set15, set16, set17, set18, set19, set20,
        set21, set22, set23, set24, set25, set26, set27, set28, set29, set30,
        set31, set32, set33, set34, set35, set36, set37, set38, set39, set40,
        set41, set42, set43, set44, set45, set46, set47, set48, set49, set50
    };
    
    printf("Testing with %zu SAT clauses:\n", allSets.size());
    for (size_t i = 0; i < allSets.size(); i++) {
        printf("  Clause %zu: %zu vectors\n", i, allSets[i].size());
    }
    
    // Run tournament
    printf("\nInitializing tournament...\n");
    TournamentTree tournament;
    if (tournament.initialize(allSets)) {
        printf("Tournament initialized successfully. Starting computation...\n");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (tournament.runTournament()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            printf("Tournament completed in %ld ms\n", duration.count());
            
            std::vector<int8_t> result = tournament.getFinalResult();
            printf("\nFinal result size: %zu\n", result.size());
            if (result.size() > 0) {
                printf("Final result: ");
                for (int8_t val : result) {
                    printf("%d ", val);
                }
                printf("\n");
            }
            
            std::vector<std::vector<int8_t>> allResults = tournament.getAllFinalResults();
            printf("All final results: %zu\n", allResults.size());
            for (size_t i = 0; i < std::min(size_t(20), allResults.size()); i++) {
                printf("  Result %zu: ", i);
                for (int8_t val : allResults[i]) {
                    printf("%d ", val);
                }
                printf("\n");
            }
            if (allResults.size() > 20) {
                printf("  ... and %zu more results\n", allResults.size() - 20);
            }
        } else {
            printf("Tournament failed to run\n");
        }
    } else {
        printf("Failed to initialize tournament\n");
    }
    
    printf("\n=== SAT Tournament Complete ===\n");
}

int main() {
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found or CUDA not available" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    // Set device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Run only the SAT test
    testSATClauses();
    
    std::cout << "\n=== SAT Test Completed ===" << std::endl;
    return 0;
} 