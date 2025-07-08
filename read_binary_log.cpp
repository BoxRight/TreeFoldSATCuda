#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

// Function to check if a value is within the int8_t range
bool is_in_int8_range(int value) {
    return value >= INT8_MIN && value <= INT8_MAX;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file_path>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];
    std::ifstream inFile(filePath, std::ios::binary);

    if (!inFile) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return 1;
    }

    std::cout << "Reading data from: " << filePath << std::endl;

    int vectorCount = 0;
    while (inFile) {
        int vecSize = 0;
        // Read the size of the next vector
        inFile.read(reinterpret_cast<char*>(&vecSize), sizeof(int));

        if (inFile.eof()) {
            break; // End of file reached cleanly
        }

        if (vecSize < 0 || vecSize > 1024) { // Sanity check
             std::cerr << "Error: Invalid vector size read: " << vecSize << std::endl;
             break;
        }

        std::vector<int> tempVec(vecSize);
        // Read the vector data
        inFile.read(reinterpret_cast<char*>(tempVec.data()), vecSize * sizeof(int));
        
        if (inFile.fail()) {
            std::cerr << "Error reading vector data." << std::endl;
            break;
        }

        std::cout << "Vector " << vectorCount++ << " (size " << vecSize << "): [";
        bool first = true;
        for (int val : tempVec) {
            if (!first) {
                std::cout << ", ";
            }
            std::cout << val;
            if (!is_in_int8_range(val)) {
                std::cout << " (!!! OUT OF RANGE !!!)";
            }
            first = false;
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "\nFinished reading file." << std::endl;

    return 0;
} 