// read_zdd.cpp
//
// A simple utility to read and inspect the contents of the zdd.bin file.
// It streams the data to avoid high memory usage and prints a limited
// number of vectors to the console.
//
// How to compile and run:
//   g++ -std=c++17 read_zdd.cpp -o read_zdd
//   ./read_zdd
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

void inspect_binary_file(const char* filename, int vectors_to_print = 100) {
    // Open the file in binary read mode
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::cout << "Inspecting contents of " << filename << "...\n" << std::endl;

    int vectors_read = 0;
    while (vectors_read < vectors_to_print && !file.eof()) {
        uint32_t vec_size;

        // Read the 32-bit size of the next vector
        file.read(reinterpret_cast<char*>(&vec_size), sizeof(uint32_t));

        // Check if the read was successful. If we reached the end of the file, break.
        if (file.gcount() == 0) {
            break;
        }

        if (file.gcount() != sizeof(uint32_t)) {
            std::cerr << "Error: Corrupted file or incomplete read while fetching vector size." << std::endl;
            break;
        }
        
        // Read the vector data
        std::vector<int> vec_data(vec_size);
        if (vec_size > 0) {
            file.read(reinterpret_cast<char*>(vec_data.data()), vec_size * sizeof(int));

            if (file.gcount() != vec_size * sizeof(int)) {
                std::cerr << "Error: Corrupted file or incomplete read while fetching vector data." << std::endl;
                break;
            }
        }

        // Print the vector to the console
        std::cout << "Vector " << (vectors_read + 1) << " (size: " << vec_size << "): [";
        for (size_t i = 0; i < vec_data.size(); ++i) {
            std::cout << vec_data[i] << (i == vec_data.size() - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        vectors_read++;
    }

    if (vectors_read == 0) {
        std::cout << "File appears to be empty or could not be read." << std::endl;
    } else if (vectors_read == vectors_to_print && !file.eof()) {
        std::cout << "\n...and possibly many more vectors." << std::endl;
    } else {
        std::cout << "\nEnd of file reached. Total vectors read: " << vectors_read << std::endl;
    }

    file.close();
}

int main() {
    inspect_binary_file("zdd.bin");
    return 0;
} 