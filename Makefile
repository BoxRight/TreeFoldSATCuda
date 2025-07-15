# CUDA Tournament Tree Makefile

# Compiler settings
NVCC = nvcc
CXX = g++

# CUDA settings
CUDA_ARCH = -arch=sm_86
CUDA_FLAGS = -std=c++14 -O3 -use_fast_math --compiler-options -fPIC

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin

# Source files
CUDA_SOURCES = tournament_tree.cu test_tournament.cu
HEADERS = tournament_tree.h

# Object files
CUDA_OBJECTS = $(BUILD_DIR)/tournament_tree.o $(BUILD_DIR)/test_tournament.o

# Targets
TARGET = $(BIN_DIR)/tournament_test

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build the main executable
$(TARGET): $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) -o $@ $^

# Compile CUDA source files
$(BUILD_DIR)/tournament_tree.o: tournament_tree.cu tournament_tree.h | directories
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) -c -o $@ $<

$(BUILD_DIR)/test_tournament.o: test_tournament.cu tournament_tree.h | directories
	$(NVCC) $(CUDA_ARCH) $(CUDA_FLAGS) -c -o $@ $<

# Run tests
test: directories $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)

# Install (optional)
install: $(TARGET)
	@echo "Installing tournament_test to /usr/local/bin..."
	@sudo cp $(TARGET) /usr/local/bin/

# Debug build
debug: CUDA_FLAGS += -g -G -DDEBUG
debug: $(TARGET)

# Profile build
profile: CUDA_FLAGS += -lineinfo
profile: $(TARGET)

# Help
help:
	@echo "CUDA Tournament Tree Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  test      - Build and run tests"
	@echo "  clean     - Remove build artifacts"
	@echo "  debug     - Build with debug symbols"
	@echo "  profile   - Build with profiling info"
	@echo "  install   - Install to system"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - NVIDIA CUDA Toolkit"
	@echo "  - Compatible GPU with compute capability 5.2+"
	@echo ""
	@echo "Example usage:"
	@echo "  make           # Build the project"
	@echo "  make test      # Build and run tests"
	@echo "  make clean     # Clean build files"

# Phony targets
.PHONY: all directories test clean install debug profile help

# Dependencies
$(BUILD_DIR)/tournament_tree.o: tournament_tree.h
$(BUILD_DIR)/test_tournament.o: tournament_tree.h 