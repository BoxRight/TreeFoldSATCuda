import sys
import re

MAX_VECTORS_TO_PRINT = 100

def query_vectors(filename, required_elements, forbidden_elements):
    """Query a vector database file directly"""
    found_vectors = []
    processed = 0
    
    print(f"Querying for vectors with:")
    print(f"  Required elements: {required_elements}")
    print(f"  Forbidden elements: {forbidden_elements}")
    
    with open(filename, 'r') as f:
        # Skip header if present
        first_line = f.readline()
        if first_line.startswith('#') or 'Final results' in first_line:
            pass  # Skip header
        else:
            f.seek(0)  # Reset if no header
        
        # Process each vector
        for line in f:
            processed += 1
            
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            # Parse vector
            vector_match = re.search(r'\[(.*?)\]', line)
            if not vector_match:
                continue
                
            vector_str = vector_match.group(1)
            vector = [int(x.strip()) for x in vector_str.split(',') if x.strip()]
            
            # Check constraints
            if all(elem in vector for elem in required_elements) and \
               not any(elem in vector for elem in forbidden_elements):
                found_vectors.append(vector)
                
                # Print immediately to show progress
                if len(found_vectors) <= MAX_VECTORS_TO_PRINT:
                    print(f"{vector}")
            
            # Show progress periodically
            if processed % 1000000 == 0:
                print(f"Processed {processed:,} vectors, found {len(found_vectors)} matches so far...")
            
            # Stop if we have enough matches
            if len(found_vectors) >= MAX_VECTORS_TO_PRINT and processed >= 1000000:
                break
    
    print(f"Query complete: found {len(found_vectors)} matching vectors (processed {processed:,} vectors)")
    return found_vectors

def main():
    if len(sys.argv) < 2:
        print("Usage: python vector_query.py <vector_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Vector Query Tool - reading from {filename}")
    print("Type 'help' for available commands")
    
    while True:
        try:
            command = input("> ").strip()
            if not command:
                continue
            
            parts = command.split(None, 1)
            cmd = parts[0].lower()
            
            if cmd in ('quit', 'exit'):
                break
            elif cmd == 'help':
                print("\nVector Query Tool - Commands:")
                print("  help            - Show this help menu")
                print("  path <elements> - Find vectors matching path [1,-2,3] (1=required, -2=forbidden)")
                print("  count           - Count total vectors in the file")
                print("  sample <n>      - Show first n vectors from the file")
                print("  quit            - Exit the program\n")
            elif cmd == 'path':
                if len(parts) < 2:
                    print("Error: Missing path elements")
                    continue
                
                # Parse path elements
                try:
                    # Remove brackets and split by commas
                    path_str = parts[1].strip()
                    if path_str.startswith('[') and path_str.endswith(']'):
                        path_str = path_str[1:-1]
                    
                    path_elements = [int(x) for x in path_str.split(',')]
                    
                    # Separate required and forbidden
                    required = [x for x in path_elements if x > 0]
                    forbidden = [-x for x in path_elements if x < 0]
                    
                    query_vectors(filename, required, forbidden)
                except ValueError:
                    print("Error: Invalid path format. Use format [1,-2,3]")
            elif cmd == 'count':
                try:
                    count = 0
                    with open(filename, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                count += 1
                    print(f"Total vectors in file: {count:,}")
                except Exception as e:
                    print(f"Error counting vectors: {e}")
            elif cmd == 'sample':
                try:
                    n = 10  # Default
                    if len(parts) > 1:
                        n = int(parts[1])
                    
                    with open(filename, 'r') as f:
                        count = 0
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                print(line.strip())
                                count += 1
                                if count >= n:
                                    break
                except Exception as e:
                    print(f"Error sampling vectors: {e}")
            else:
                print(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
