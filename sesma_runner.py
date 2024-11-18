import torch
import numpy as np
import logging
import sys
from tabulate import tabulate  # for nice table formatting

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sesma_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

def load_atom_mapping():
    """Load the atom mapping file"""
    id_to_atom = {}
    with open("atom_mapping.txt", "r") as f:
        for line in f:
            idx, atom = line.strip().split(" ", 1)
            id_to_atom[int(idx)] = atom
    return id_to_atom

def read_matrix_files(index):
    """Read matrix and atoms"""
    try:
        # Read matrix
        matrix = np.loadtxt(f"clause_{index}_matrix.txt", dtype=np.int32)
        # Read atoms as integers
        atoms = np.loadtxt(f"clause_{index}_atoms.txt", dtype=np.int32)
        
        logging.debug(f"Read clause {index}:")
        logging.debug(f"Matrix shape: {matrix.shape}")
        logging.debug(f"Atoms: {atoms}")
        
        return matrix, atoms
    except Exception as e:
        logging.error(f"Error reading files for clause {index}: {str(e)}")
        return None, None

def visualize_matrix(matrix, atoms, atom_mapping, title="Matrix"):
    """Visualize a matrix with its atoms"""
    # Create header with atom names (shortened for display)
    headers = [atom_mapping[a][:20] + "..." if len(atom_mapping[a]) > 20 
              else atom_mapping[a] for a in atoms]
    
    # Format matrix rows
    rows = [[f"{val}" for val in row] for row in matrix]
    
    table = tabulate(rows, headers=headers, tablefmt='grid')
    logging.info(f"\n{title}:\n{table}")

def visualize_combination(combination, atom_mapping):
    """Visualize a single combination"""
    items = []
    for atom, value in combination.items():
        atom_str = atom_mapping[atom]
        items.append(f"{atom_str[:20]}{'...' if len(atom_str) > 20 else ''}: {value}")
    return "\n".join(items)


def get_sesma_product(m1, atoms1, m2, atoms2):
    logging.debug(f"Running Sesma product on device: {device}")
    if len(m1) == 0 or len(m2) == 0:
        return []
        
    # Convert to boolean tensors and move to GPU
    t1 = torch.tensor(m1, dtype=torch.bool, device=device)
    t2 = torch.tensor(m2, dtype=torch.bool, device=device)
    
    logging.debug(f"Matrix shapes: t1={t1.shape}, t2={t2.shape}")
    logging.debug(f"Atoms1: {atoms1}, Atoms2: {atoms2}")
    
    # Create expanded tensors for comparison
    n1, m1_cols = t1.shape
    n2, m2_cols = t2.shape
    
    # Create result tensor
    result = torch.zeros((n1, n2), dtype=torch.bool, device=device)
    
    # Find common atoms and their positions
    common_atoms = set(atoms1) & set(atoms2)
    common_indices = [(list(atoms1).index(atom), list(atoms2).index(atom)) 
                     for atom in common_atoms]
    
    logging.debug(f"Common atoms: {common_atoms}")
    logging.debug(f"Common indices: {common_indices}")
    
    # Compare row by row
    for i in range(n1):
        for j in range(n2):
            # Check if all common atoms match
            matches = torch.tensor(True, dtype=torch.bool, device=device)
            for idx1, idx2 in common_indices:
                matches = matches & (t1[i, idx1] == t2[j, idx2])
            
            result[i, j] = matches
    
    # Convert result to long for final output
    result = result.long()
    
    logging.debug(f"Result shape: {result.shape}")
    if torch.cuda.is_available():
        logging.debug(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    return result


def get_batch_size(n1, n2, atom_count, available_memory=800*1024*1024):  # 800MB default
    """Calculate optimal batch size based on available GPU memory"""
    # Each boolean tensor element takes 1 byte
    # Add 20% overhead for GPU operations
    element_size = 1  # byte for boolean
    memory_per_row = atom_count * element_size
    total_combinations = n1 * n2
    
    # Calculate batch size leaving room for other operations
    safe_memory = available_memory * 0.8  # Use 80% of available memory
    batch_size = int(safe_memory / memory_per_row)
    
    # Cap batch size at total combinations
    batch_size = min(batch_size, total_combinations, 1000)  # Also cap at 1000 for safety
    
    return max(1, batch_size)  # Ensure at least 1

def get_intersection(m1, atoms1, m2, atoms2):
    logging.debug(f"Starting intersection with shapes: m1={np.array(m1).shape}, m2={np.array(m2).shape}")
    
    atoms1 = atoms1.tolist() if isinstance(atoms1, np.ndarray) else list(atoms1)
    atoms2 = atoms2.tolist() if isinstance(atoms2, np.ndarray) else list(atoms2)
    
    common_atoms = set(atoms1).intersection(set(atoms2))
    logging.debug(f"Common atoms: {common_atoms}")
    
    if not common_atoms:
        logging.debug("No common atoms found - using cartesian product")
        atoms_intersection = sorted(list(set(atoms1) | set(atoms2)))
        
        # Convert to CPU tensors first
        t1 = torch.tensor(m1, dtype=torch.bool)
        t2 = torch.tensor(m2, dtype=torch.bool)
        
        n1, m1_cols = t1.shape
        n2, m2_cols = t2.shape
        
        # Calculate batch size based on available GPU memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            batch_size = get_batch_size(n1, n2, len(atoms_intersection), free_memory)
        else:
            batch_size = 100
            
        logging.debug(f"Using batch size: {batch_size}")
        result_set = set()  # Use set for deduplication
        
        # Process in smaller batches
        for i in range(0, n1, batch_size):
            i_end = min(i + batch_size, n1)
            
            # Move batch to GPU
            batch1 = t1[i:i_end].to(device)
            
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                
                # Move second batch to GPU
                batch2 = t2[j:j_end].to(device)
                
                # Process combinations
                for bi in range(batch1.size(0)):
                    row1 = batch1[bi]
                    for bj in range(batch2.size(0)):
                        row2 = batch2[bj]
                        
                        # Create combination
                        result_row = []
                        for atom in atoms_intersection:
                            if atom in atoms1:
                                idx = atoms1.index(atom)
                                result_row.append(int(row1[idx].item()))
                            elif atom in atoms2:
                                idx = atoms2.index(atom)
                                result_row.append(int(row2[idx].item()))
                            else:
                                result_row.append(0)
                        
                        result_set.add(tuple(result_row))
                
                # Clear GPU cache after each inner batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            logging.debug(f"Processed batch {i}/{n1}, current combinations: {len(result_set)}")
        
        result_matrix = [list(row) for row in result_set]
        logging.debug(f"Found {len(result_matrix)} unique combinations")
        return [atoms_intersection, result_matrix]
    
    else:
        # Handle common atoms case with memory management
        n1 = len(m1)
        n2 = len(m2)
        
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            batch_size = get_batch_size(n1, n2, len(common_atoms), free_memory)
        else:
            batch_size = 100
            
        result_set = set()
        
        # Process in batches
        for i in range(0, n1, batch_size):
            i_end = min(i + batch_size, n1)
            batch1 = torch.tensor(m1[i:i_end], dtype=torch.bool, device=device)
            
            for j in range(0, n2, batch_size):
                j_end = min(j + batch_size, n2)
                batch2 = torch.tensor(m2[j:j_end], dtype=torch.bool, device=device)
                
                # Check matches within batch
                matches = True
                for atom in common_atoms:
                    idx1 = atoms1.index(atom)
                    idx2 = atoms2.index(atom)
                    matches = matches & (batch1[:, idx1].unsqueeze(1) == batch2[:, idx2].unsqueeze(0))
                
                if matches.any():
                    match_indices = torch.nonzero(matches)
                    for idx1, idx2 in match_indices:
                        # Create combination
                        result_row = []
                        for atom in sorted(list(set(atoms1) | set(atoms2))):
                            if atom in atoms1:
                                idx = atoms1.index(atom)
                                result_row.append(int(batch1[idx1, idx].item()))
                            elif atom in atoms2:
                                idx = atoms2.index(atom)
                                result_row.append(int(batch2[idx2, idx].item()))
                            else:
                                result_row.append(0)
                                
                        result_set.add(tuple(result_row))
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        result_matrix = [list(row) for row in result_set]
        logging.debug(f"Found {len(result_matrix)} unique combinations")
        return [sorted(list(set(atoms1) | set(atoms2))), result_matrix]



def get_sesma_product(m1, atoms1, m2, atoms2):
    if len(m1) == 0 or len(m2) == 0:
        return []
        
    # Move data to GPU once
    t1 = torch.tensor(m1, dtype=torch.bool, device=device)
    t2 = torch.tensor(m2, dtype=torch.bool, device=device)
    
    # Find common atoms
    common_atoms = set(atoms1) & set(atoms2)
    common_indices = [(atoms1.index(atom), atoms2.index(atom)) 
                     for atom in common_atoms]
    
    # Create result tensor on GPU
    result = torch.ones((t1.shape[0], t2.shape[0]), dtype=torch.bool, device=device)
    
    # Compute matches entirely on GPU
    for idx1, idx2 in common_indices:
        # Use broadcasting for efficient comparison
        matches = t1[:, idx1].unsqueeze(1) == t2[:, idx2].unsqueeze(0)
        result &= matches
    
    return result.long()


def process_all_clauses():
    try:
        global atom_mapping  # Make atom_mapping available for visualization
        # Load atom mapping
        atom_mapping = load_atom_mapping()
        logging.info("Loaded atom mapping:")
        for idx, atom in sorted(atom_mapping.items()):
            logging.info(f"{idx}: {atom}")
        
        # Read matrices
        matrices = []
        atoms_lists = []
        i = 1
        
        while True:
            matrix, atoms = read_matrix_files(i)
            if matrix is None:
                break
            matrices.append(matrix)
            atoms_lists.append(atoms)
            i += 1
        
        if not matrices:
            logging.error("No matrix files found!")
            return False
        
        logging.info(f"\nFound {len(matrices)} clause matrices to process")
        
        # Process matrices sequentially
        result_atoms = atoms_lists[0]
        result_matrix = matrices[0]
        
        for i in range(1, len(matrices)):
            result = get_intersection(result_matrix, result_atoms, 
                                    matrices[i], atoms_lists[i])
            
            result_atoms = result[0]
            result_matrix = result[1]
            
            if not result_matrix:
                logging.warning("No valid intersections found!")
                return False
        
        # Save and visualize final result
        if result_matrix:
            np.savetxt("final_matrix.txt", result_matrix, fmt='%d')
            with open("final_atoms.txt", 'w') as f:
                f.write("\n".join(atom_mapping[a] for a in result_atoms))
            
            logging.info("\nFinal Results:")
            logging.info(f"Matrix size: {len(result_matrix)}x{len(result_atoms)}")
            logging.info("Atoms:")
            for atom_id in result_atoms:
                logging.info(f"{atom_id}: {atom_mapping[atom_id]}")
            logging.info("\nResults saved to 'final_matrix.txt' and 'final_atoms.txt'")
            return True  # Indicate success with results
        else:
            logging.warning("\nNo valid solutions found!")
            return False  # Indicate no results
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Stack trace:", exc_info=True)
        return False


if __name__ == "__main__":
    logging.info("Starting Sesma product processing...")
    success = process_all_clauses()
    sys.exit(0 if success else 1) 

