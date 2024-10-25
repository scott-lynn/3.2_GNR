##########################################
# AGNR - TB Model                        
##########################################
'''
Simulates AGNR using the TB formulation.
'''


##########################################
# Imports
##########################################
import numpy as np 

##########################################
# Functions
##########################################

def is_odd(num):
    '''
    Determines if number is odd.

    Args:
       num : int, Number for evaluation
    
    Returns:
        True / False : boolean
    '''
    mod = num % 2 
    if mod > 0:
        return(True)
    else:
        return(False)

def is_even(num):
    '''
    Determines if number is even.

    Args:
       num : int, Number for evaluation
    
    Returns:
        True / False : boolean
    '''
    mod = num % 2 
    if mod == 0:
        return(True)
    else:
        return(False)

def generate_hamiltonian(N, k, a, t1, epsilon_list):
    '''
    Generates the AGNR Hamiltonian for a system of size N.

    Args:
        N : int, Unit cell size (no of dimer lines [does not include zero])
        k : float, Current wavevector value
        t1 : float, 1st nearest neighbour hopping parameter
        epsilon_list : list(float), On-site energies

    Returns:
        H : np.array(dtype=complex), AGNR Hamiltonian
    '''

    # Calculate phase factor
    phase = np.exp(1j * k * a)

    # Initialise Hamiltonian matrix
    H = np.zeros((2*N, 2*N), dtype=complex)
    print(np.size(H))

    # Fill diagonal elements with on-site energies
    np.fill_diagonal(H, epsilon_list)

    # Bottom Row
    for i in range(0, N, 1):
        # LHS edge (Start on closed dimer line)
        if i == 0:
            # intra
            H[i, i+1] = -t1 
            H[i, i+N] = -t1

        # RHS edge 
        elif i == N-1:
            # Closed dimer line (even index)
            if is_even(i):
                #intra
                H[i, i - 1] = -t1 
                H[i, i + N] = -t1
            # Open dimer line (odd index)
            if is_odd(i):
                # intra
                H[i, i-1] = -t1
                # inter
                H[i, i+N] = -t1 * phase 

        # Even index cases
        elif is_even(i): 
            # intra
            H[i, i-1] = -t1 
            H[i, i+1] = -t1 
            H[i, i+N] = -t1 

        # Odd index cases
        elif is_odd(i):
            # intra
            H[i, i-1] = -t1 
            H[i, i+1] = -t1 
            # inter
            H[i, i+N] = -t1 * phase

        # Catch wrong input
        else:
            print('generate_hamiltonian : atom indice error')

    # Top Row
    for i in range(0, N, 1):
        atom_i = N + i
        # LHS edge (closed dimer line)
        if i == 0:
            # intra
            H[atom_i, atom_i+1] = -t1 
            H[atom_i, atom_i-N] = -t1 

        # RHS edge
        elif i == N - 1: 
            # Closed dimer line (even index)
            if is_even(i):
                # intra
                H[atom_i, atom_i - 1] = -t1 
                H[atom_i, atom_i - N] = -t1
            # Open dimer line (odd index)
            if is_odd(i):
                # intra
                H[atom_i, atom_i-1] = -t1
                # inter
                H[atom_i, atom_i-N] = -t1 * np.conj(phase)
            
        # Even index cases (closed dimer line)
        elif is_even(i): 
            # intra
            H[atom_i, atom_i-1] = -t1 
            H[atom_i, atom_i+1] = -t1 
            H[atom_i, atom_i-N] = -t1 

        # Odd index cases (open dimer line)
        elif is_odd(i):
            # intra
            H[atom_i, atom_i-1] = -t1 
            H[atom_i, atom_i+1] = -t1 
            # inter
            H[atom_i, atom_i-N] = -t1 * np.conj(phase)

        # Catch wrong input
        else:
            print('generate_hamiltonian : atom indice error')

    return H 


##########################################
# Main Execution Block
##########################################
def main():

    # Define params
    N: int = 2 
    a: float = 1 
    t1: float = 2.7 # eV
    epsilon_list = [1.0, 1.0] # eV

    # Set k-space range
    k_start: float = -np.pi/a 
    k_end: float = np.pi/a
    k_points: int = 200 
    k = np.linspace(k_start, k_end, k_points)
    k1 = 1.0

    # Generate hamiltonian using test logic
    H = generate_hamiltonian(N, k1, a, t1, epsilon_list)

    phase = np.exp(1j * k1 * a)

    # Create small hamiltonian for test
    H_test = np.zeros((4,4), dtype=complex)
    np.fill_diagonal(H_test, 1.0)
    H_test[0, 1] = -t1
    H_test[1, 0] = -t1

    H_test[0, 2] = -t1
    H_test[2, 0] = -t1 

    H_test[2, 3] = -t1 
    H_test[3, 2] = -t1 

    H_test[1, 3] = -t1 * phase
    H_test[3, 1] = -t1 * np.conj(phase)


    np.set_printoptions(linewidth=200)  # Adjust the value as needed

    print("Hamiltonian H:")
    print(H)

    print("\nTest Hamiltonian H_test:")
    print(H_test)

    if (np.array_equal(H, H_test) == True):
        print('\n Test result : Hamiltonian matrices are equal')
    else:
        print('\n Test result : Hamiltonian matrices do not match')

# Execute program
main()
    