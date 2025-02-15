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
import matplotlib.pyplot as plt

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

    # Fill diagonal elements with on-site energies
    np.fill_diagonal(H, epsilon_list)

    # Count 0 -> N-1 and generate Hamiltonian
    for i in range(0, N, 1):

        # Assign atom index
        bottom_atom = i
        top_atom = N + i

        # LHS edge (Start on closed dimer line)
        if i == 0:
            # bottom row
            H[bottom_atom, bottom_atom + 1] = -t1     
            H[bottom_atom, bottom_atom + N] = -t1     
            # top row
            H[top_atom, top_atom + 1] = -t1           
            H[top_atom, top_atom - N] = -t1          

        # RHS edge 
        elif i == N-1:
            # Closed dimer line (even index)
            if is_even(i):
                # bottom row
                H[bottom_atom, bottom_atom - 1] = -t1 
                H[bottom_atom, bottom_atom + N] = -t1
                # top row
                H[top_atom, top_atom - 1] = -t1 
                H[top_atom, top_atom - N] = -t1
                
            # Open dimer line (odd index)
            elif is_odd(i):
                # bottom row
                H[bottom_atom, bottom_atom - 1] = -t1             
                H[bottom_atom, bottom_atom + N] = -t1 * phase     
                # top row
                H[top_atom, top_atom-1] = -t1
                H[top_atom, top_atom-N] = -t1 * np.conj(phase)

        # Even index cases
        elif is_even(i): 
            # bottom row
            H[bottom_atom, bottom_atom - 1] = -t1 
            H[bottom_atom, bottom_atom + 1] = -t1 
            H[bottom_atom, bottom_atom + N] = -t1 
            # top row
            H[top_atom, top_atom - 1] = -t1 
            H[top_atom, top_atom + 1] = -t1 
            H[top_atom, top_atom - N] = -t1 

        # Odd index cases
        elif is_odd(i):
            # bottom row
            H[bottom_atom, bottom_atom - 1] = -t1 
            H[bottom_atom, bottom_atom + 1] = -t1 
            H[bottom_atom, bottom_atom + N] = -t1 * phase
            # top row
            H[top_atom, top_atom - 1] = -t1 
            H[top_atom, top_atom + 1] = -t1 
            H[top_atom, top_atom - N] = -t1 * np.conj(phase)

        # Catch wrong input
        else:
            print('generate_hamiltonian : atom indice error')

    return H 

def test_hamiltonian(type, print_matrix=None):
    
    if type == 'even':
        N = 2
        epsilon_list = [1.0, 1.0]
    elif type == 'odd':
        N = 3
        epsilon_list = [1.0, 1.0, 1.0]
    else:
        print('Error: test_hamiltonian : Invalid type input')

    # Define params
    a: float = 1 
    t1: float = 2.7 # eV
    k1 = 1.0

    # Phase factor
    phase = np.exp(1j * k1 * a)

    # Generate hamiltonian using test logic
    H = generate_hamiltonian(N, k1, a, t1, epsilon_list)

    # Test visually
    if (print_matrix == True):
        np.set_printoptions(linewidth=200)  # Correctly display matrix in terminal
        print('Hamiltonian H:')
        print(H)

    # Test N=2
    if (N==2): 

        # Generate N=2 Hamiltonian
        H_even = np.zeros((4,4), dtype=complex)
        np.fill_diagonal(H_even, 1.0)
        H_even[0, 1] = -t1
        H_even[1, 0] = -t1

        H_even[0, 2] = -t1
        H_even[2, 0] = -t1 

        H_even[2, 3] = -t1 
        H_even[3, 2] = -t1 

        H_even[1, 3] = -t1 * phase
        H_even[3, 1] = -t1 * np.conj(phase)

        if (print_matrix == True):
            print('Test Hamiltonian H_test:')
            print(H_even)

        # Numerical element matching
        if (np.array_equal(H, H_even) == True):
            print('Test result : Hamiltonian matrices are equal for even N')
        else:
            print('Test result : Hamiltonian matrices do not match for even N')

    # Test N=3
    elif (N==3):

        # Generate N=3 Hamiltonian
        H_odd = np.zeros((6,6), dtype=complex)
        np.fill_diagonal(H_odd, 1.0)
        # Row 0
        H_odd[0, 1] = -t1
        H_odd[0, 3] = -t1
        # Row 1
        H_odd[1, 0] = -t1 
        H_odd[1, 2] = -t1 
        H_odd[1, 4] = -t1 * phase 
        # Row 2
        H_odd[2, 1] = -t1 
        H_odd[2, 5] = -t1 
        # Row 3
        H_odd[3, 0] = -t1 
        H_odd[3, 4] = -t1 
        # Row 4
        H_odd[4, 1] = -t1 * np.conj(phase)
        H_odd[4, 3] = -t1 
        H_odd[4, 5] = -t1 
        # Row 5
        H_odd[5, 2] = -t1 
        H_odd[5, 4] = -t1

        if (print_matrix == True):
            print('\nTest Hamiltonian H_test:')
            print(H_odd)

        # Numerical element matching
        if (np.array_equal(H, H_odd) == True):
            print('Test result : Hamiltonian matrices are equal for odd N')
        else:
            print('Test result : Hamiltonian matrices do not match for odd N')

def eigensolver(N, k_vals, a, t1, epsilon_list):
    
    k_len = len(k_vals)
    eigenvalues = np.zeros((2*N, k_len))

    # Loop through k values and solve for the energy eigenvalues
    for i, k in enumerate(k_vals): 
        # Create Hamiltonian for the current k
        H = generate_hamiltonian(N, k, a, t1, epsilon_list)

        # Solve for eigenvalues
        eigenvalues[:, i] = np.linalg.eigvalsh(H)

    return eigenvalues

def plot_bands(k_vals, E_bands):

    plt.figure(dpi=200)

    for band in E_bands:
        plt.plot(k_vals, band, color='black')

    plt.title('Band Structure of 14-AGNR: TB NN-Hopping ')
    plt.xlabel('k (1/a)')
    plt.ylabel('E (eV)')
    #plt.ylim(-3, 3)
    plt.xlim(0, np.pi)
    plt.grid(True, which='both', axis='both', linewidth=0.5)
    plt.show()

##########################################
# Main Execution Block
##########################################
def main():

    # Define params
    N: int = 2
    a: float = 1 
    t1: float = 2.7 # eV
    epsilon_list = [0.0] # eV

    # Set k-space range
    k_start: float = 0
    k_end: float = np.pi/a
    k_points: int = 200 
    k = np.linspace(k_start, k_end, k_points)

    # Test for odd and even N cases
    test_hamiltonian('odd')
    test_hamiltonian('even', print_matrix=True)

    E_bands = eigensolver(N, k, a, t1, epsilon_list)
    plot_bands(k, E_bands)

# Execute program (professionally)

if __name__=="__main__":
    main()
    