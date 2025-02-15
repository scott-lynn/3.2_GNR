##########################################
# Imports
##########################################
import numpy as np 
import matplotlib.pyplot as plt
import gc


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

def generate_hamiltonian_ZGNR(N, k, a, epsilon_list, t1, t2, t3):
    '''
    Generates the ZGNR Hamiltonian for a system of size N.

    Args:
        N : int, Unit cell size (no of dimer lines [does not include zero])
        k : float, Current wavevector value
        t1 : float, 1st nearest neighbour hopping parameter
        epsilon_list : list(float), On-site energies

    Returns:
        H : np.array(dtype=complex), ZGNR Hamiltonian
    '''

    # Calculate phase factor
    phase = np.exp(1j * k * a)
    
    # Initialise Hamiltonian matrix
    H = np.zeros((2*N, 2*N), dtype=complex)

    # Fill diagonal elements with on-site energies
    np.fill_diagonal(H, epsilon_list)

    # Define edges
    LHS = 0
    RHS = N-1 

    for i in range(0, N):

        # Set indices
        bi = i 
        ti = i + N 

        # Irrespective of odd/even or edge
        H[bi, ti] += -t1 * (1 + np.conj(phase))
        H[ti, bi] += -t1 * (1 + phase)

        # LHS edge
        if i-1 >= LHS:
            if is_even(i):
                H[bi, bi-1] += -t1 
            elif is_odd(i):
                H[ti, ti-1] += -t1 

        # RHS edge
        if i+1 <= RHS:
            if is_even(i): 
                H[ti, ti+1] += -t1 
            elif is_odd(i):
                H[bi, bi+1] += -t1 

        if t2 != 0:

            # Irrespective of odd even or edge
            H[bi, bi] += -2 * t2 * np.cos(k*a)
            H[ti, ti] += -2 * t2 * np.cos(k*a)

            # LHS
            if i-1 >= LHS:
                H[ti, bi-1] += -t2 * (1 + phase)
                H[bi, ti-1] += -t2 * (1 + np.conj(phase))

            # RHS 
            if i+1 <= RHS:
                H[ti, bi+1] += -t2 * (1 + phase)
                H[bi, ti+1] += -t2 * (1 + np.conj(phase))
        
        if t3 != 0:

            # Irrespective of odd/even or edge
            H[bi, ti] += -2 * t3 * np.cos(k*a)
            H[ti, bi] += -2 * t3 * np.cos(k*a)

            # LHS
            if i-1 >= LHS:
                if is_even(i):
                    H[bi, bi-1] += -4 * t3 * np.cos(k*a)
                    H[ti, ti-1] += -t3 * (1 + np.conj(phase) + 2 * phase)
                elif is_odd(i):
                    H[bi, bi-1] += -t3 * (1 + phase + 2 * np.conj(phase))
                    H[ti, ti-1] += -4 * t3 * np.cos(k*a)
            if i-2 >= LHS:
                if is_even(i):
                    H[ti, bi-2] += -t3 * (1 + phase)
                elif is_odd(i):
                    H[bi, ti-2] += -t3 * (1 + np.conj(phase))

            if i+1 <= RHS:
                if is_even(i):
                    H[bi, bi+1] += -2 * t3 * (1 + np.cos(k*a))
                    H[ti, ti+1] += -4 * t3 * np.cos(k*a)
                if is_odd(i):
                    H[bi, bi+1] += -t3 * (phase + 3 * np.conj(phase))
                    H[ti, ti+1] += -t3 * (1 + np.conj(phase) + 2 * phase)
            if i+2 <= RHS:
                if is_even(i):
                    H[ti, bi+2] += -t3 * (1 + phase)
                if is_odd(i):
                    H[bi, ti+2] += -t3 * (1 + np.conj(phase))

    return H 

def generate_hamiltonian_AGNR(N, k, a, epsilon_list, t1, t2, t3):
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

    # Mark edges
    LHS = 0
    RHS = N-1

    # Count 0 -> N-1 and generate Hamiltonian
    for i in range(N):

        # Assign atom index
        bi = i
        ti = N + i

        # print('i:', i)
        # print('bi:', bi)
        # print('ti:', ti)

        if is_even(i):
            #print(i, 'is even')
            H[bi, ti] += -t1
            H[ti, bi] += -t1 
        if is_odd(i):
            #print(i, 'is odd')
            H[bi, ti] += -t1 * np.conj(phase)
            H[ti, bi] += -t1 * phase 

        # Nearest neighbours on left and right
        if i - 1 >= LHS:
            #print(i, 'is not LHS')
            H[bi, bi-1] += -t1 
            H[ti, ti-1] += -t1 
        if i +1 <= RHS:
            #print(i, 'is not RHS')
            H[bi, bi+1] += -t1 
            H[ti, ti+1] += -t1

        # 2NN
        if t2 != 0:
            # LHS
            if i - 1>= LHS:
                H[ti, bi-1] += -t2 * (1 + phase)
                H[bi, ti-1] += -t2 * (1 + np.conj(phase))
            if i - 2 >= LHS:
                H[bi, bi-2] += -t2
                H[ti, ti-2] += -t2 

            # RHS
            if i + 1 <= RHS:
                H[ti, bi+1] += -t2 * (1 + phase) 
                H[bi, ti+1] += -t2 * (1 + np.conj(phase))
            if i + 2 <= RHS:
                H[bi, bi+2] += -t2 
                H[ti, ti+2] += -t2

        # 3NN
        if t3 != 0:
            # LHS
            if i - 1 >= LHS:
                if is_even(i):
                    H[bi, bi-1] += -t3 * phase 
                    H[ti, ti-1] += -t3 * np.conj(phase)
                elif is_odd(i):
                    H[bi, bi-1] += -t3 * np.conj(phase)
                    H[ti, ti-1] += -t3 * phase 
            if i - 2 >= LHS: 
                if is_even(i):
                    H[ti, bi-2] += -t3 * (2 + phase)
                    H[bi, ti-2] += -t3 * (2 + np.conj(phase))
                elif is_odd(i):
                    H[ti, bi-2] += -t3 * (1 + 2 * phase)
                    H[bi, ti-2] += -t3 * (1 + 2 * np.conj(phase))
            if i - 3 >= LHS:
                H[bi, bi-3] += -t3
                H[ti, ti-3] += -t3 
            
            # RHS
            if i + 1 <= RHS:
                if is_even(i):
                    H[bi, bi+1] += -t3 * phase 
                    H[ti, ti + 1] += -t3 * np.conj(phase)
                elif is_odd(i): 
                    H[bi, bi+1] += -t3 * np.conj(phase)
                    H[ti, ti+1] += -t3 * phase 
            if i +2 <= RHS:
                if is_even(i): 
                    H[ti, bi+2] += -t3 * (2 + phase)
                    H[bi, ti+2] += -t3 * (2 + np.conj(phase))
                elif is_odd(i):
                    H[ti, bi+2] += -t3 * (1 + 2 * phase)
                    H[bi, ti+2] += -t3 * (1 + 2 * np.conj(phase))
            if i + 3 <= RHS: 
                H[bi, bi+3] += -t3 
                H[ti, ti+3] += -t3 

            # Even
            if is_even(i):
                H[ti, bi] += -2 * t3 * phase 
                H[bi, ti] += -2 * t3 * np.conj(phase)
            # Odd
            elif is_odd(i):
                H[ti, bi] += -2 * t3 
                H[bi, ti] += -2 * t3 
 
    np.set_printoptions(linewidth=200)  # Correctly display matrix in terminal
    # print(H)
    # if np.allclose(H, H.T.conj()):
    #     print("The Hamiltonian is Hermitian.")
    # else:
    #     print("The Hamiltonian is not Hermitian.")

    return H 

def test_hamiltonian_AGNR(type, print_matrix=None):
    
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
    t2: float = 0.0
    t3: float = 0.0
    k1 = 1.0

    # Phase factor
    phase = np.exp(1j * k1 * a)

    # Generate hamiltonian using test logic
    H = generate_hamiltonian_AGNR(N, k1, a, epsilon_list, t1, t2, t3)

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

def test_hamiltonian_ZGNR():
    # Define params
    N = 2
    a: float = 1 
    t1: float = 2.7 # eV
    k1 = 1.0
    epsilon_list = [0.0, 0.0, 0.0, 0.0]

    # Phase factor
    phase = np.exp(1j * k1 * a)

    H_test = generate_hamiltonian_ZGNR(N, k1, a, t1, epsilon_list)

    H = np.zeros((4,4), dtype=complex)

    H[0, 0] = 0.0
    H[0, 1] = -t1*(1 + np.conj(phase))

    H[1, 1] = 0.0
    H[1, 0] = -t1*(1 + phase)
    H[1, 2] = -t1 

    H[2, 2] = 0.0 
    H[2, 1] = -t1 
    H[2, 3] = -t1*(1 + phase) 

    H[3, 3] = 0.0 
    H[3, 2] = -t1*(1 + np.conj(phase)) 

    np.set_printoptions(linewidth=200)  # Correctly display matrix in terminal
    print('Hamiltonian H:')
    print(H)
    print('Hamiltonian Test:')
    print(H_test)

    if (np.array_equal(H, H_test) == True):
        print('Test result : Hamiltonian matrices are equal ZGNR')
    else:
        print('Test result : Hamiltonian matrices do not match for ZGNR')

def eigensolver_AGNR(N, k_vals, a, epsilon_list, t1, t2, t3):
    
    k_len = len(k_vals)
    eigenvalues = np.zeros((2*N, k_len))

    # Loop through k values and solve for the energy eigenvalues
    for i, k in enumerate(k_vals): 
        # Create Hamiltonian for the current k
        H = generate_hamiltonian_AGNR(N, k, a, epsilon_list, t1, t2, t3)

        # Solve for eigenvalues
        eigenvalues[:, i] = np.linalg.eigvalsh(H)
    
    # Calculate fermi energy shift
    E_F = np.median(eigenvalues)
    # Shift by fermi energy shift to centre about E_f = 0
    eigenvalues = eigenvalues - E_F

    return eigenvalues

def eigensolver_ZGNR(N, k_vals, a, epsilon_list, t1, t2, t3):
    
    k_len = len(k_vals)
    eigenvalues = np.zeros((2*N, k_len))

    # Loop through k values and solve for the energy eigenvalues
    for i, k in enumerate(k_vals): 
        # Create Hamiltonian for the current k
        H = generate_hamiltonian_ZGNR(N, k, a, epsilon_list, t1, t2, t3)

        # Solve for eigenvalues
        eigenvalues[:, i] = np.linalg.eigvalsh(H)

        # Calculate fermi energy shift
        E_F = np.median(eigenvalues)
        # Shift by fermi energy shift to centre about E_f = 0
        eigenvalues = eigenvalues - E_F

    return eigenvalues

def plot_bands(GNR_type, NN_type, N, k_vals, E_bands):

    plt.figure(dpi=200)

    for band in E_bands:
        plt.plot(k_vals, band, color='black')

    if GNR_type in ['AGNR', 'agnr', 'A', 'a']:
        plt.title('Band Structure of ' + str(N) + '-AGNR: TB ' + str(NN_type) +'NN')
    else:
        plt.title('Band Structure of ' + str(N) + '-ZGNR: TB ' + str(NN_type) + 'NN')

    plt.xlabel('k (1/a)')
    plt.ylabel('E (eV)')
    plt.ylim(-3, 3)
    plt.xlim(0, np.pi)
    plt.grid(True, which='both', axis='both', linewidth=0.5)
    plt.show()

def calculate_band_gap(GNR_type, k, N, a, epsilon_list, t1, t2, t3):

    # AGNR selected
    if GNR_type in ['AGNR', 'agnr', 'A', 'a']:
        H = generate_hamiltonian_AGNR(N, k, a, epsilon_list, t1, t2, t3)
    # ZGNR selected
    elif GNR_type in ['ZGNR', 'zgnr', 'Z', 'z']:
        H = generate_hamiltonian_ZGNR(N, k, a, epsilon_list, t1, t2, t3)
    else:
        print('calculate_band_gap() -  Error : Invalid GNR_type')
        exit()

    # Solve for eigenvalues
    eigenvalues = np.linalg.eigvalsh(H)
    E_F = np.median(eigenvalues)
    eigenvalues = eigenvalues - E_F
    #print('N:', N)

    # Remove numerical artefacts smaller than epsilon
    threshold = 1.0e-13
    eigenvalues[np.abs(eigenvalues) <= threshold] = 0

    # Sort array small to large
    eigenvalues = np.sort(eigenvalues) 
    #print(eigenvalues)

    # Check for pairs of zero eigenvalues
    if np.any(eigenvalues == 0):
        band_gap = 0
    else: 
        # Calculate band gap
        valence_max = np.max(eigenvalues[eigenvalues < 0])
        conduction_min = np.min(eigenvalues[eigenvalues > 0])
        #print('conduction_min:', conduction_min, 'valence_max', valence_max)

        # Remove numerical artefacts
        band_gap = conduction_min - valence_max
        if band_gap <= threshold:
            band_gap = 0

    #print('Band gap:', band_gap)
    return band_gap


def plot_band_gaps(GNR_type, NN_type, N_values, a, epsilon_list, t1, t2, t3): 

    k = 0
    band_gaps = [] 

    for N in N_values:

        band_gap = calculate_band_gap(GNR_type, k, N, a, epsilon_list, t1, t2, t3)
        band_gaps.append(band_gap)
        #print(N, band_gap)

    # If ZGNR additionally calculate band gap at k = pi/a
    if GNR_type in ['ZGNR', 'zgnr', 'Z', 'z']:
        k = np.pi/a 
        band_gaps_zgnr = []
        for N in N_values:

            band_gap_zgnr = calculate_band_gap(GNR_type, k, N, a, epsilon_list, t1, t2, t3)
            band_gaps_zgnr.append(band_gap_zgnr)
        
        # Plot ZGNR at k=pi/a
        plt.figure(dpi=200)
        plt.scatter(N_values, band_gaps_zgnr, edgecolors='black', facecolors='none', s=50)
        plt.xlabel('Ribbon Width (N)')
        plt.ylabel('Band Gap at k=pi/a (eV)')
        plt.xlim(0, np.max(N_values) + 1)
        plt.xticks(range(0, np.max(N_values) + 2, 2))
        plt.title('ZGNR-' + str(NN_type) + 'NN: Band Gap at k=pi/a as a function of Ribbon Width N')
        plt.grid(True)
        plt.show()

    # IF AGNR, separate into metallic/non-metallic cases
    if GNR_type in ['AGNR', 'agnr', 'A', 'a']:
        
        # Sort N
        N_3p = []
        N_3p1 = []
        N_3p2 = []

        # Sort band gaps
        band_gaps_3p = []
        band_gaps_3p1 = []
        band_gaps_3p2 = []
        
        # Generate values for Na=3p, Na=3p+1, Na=3p+2
        for p in range(0, N, 1):
            if 3 * p in N_values:
                N_3p.append(3 * p)
            if 3 * p + 1 in N_values:
                N_3p1.append(3 * p + 1)
            if 3 * p + 2 in N_values:
                N_3p2.append(3* p + 2)

        for i, N in enumerate(N_values):
            if N in N_3p:
                band_gaps_3p.append(band_gaps[i])
            elif N in N_3p1:
                band_gaps_3p1.append(band_gaps[i])
            elif N in N_3p2:
                band_gaps_3p2.append(band_gaps[i])

        # Plot at k=0
        plt.figure(dpi=200)
        plt.scatter(N_3p, band_gaps_3p, edgecolors='black', facecolors='none', s=50, label='Na=3p')
        plt.scatter(N_3p1, band_gaps_3p1, edgecolors='blue', facecolors='none', s=50, label='Na=3p+1')
        plt.scatter(N_3p2, band_gaps_3p2, edgecolors='red', facecolors='none', s=50, label='Na=3p+2')
        plt.xlabel('Ribbon Width (N)')
        plt.ylabel('Band Gap at k=0 (eV)')
        plt.xlim(0, np.max(N_values) + 1)
        plt.xticks(range(0, np.max(N_values) + 2, 2))
        plt.title('AGNR-' + str(NN_type) + 'NN: Band Gap at k=0 as a function of Ribbon Width N')
        plt.grid(True)
        plt.legend()
        plt.show()
                
    # General plot at k=0
    plt.figure(dpi=200)
    plt.scatter(N_values, band_gaps, edgecolors='black', facecolors='none', s=50)
    plt.xlabel('Ribbon Width (N)')
    plt.ylabel('Band Gap at k=0 (eV)')
    plt.xlim(0, np.max(N_values) + 1)
    plt.xticks(range(0, np.max(N_values) + 2, 2))
    plt.grid(True)

    # Determine title based on GNR type
    if GNR_type in ['AGNR', 'agnr', 'A', 'a']:
        plt.title('AGNR-' + str(NN_type) +'NN: Band Gap at k=0 as a function of Ribbon Width N')
    elif GNR_type in ['ZGNR', 'zgnr', 'Z', 'z']:
        plt.title('ZGNR-' + str(NN_type) + 'NN: Band Gap at k=0 as a function of Ribbon Width N')
    
    plt.show()
    

##########################################
# Main Execution Block
##########################################
def main():

    print('---------- PROGRAM START: GNR.py ----------')
    # Ask user for GNR type
    GNR_type = input('Enter GNR type (AGNR or ZGNR): ')

    if GNR_type not in ['AGNR', 'agnr', 'A', 'a', 'ZGNR', 'zgnr', 'Z', 'z']:
        print('GNR_type Error : ', GNR_type, ' is not a valid GNR_type')
        exit()

    # Ask user for N type
    try:
        N = int(input('Enter width (N) value: '))
    except ValueError:
        print('N type error:', N, 'is not a valid value for N, it must be a positive integer greater than 2')
        exit()
    if N < 2:
        print('N type error:', N, 'is not a valid value for N, it must be a positive integer greater than 2')
        exit()

    # Ask user for number of nearest neighbours
    NN_type = int(input('Enter number of nearest neighbours:'))

    if NN_type not in [1, 2, 3]:
        print('NN_type error:', NN_type, ' is not a valid input - must be 1, 2 or 3')
        exit()

    # 1st NN
    t1: float = 2.7     # 2.7 eV

    # 2nd NN
    if NN_type == 2 or NN_type == 3:
        t2: float = 0.2     # 0.2 eV
    else: 
        t2: float = 0.0

    # 3rd NN
    if NN_type == 3:
        t3: float = 0.18    # 0.18 eV
    else: 
        t3: float = 0.0 

    # Define params
    a: float = 1.0
    epsilon_list = [0.0] 

    # Set k-space range
    k_start: float = 0
    k_end: float = np.pi/a
    k_points: int = 200
    k = np.linspace(k_start, k_end, k_points)

    # Set N_values range for band gap calculations
    N_start = 2 
    N_end = 20
    N_values = range(N_start, N_end + 1, 1)

    # Call eigensolver based on GNR_Type
    if GNR_type in ['AGNR', 'agnr', 'A', 'a']:
        E_bands = eigensolver_AGNR(N, k, a, epsilon_list, t1, t2, t3)
    elif GNR_type in ['ZGNR', 'zgnr', 'Z', 'z']:
        E_bands = eigensolver_ZGNR(N, k, a, epsilon_list, t1, t2, t3)
    else:
        print('GNR_type Error : ', GNR_type, ' is not a valid input. Use AGNR or ZGNR')
    
    ###### Plotting ######

    # Plot band structure
    plot_bands(GNR_type, NN_type, N, k, E_bands)
    
    # Plot band gap
    plot_band_gaps(GNR_type, NN_type, N_values, a, epsilon_list, t1, t2, t3)

# Execute program 
if __name__=="__main__":
    main()
    
# Clear memory
gc.collect()