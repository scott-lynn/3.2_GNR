import numpy as np
import matplotlib.pyplot as plt

##########################################
# Utility Functions
##########################################

def is_odd(num):
    """Determine if a number is odd."""
    return num % 2 != 0

def is_even(num):
    """Determine if a number is even."""
    return num % 2 == 0

##########################################
# ZGNR Hamiltonian Function
##########################################

def generate_hamiltonian_ZGNR(N, k, a, t1, epsilon_list):
    """
    Generate the ZGNR Hamiltonian for a system of size N.

    Args:
        N (int): Number of dimer lines in the unit cell.
        k (float): Wavevector.
        a (float): Lattice constant.
        t1 (float): Nearest-neighbor hopping parameter.
        epsilon_list (list of float): On-site energies for each atom in the unit cell.

    Returns:
        np.ndarray: Hamiltonian matrix for the ZGNR.
    """
    # Calculate phase factor
    phase = np.exp(1j * k * a)
    
    # Initialize Hamiltonian matrix
    H = np.zeros((2 * N, 2 * N), dtype=complex)

    # Fill diagonal elements with on-site energies
    np.fill_diagonal(H, epsilon_list)

    # Define connections for nearest-neighbor hopping
    for i in range(N):
        bottom_atom = i
        top_atom = N + i
        
        # Left edge of the ribbon
        if i == 0:
            H[bottom_atom, bottom_atom + 1] = -t1  # Horizontal bond on bottom row
            H[bottom_atom, top_atom] = -t1  # Vertical bond to top row

            H[top_atom, top_atom + 1] = -t1  # Horizontal bond on top row
            H[top_atom, bottom_atom] = -t1  # Back vertical bond

        # Right edge of the ribbon
        elif i == N - 1:
            H[bottom_atom, bottom_atom - 1] = -t1  # Horizontal bond on bottom row
            H[bottom_atom, top_atom] = -t1 * phase  # Vertical bond with phase

            H[top_atom, top_atom - 1] = -t1  # Horizontal bond on top row
            H[top_atom, bottom_atom] = -t1 * np.conj(phase)  # Back vertical bond

        # Interior atoms
        else:
            H[bottom_atom, bottom_atom + 1] = -t1  # Right horizontal bond
            H[bottom_atom, bottom_atom - 1] = -t1  # Left horizontal bond
            H[bottom_atom, top_atom] = -t1  # Vertical bond

            H[top_atom, top_atom + 1] = -t1  # Right horizontal bond
            H[top_atom, top_atom - 1] = -t1  # Left horizontal bond
            H[top_atom, bottom_atom] = -t1  # Back vertical bond

    return H

##########################################
# Band Structure Calculation
##########################################

def calculate_band_structure(N, k_values, a, t1, epsilon_list):
    """
    Calculate the band structure for a ZGNR.

    Args:
        N (int): Number of dimer lines in the unit cell.
        k_values (np.ndarray): Array of k values.
        a (float): Lattice constant.
        t1 (float): Nearest-neighbor hopping parameter.
        epsilon_list (list of float): On-site energies.

    Returns:
        np.ndarray: Array of eigenvalues for each k.
    """
    energies = []

    for k in k_values:
        H = generate_hamiltonian_ZGNR(N, k, a, t1, epsilon_list)
        eigenvalues = np.linalg.eigvalsh(H)
        energies.append(eigenvalues)

    return np.array(energies).T  # Transpose for easier plotting

##########################################
# Main Execution Code
##########################################

# Parameters
N = 16  # Number of dimer lines (can be adjusted)
a = 1.0  # Lattice constant
t1 = 2.7  # Nearest-neighbor hopping parameter (eV)
epsilon_list = [0.0] * (2 * N)  # On-site energies set to zero

# Define k values in the Brillouin zone
k_min, k_max = 0, np.pi / a
k_values = np.linspace(k_min, k_max, 500)

# Calculate band structure
energies = calculate_band_structure(N, k_values, a, t1, epsilon_list)

# Plot the band structure
plt.figure(figsize=(8, 6))
for band in energies:
    plt.plot(k_values, band, color='b', lw=0.8)
plt.xlabel("Wavevector $k$")
plt.ylabel("Energy (eV)")
plt.ylim(-3, 3)
plt.title("Band Structure of ZGNR with N={} Dimer Lines".format(N))
plt.grid(True)
plt.show()
