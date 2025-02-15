import numpy as np
import matplotlib.pyplot as plt

def build_k_dependent_hamiltonian(width_dimer_lines, t=-2.7, t2=-0.1, t3=-0.05, k=0):
    """
    Constructs the Hamiltonian matrix for a single AGNR unit cell at a given k-point,
    including first, second, and third nearest-neighbor interactions and intra-cell
    hopping terms with phase factors e^(i * k * a).
    """
    num_atoms = 2 * width_dimer_lines
    H_k = np.zeros((num_atoms, num_atoms), dtype=complex)

    # First Nearest Neighbor Hopping (1NN) with intra-cell hopping
    for j in range(width_dimer_lines):
        atom_a = 2 * j
        atom_b = atom_a + 1

        # 1NN within the dimer line
        H_k[atom_a, atom_b] = t
        H_k[atom_b, atom_a] = t

        # 1NN connections to next dimer with phase factor
        if j < width_dimer_lines - 1:
            next_a = 2 * (j + 1)
            H_k[atom_a, next_a + 1] = t
            H_k[next_a + 1, atom_a] = t
            H_k[atom_b, next_a] = t
            H_k[next_a, atom_b] = t

    # Periodic boundary (1NN) with phase factor e^(i * k) for inter-cell hopping
    for j in range(width_dimer_lines - 1):
        H_k[2 * j, 2 * (j + 1) + 1] = t * np.exp(1j * k)
        H_k[2 * (j + 1) + 1, 2 * j] = t * np.exp(-1j * k)

    # Second Nearest Neighbor Hopping (2NN) with intra-cell phase factors
    for j in range(width_dimer_lines - 1):
        atom_a = 2 * j
        atom_b = atom_a + 1
        next_a = 2 * (j + 1)
        H_k[atom_a, next_a] = t2 * np.exp(1j * k)
        H_k[next_a, atom_a] = t2 * np.exp(-1j * k)
        H_k[atom_b, next_a + 1] = t2 * np.exp(1j * k)
        H_k[next_a + 1, atom_b] = t2 * np.exp(-1j * k)

    # Third Nearest Neighbor Hopping (3NN) with intra-cell phase factors
    for j in range(width_dimer_lines - 2):
        atom_a = 2 * j
        atom_b = atom_a + 1
        next_next_a = 2 * (j + 2)
        next_next_b = next_next_a + 1
        H_k[atom_a, next_next_b] = t3 * np.exp(2j * k)
        H_k[next_next_b, atom_a] = t3 * np.exp(-2j * k)
        H_k[atom_b, next_next_a] = t3 * np.exp(2j * k)
        H_k[next_next_a, atom_b] = t3 * np.exp(-2j * k)

    return H_k

def calculate_band_structure(width_dimer_lines, t=-2.7, t2=-0.1, t3=-0.05, num_k_points=100):
    """
    Calculates the band structure for an AGNR by sweeping through k-points.
    """
    k_vals = np.linspace(0, np.pi, num_k_points)  # k points in the Brillouin zone
    eigenvalues = np.zeros((num_k_points, 2 * width_dimer_lines))

    for i, k in enumerate(k_vals):
        H_k = build_k_dependent_hamiltonian(width_dimer_lines, t, t2, t3, k)
        eigenvalues[i, :] = np.linalg.eigvalsh(H_k)  # Compute eigenvalues

    return k_vals, eigenvalues

def plot_band_structure(k_vals, eigenvalues):
    """
    Plots the band structure.
    """
    plt.figure(figsize=(8, 6))
    for band in eigenvalues.T:
        plt.plot(k_vals, band, color='b')
    plt.xlabel("k (1/Å)")
    plt.ylabel("Energy (eV)")
    plt.ylim(-3,3)
    plt.title("Band Structure of AGNR")
    plt.grid(True)
    plt.show()

def build_zgnr_k_dependent_hamiltonian(width_zigzag_lines, t=-2.7, t2=-0.1, t3=-0.05, k=0):
    """
    Constructs the Hamiltonian matrix for a single ZGNR unit cell at a given k-point,
    including first, second, and third nearest-neighbor interactions.
    """
    num_atoms = 2 * width_zigzag_lines  # Each zigzag line has two atoms (top and bottom)
    H_k = np.zeros((num_atoms, num_atoms), dtype=complex)

    # First Nearest Neighbor Hopping (1NN) with intra-cell hopping
    for j in range(width_zigzag_lines):
        atom_a = 2 * j
        atom_b = atom_a + 1

        # 1NN within the zigzag line
        H_k[atom_a, atom_b] = t
        H_k[atom_b, atom_a] = t

        # 1NN connections to adjacent zigzag lines
        if j < width_zigzag_lines - 1:
            next_a = 2 * (j + 1)
            H_k[atom_b, next_a] = t
            H_k[next_a, atom_b] = t

    # 1NN inter-cell hopping with phase factor e^(i * k a)
    for j in range(width_zigzag_lines):
        if j < width_zigzag_lines - 1:
            atom_a = 2 * j
            atom_b = atom_a + 1
            next_a = 2 * (j + 1)
            H_k[atom_a, next_a + 1] = t * np.exp(1j * k)
            H_k[next_a + 1, atom_a] = t * np.exp(-1j * k)

    # Second Nearest Neighbor Hopping (2NN)
    for j in range(width_zigzag_lines - 1):
        atom_a = 2 * j
        atom_b = atom_a + 1
        next_a = 2 * (j + 1)
        H_k[atom_a, next_a] = t2 * np.exp(1j * k)
        H_k[next_a, atom_a] = t2 * np.exp(-1j * k)
        H_k[atom_b, next_a + 1] = t2 * np.exp(1j * k)
        H_k[next_a + 1, atom_b] = t2 * np.exp(-1j * k)

    # Third Nearest Neighbor Hopping (3NN)
    for j in range(width_zigzag_lines - 2):
        atom_a = 2 * j
        atom_b = atom_a + 1
        next_next_a = 2 * (j + 2)
        H_k[atom_a, next_next_a + 1] = t3 * np.exp(2j * k)
        H_k[next_next_a + 1, atom_a] = t3 * np.exp(-2j * k)
        H_k[atom_b, next_next_a] = t3 * np.exp(2j * k)
        H_k[next_next_a, atom_b] = t3 * np.exp(-2j * k)

    return H_k

def calculate_band_structure_zgnr(width_zigzag_lines, t=-2.7, t2=-0.1, t3=-0.05, num_k_points=100):
    """
    Calculates the band structure for a ZGNR by sweeping through k-points.
    """
    k_vals = np.linspace(-np.pi, np.pi, num_k_points)  # k points in the Brillouin zone
    eigenvalues = np.zeros((num_k_points, 2 * width_zigzag_lines))

    for i, k in enumerate(k_vals):
        H_k = build_zgnr_k_dependent_hamiltonian(width_zigzag_lines, t, t2, t3, k)
        eigenvalues[i, :] = np.linalg.eigvalsh(H_k)  # Compute eigenvalues

    return k_vals, eigenvalues

# Define parameters
width_dimer_lines = 14  # Width in dimer lines
width_zigzag_lines = 16
t = -2.7               # 1NN hopping parameter
t2 = -0.0              # 2NN hopping parameter
t3 = -0.0             # 3NN hopping parameter

# Calculate and plot the band structure
k_vals, eigenvalues = calculate_band_structure(width_dimer_lines, t, t2, t3)
plot_band_structure(k_vals, eigenvalues)
k_vals, eigenvalues = calculate_band_structure_zgnr(width_zigzag_lines, t, t2, t3)
plot_band_structure(k_vals, eigenvalues)
