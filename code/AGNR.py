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
# Main Execution Block
##########################################
def main():

    # Define params
    N: int = 2 
    a: float = 1 
    t1: float = 2.7 # eV

    # Set k-space range
    k_start: float= -np.pi/a 
    k_end: float = np.pi/a
    k_points: int = 200 
    k = np.linspace(k_start, k_end, k_points)


    phase = np.exp(1j * k * a)

# Loop 1->N BOTTOM ROW
for i in range(1, N, 1):
    # Edge Cases
    if i == 1:
        H(i, i+1) = -t 
        H(i, i+N) = -t
    if i == N:
        H(i, i-1) = -t 
        H(i, i+N) = -t 

    # Odd Cases
    if is_odd(i): 
        H(i,i-1) = -t 
        H(i, i+1) = -t 
        H(i, i+N) = -t 

    # Even Case
    if is_even(i):
        H(i, i-1) = -t 
        H(i, i+1) = -t 
        H(i, i+N) = -t