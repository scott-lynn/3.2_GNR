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


##########################################
# Main Execution Block
##########################################
def main():

    # Define params
    N: int = 2 
    a: float = 1 
    t1: float = 2.7 # eV

    # Set k-space range
    k_start: float = -np.pi/a 
    k_end: float = np.pi/a
    k_points: int = 200 
    k = np.linspace(k_start, k_end, k_points)

    phase = np.exp(1j * k * a)

    H = np.zeros(N,N)

    #LOOP THROUGH K VALUES

    # Loop 1->N : BOTTOM ROW
    for i in range(1, N, 1):
        # Edge cases
        # LHS
        if i == 1:
            # intra
            H[i, i+1] = -t1 
            H[i, i+N] = -t1
        # RHS
        if i == N:
            # intra
            H[i, i-1] = -t1
            H[i, i+N] = -t1 

        # Odd cases
        if is_odd(i): 
            # intra
            H[i,i-1] = -t1 
            H[i, i+1] = -t1 
            H[i, i+N] = -t1 

        # Even cases
        if is_even(i):
            # intra
            H[i, i-1] = -t1 
            H[i, i+1] = -t1 
            # inter
            H[i, i+N] = -t1 * phase

    # Loop N+1->2N : TOP ROW
    for i in range(N+1, 2*N, 1):
        # Edge cases
        # LHS
        if i == N+1:
            # intra
            H[i, N+1] = -t1 
            H[i, i-N] = -t1 
        # RHS
        if i == 2*N: 
            # intra
            H[i, i-1] = -t1 
            H[i, i-N] = -t1 

        # Odd cases
        if is_odd(i): 
            # intra
            H[i,i-1] = -t1 
            H[i, i+1] = -t1 
            H[i, i+N] = -t1 

        # Even cases
        if is_even(i):
            # intra
            H[i, i-1] = -t1 
            H[i, i+1] = -t1 
            # inter
            H[i, i+N] = -t1 * (-phase)



    