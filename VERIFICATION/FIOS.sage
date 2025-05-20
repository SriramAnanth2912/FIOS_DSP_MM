import math
import random
from typing import Optional

"""
    This module provides the following functions:
    - FIOS: This is a python description of the FIOS block variant of the Montgomery Multiplication
method used to perform repeated modular multiplications with respect to a single modulus efficiently.
    - FIOS_modexp: This function is based on the FIOS function and uses the square and multiply algorithm
to compute a Montgomery modular exponentiation.

    These function exist for the sake of demonstration, verification and test vector generation. FIOS
is intended for operands of bit-width ranging from 128 to 4096 bits, while FIOS_modexp will hang for widths
greater than 25.

    Note that FIOS uses an additionnal bit as per C.D Walter's paper "Montgomery Exponentiation Needs No Final Subtraction"
in order to avoid performing the final subtraction at the end of the Montgomery Multiplication's algorithm.

    Example:
    >>> p = 0xe26e92fa5a9ee154051ceea3263af5a9
    >>> a = 0xdde7dcd1ae5f587d9ec2d7e499467dda
    >>> b = 0xa162473cb38ec289182937eb7fc94fae
    >>> WIDTH = p.bit_length()+1
    >>> w = 17
    >>> s = ((WIDTH-1)//w+1)
    >>> R = 2**(s*w)
    >>> R2_mod = R**2 % p
    >>> a_mgt = FIOS(a, R2_mod, p)
    >>> b_mgt = FIOS(b, R2_mod, p)
    >>> res_mgt = FIOS(a_mgt, b_mgt, p)
    >>> res = FIOS(res_mgt, 1, p)
    >>> verif = a*b % p 
    >>> res == verif
    >>> True
"""

def FIOS(a: int, b: int, p: int, w: int = 17, p_prime_0: Optional[int] = None) -> int:
    """
        Function: FIOS.
        Description: This function takes in two integers a and b as well as modulus p and computes
    the Montgomery Multiplication of a and b modulo p using the FIOS block variant description of
    the algorithm. It returns an integer less than 2*p and equal to a*b*R^(-1) modulo p where R is
    a power of two greater than 2*p.
        Arguments:
        - a: First operand of the Montgomery Multiplication.
        - b: Second operand of the Montgomery Multiplication.
        - p: Odd modulus (must be coprime with R).
        - w: bit-width of the blocks used to split arguments a, b and p.
        - p_prime_0: first block of the opposite of the modular inverse of R modulo p (p*p_prime_0 mod 2^w = -1 mod 2^w).
        Returns:
        - res: an integer less than 2*p such that res mod p = a*b*R^(-1) mod p.
    """

    if bin(p)[-1] == '0':
        raise ValueError("p must be an odd modulus.")

    WIDTH = p.bit_length()

    # WIDTH is taken to be the width of p and the operands plus 2
    # in order for intermediate results to be contained (in the case of modular exponentiation)
    # without having to perform the final subtraction of the Montgomery Multiplication
    WIDTH = WIDTH + 1

    if a.bit_length() > WIDTH:
        raise ValueError("Operand a should be no greater than 2*p.")
    if b.bit_length() > WIDTH:
        raise ValueError("Operand b should be no greater than 2*p.")
    
    # s is the number of blocks of width w required to slice operands
    s = (WIDTH-1)//w + 1
    
    W = 2**w

    if p_prime_0 is None:
        R = 2**(s*w)
        p_prime_0 = inverse_mod(-p, R) % W
    
    a_arr = [(a >> (i*w)) % W for i in range(s)]
    b_arr = [(b >> (i*w)) % W for i in range(s)]
    n_arr = [(p >> (i*w)) % W for i in range(s)]
    
    res_arr = s*[0]
    
    for i in range(s):
        
        # The outer loop scans the a operand.
        # the least significant block of the result is processed
        # and reduced at the beginning of each iterations
        
        res_arr[0] += a_arr[i]*b_arr[0]
        
        m = res_arr[0]*p_prime_0 % W
        
        res_arr[0] += m*n_arr[0]
        res_arr[0] = res_arr[0] >> w
        
        for j in range(1, s):
        
            # The inner loop scans the b operand.
            # The remaining blocks are processed in this loop.
        
            res_arr[j-1] += a_arr[i]*b_arr[j] + m*n_arr[j] + res_arr[j]
            
            res_arr[j] = res_arr[j-1] >> w
            res_arr[j-1] = res_arr[j-1] % W
            
    res = 0
    for i in range(len(res_arr)):
    
        res += res_arr[i] << (i*w)
    
    return res
    
def FIOS_modexp(a: int, b: int, p: int, w: int = 17, p_prime_0: Optional[int] = None) -> int:
    """
        Function: FIOS_modexp
        Description: This function uses the FIOS function to perform a Montgomery Modular Exponentiation
    in the Montgomery domain modulo p. It returns an integer less than 2*p and equal to (a^b)*R^(-b) modulo p.
    where R is a power of 2 greater than 2*p.
        This function is intended for demonstration purpose. It is not efficient.
        Arguments:
        - a: Operand base of the modular exponentiation.
        - b: Operand exponent of the modular exponentiation.
        - p: Odd modulus.
        - w: bit-width of the blocks used to split arguments a and p.
        - p_prime_0: first block of the opposite of the modular inverse of R modulo p (p*p_prime_0 mod 2^w = -1 mod 2^w).
        Returns:
        - res: an integer less than 2*p such that res mod p = (a^b)*R^(-b) modulo p.
    """
    if (b == 0):
        return 1
    else:
        res = a
        for digit in bin(b)[3:]:
            res = FIOS(res, res, p, w = w, p_prime_0 = p_prime_0)
            if (digit == '1'):
                res = FIOS(res, a, p, w = w, p_prime_0 = p_prime_0)
        return res
    
if __name__ == "__main__":

    # Running this file will test one random set of inputs fed to the FIOS function

    WIDTH_list = [2**(7+i) for i in range(6)]
    WIDTH = random.choice(WIDTH_list)

    p = random_prime(2**WIDTH, False, 2**(WIDTH-1))
    
    a = random.randrange(2**(WIDTH-1), p)
    b = random.randrange(2**(WIDTH-1), p)

    w = 17 
    W = 2**w
    
    s = ((WIDTH+1)-1)//w + 1

    R = 2**(s*w)
    R2_mod = R**2 % p
    R_inv = inverse_mod(R, p)

    p_prime_0 = inverse_mod(-p, R) % W

    a_mgt = FIOS(a, R2_mod, p, w = w, p_prime_0 = p_prime_0)
    b_mgt = FIOS(b, R2_mod, p, w = w, p_prime_0 = p_prime_0)
    
    verif = a*b % p
    res_mgt = FIOS(a_mgt, b_mgt, p, w = w, p_prime_0 = p_prime_0)
    res = FIOS(res_mgt, 1, p, w = w, p_prime_0 = p_prime_0)

    verif_string = (
    f"{WIDTH = }\n"
    f"p = {hex(p)}\n"
    f"a = {hex(a)}\n"
    f"b = {hex(b)}\n"
     "\n"
    f"res   = {hex(res)}\n"
    f"verif = {hex(verif)}\n"
    f"match = {res == verif}\n" )

    print(verif_string)
