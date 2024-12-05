import numpy as np
import time

def create_tridiagonal_matrix(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = 4
        if i > 0:
            A[i,i-1] = -1
        if i < n-1:
            A[i,i+1] = -3
    return A

def compare_methods(n):
    A = create_tridiagonal_matrix(n)
    x_exact = np.ones(n)
    b = A @ x_exact

    # Thomas method timing
    start = time.time()
    x_thomas = thomas_solve([-1]*(n-1), [4]*n, [-3]*(n-1), b)
    thomas_time = time.time() - start

    # Gaussian elimination timing
    start = time.time()
    x_gaussian = np.linalg.solve(A, b)
    gaussian_time = time.time() - start

    return thomas_time, gaussian_time