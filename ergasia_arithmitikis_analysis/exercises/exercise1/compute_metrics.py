import numpy as np

def compute_error(x_computed, x_exact):
    return np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))

def compute_residual(A, x_computed, b):
    return np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))

def compute_condition_number(A):
    return np.linalg.cond(A, np.inf)