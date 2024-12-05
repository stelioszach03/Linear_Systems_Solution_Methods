import numpy as np

def format_matrix(A):
    """Format matrix for pretty printing with aligned columns"""
    return '\n'.join(['  '.join(f'{x:12.6f}' for x in row) for row in A])

def format_vector(v):
    """Format vector for pretty printing"""
    return '\n'.join([f'{x:12.6f}' for x in v])

def find_max_pivot(A, k, n):
    """Find maximum pivot element in column k"""
    max_val = abs(A[k,k])
    max_idx = k
    for i in range(k+1, n):
        if abs(A[i,k]) > max_val:
            max_val = abs(A[i,k])
            max_idx = i
    return max_idx

def solve_system_with_details(A, b, output_file):
    """Solve system using Jordan method with partial pivoting, showing all steps"""
    n = len(A)
    # Create augmented matrix [A|b]
    augmented = np.concatenate((A, b.reshape(-1,1)), axis=1)
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    with open(output_file, 'w') as f:
        f.write("JORDAN METHOD WITH PARTIAL PIVOTING - DETAILED SOLUTION\n")
        f.write("="*80 + "\n\n")

        f.write("Original Matrix A:\n")
        f.write(format_matrix(A) + "\n\n")

        f.write("Vector b = Ax where x = [-1,1,-1,1,-1,1,-1,1]^T:\n")
        f.write(format_vector(b) + "\n\n")

        f.write("Initial augmented matrix [A|b]:\n")
        f.write(format_matrix(augmented) + "\n\n")

        for i in range(n):
            f.write(f"STEP {i+1}: Processing column {i+1}\n")
            f.write("-"*80 + "\n\n")

            # Partial pivoting
            pivot_row = find_max_pivot(augmented, i, n)
            if pivot_row != i:
                f.write(f"Pivot Selection: |a_{pivot_row+1}{i+1}| = {abs(augmented[pivot_row,i]):.6f} > "
                       f"|a_{i+1}{i+1}| = {abs(augmented[i,i]):.6f}\n")
                f.write(f"Swap rows {i+1} and {pivot_row+1}\n")
                augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
                f.write("Matrix after row swap:\n")
                f.write(format_matrix(augmented) + "\n\n")
            else:
                f.write(f"No row swap needed: pivot |a_{i+1}{i+1}| = {abs(augmented[i,i]):.6f} "
                       "is already maximum in column\n\n")

            # Normalize pivot row
            pivot = augmented[i,i]
            f.write(f"Normalize row {i+1} (divide by {pivot:.6f})\n")
            augmented[i] = augmented[i] / pivot
            f.write("Matrix after normalization:\n")
            f.write(format_matrix(augmented) + "\n\n")

            # Eliminate column entries
            for j in range(n):
                if i != j:
                    multiplier = augmented[j,i]
                    if abs(multiplier) > 1e-10:  # Only show significant eliminations
                        f.write(f"Eliminate entry in row {j+1}:\n")
                        f.write(f"multiplier = a_{j+1}{i+1} = {multiplier:.6f}\n")
                        f.write(f"R{j+1} = R{j+1} - ({multiplier:.6f})R{i+1}\n")
                        augmented[j] = augmented[j] - multiplier * augmented[i]
                        f.write("Matrix after elimination:\n")
                        f.write(format_matrix(augmented) + "\n\n")

        f.write("="*80 + "\n")
        f.write("SOLUTION VERIFICATION\n")
        f.write("="*80 + "\n\n")

        x_computed = augmented[:, -1]
        f.write("Computed solution x:\n")
        f.write(format_vector(x_computed) + "\n\n")

        f.write("Exact solution x:\n")
        f.write(format_vector(x_exact) + "\n\n")

        # Compute error metrics
        error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
        residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
        condition_number = np.linalg.cond(A, np.inf)

        f.write("ERROR ANALYSIS:\n")
        f.write(f"Error (‖δx‖∞/‖x‖∞): {error:.6e}\n")
        f.write(f"Residual (‖δr‖∞/‖x‖∞): {residual:.6e}\n")
        f.write(f"Condition Number κ(A): {condition_number:.6e}\n")

        return x_computed

def main():
    # Application 1: Given 8x8 matrix
    A1 = np.array([
        [10, -2, -1, 2, 3, 1, -4, 7],
        [5, 11, 3, 10, -3, 3, 3, -4],
        [7, 12, 1, 5, 3, -12, 2, 3],
        [8, 7, -2, 1, 3, 2, 2, 4],
        [2, -13, -1, 1, 4, -1, 8, 3],
        [4, 2, 9, 1, 12, -1, 4, 1],
        [-1, 4, -7, -1, 1, 1, -1, -3],
        [-1, 3, 4, 1, 3, -4, 7, 6]
    ])

    # Predetermined solution
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    # Compute b = Ax
    b1 = A1 @ x_exact

    # Solve system with detailed steps
    solve_system_with_details(A1, b1, 'application1_details.txt')

    # Application 2: Hilbert matrix
    A2 = np.array([[1/(i+j-1) for j in range(1,9)] for i in range(1,9)])
    b2 = A2 @ x_exact
    solve_system_with_details(A2, b2, 'application2_details.txt')

    # Application 3: Tridiagonal symmetric matrix
    A3 = np.zeros((8,8))
    for i in range(8):
        A3[i,i] = 4
        if i > 0:
            A3[i,i-1] = -1
        if i < 7:
            A3[i,i+1] = -3
    b3 = A3 @ x_exact
    solve_system_with_details(A3, b3, 'application3_details.txt')

if __name__ == "__main__":
    main()
