import numpy as np

def print_matrix(A, label=""):
    """Print matrix with aligned columns"""
    print(f"\n{label}")
    print('\n'.join(['  '.join(f'{x:12.6f}' for x in row) for row in A]))

def solve_jordan_method():
    # Step 1: Initialize the given 8x8 matrix A
    A = np.array([
        [10, -2, -1, 2, 3, 1, -4, 7],
        [5, 11, 3, 10, -3, 3, 3, -4],
        [7, 12, 1, 5, 3, -12, 2, 3],
        [8, 7, -2, 1, 3, 2, 2, 4],
        [2, -13, -1, 1, 4, -1, 8, 3],
        [4, 2, 9, 1, 12, -1, 4, 1],
        [-1, 4, -7, -1, 1, 1, -1, -3],
        [-1, 3, 4, 1, 3, -4, 7, 6]
    ])

    # Step 2: Define predetermined solution x
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    print("Step 1: Initial Matrix A")
    print_matrix(A)
    print("\nPredetermined solution x:")
    print(x_exact)

    # Step 3: Compute b = Ax
    b = A @ x_exact
    print("\nStep 2: Computed b = Ax")
    print(b)

    # Step 4: Create augmented matrix [A|b]
    n = len(A)
    augmented = np.concatenate((A.copy(), b.reshape(-1,1)), axis=1)
    print("\nStep 3: Augmented matrix [A|b]")
    print_matrix(augmented)

    # Step 5: Jordan elimination with partial pivoting
    for i in range(n):
        print(f"\nStep {i+1} of elimination:")

        # Partial pivoting
        pivot_row = i + np.argmax(abs(augmented[i:, i]))
        if pivot_row != i:
            print(f"\nPivoting: Swap rows {i+1} and {pivot_row+1}")
            print(f"Max element in column {i+1}: |a_{pivot_row+1}{i+1}| = {abs(augmented[pivot_row,i]):.6f}")
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            print_matrix(augmented, "After row swap:")

        # Normalize pivot row
        pivot = augmented[i,i]
        print(f"\nNormalize row {i+1} (divide by {pivot:.6f})")
        augmented[i] = augmented[i] / pivot
        print_matrix(augmented, "After normalization:")

        # Eliminate column entries
        for j in range(n):
            if i != j:
                multiplier = augmented[j,i]
                if abs(multiplier) > 1e-10:
                    print(f"\nEliminate entry in row {j+1}")
                    print(f"multiplier L_{j+1}{i+1} = {multiplier:.6f}")
                    augmented[j] = augmented[j] - multiplier * augmented[i]
                    print_matrix(augmented, "After elimination:")

    # Step 6: Extract solution
    x_computed = augmented[:, -1]
    print("\nStep 4: Final Solution")
    print("Computed solution x:")
    print(x_computed)

    # Step 7: Calculate metrics
    error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
    residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
    condition_number = np.linalg.cond(A, np.inf)

    print("\nStep 5: Error Analysis")
    print(f"Error (‖δx‖∞/‖x‖∞):")
    print(f"  Numerator ‖δx‖∞ = {np.max(np.abs(x_computed - x_exact)):.6e}")
    print(f"  Denominator ‖x‖∞ = {np.max(np.abs(x_exact)):.6e}")
    print(f"  Final Error = {error:.6e}")

    print(f"\nResidual (‖δr‖∞/‖x‖∞):")
    print(f"  Numerator ‖δr‖∞ = {np.max(np.abs(A @ x_computed - b)):.6e}")
    print(f"  Denominator ‖x‖∞ = {np.max(np.abs(x_exact)):.6e}")
    print(f"  Final Residual = {residual:.6e}")

    print(f"\nCondition Number:")
    print(f"  κ(A) = {condition_number:.6e}")

    return error, residual, condition_number

if __name__ == "__main__":
    solve_jordan_method()
