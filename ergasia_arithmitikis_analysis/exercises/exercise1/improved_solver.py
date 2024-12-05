import numpy as np

def print_matrix(A, label=""):
    """Print matrix with aligned columns"""
    print(f"\n{label}")
    print('\n'.join(['  '.join(f'{x:12.6f}' for x in row) for row in A]))

def jordan_solve(A, b, application_name):
    """Solve system using Jordan method with partial pivoting"""
    n = len(A)
    # Work with copies to preserve original matrices
    A_work = A.copy()
    b_work = b.copy()

    # Create augmented matrix [A|b]
    augmented = np.concatenate((A_work, b_work.reshape(-1,1)), axis=1)

    print(f"\n{'='*80}")
    print(f"Solving {application_name}")
    print(f"{'='*80}")

    print_matrix(A_work, "Original Matrix A:")
    print("\nOriginal vector b:")
    print(b_work)
    print_matrix(augmented, "\nAugmented matrix [A|b]:")

    # Initialize permutation vector for row swaps
    p = np.arange(n)

    for k in range(n):
        # Find pivot
        pivot_idx = k + np.argmax(np.abs(augmented[k:, k]))
        if pivot_idx != k:
            # Swap rows
            augmented[[k, pivot_idx]] = augmented[[pivot_idx, k]]
            p[[k, pivot_idx]] = p[[pivot_idx, k]]
            print(f"\nPivot step {k+1}:")
            print(f"Swapping rows {k+1} and {pivot_idx+1}")
            print(f"Pivot element: {augmented[k,k]:.6e}")
            print_matrix(augmented, "After row swap:")

        # Check for numerical stability
        if np.abs(augmented[k,k]) < 1e-10:
            raise ValueError(f"Small pivot encountered: {augmented[k,k]:.6e}")

        # Normalize row k
        pivot = augmented[k,k]
        augmented[k] = augmented[k] / pivot
        print(f"\nNormalization step {k+1}:")
        print(f"Dividing row {k+1} by {pivot:.6e}")
        print_matrix(augmented, "After normalization:")

        # Eliminate in all other rows
        for i in range(n):
            if i != k:
                multiplier = augmented[i,k]
                if abs(multiplier) > 1e-10:
                    print(f"\nElimination step {k+1}.{i+1}:")
                    print(f"L_{i+1}{k+1} = {multiplier:.6e}")
                    augmented[i] = augmented[i] - multiplier * augmented[k]
                    print_matrix(augmented, "After elimination:")

    # Extract solution
    x = augmented[:, -1]
    return x

def compute_metrics(A, x_computed, x_exact, b, name=""):
    """Compute and display error metrics"""
    print(f"\nMetrics for {name}:")
    print("=" * 40)

    # Compute actual residual
    r = b - A @ x_computed

    # Error in solution
    abs_error = np.abs(x_computed - x_exact)
    rel_error = np.max(abs_error) / np.max(np.abs(x_exact))

    # Residual
    abs_residual = np.abs(r)
    rel_residual = np.max(abs_residual) / np.max(np.abs(b))

    # Condition number
    cond = np.linalg.cond(A, np.inf)

    print("\nDetailed Error Analysis:")
    print(f"Maximum absolute error: {np.max(abs_error):.6e}")
    print(f"Maximum relative error: {rel_error:.6e}")
    print(f"Component-wise absolute errors:")
    for i, err in enumerate(abs_error):
        print(f"  x_{i+1}: {err:.6e}")

    print("\nDetailed Residual Analysis:")
    print(f"Maximum absolute residual: {np.max(abs_residual):.6e}")
    print(f"Maximum relative residual: {rel_residual:.6e}")
    print(f"Component-wise residuals:")
    for i, res in enumerate(abs_residual):
        print(f"  r_{i+1}: {res:.6e}")

    print(f"\nCondition Number Analysis:")
    print(f"κ(A) = {cond:.6e}")

    return rel_error, rel_residual, cond

def main():
    # Predetermined solution
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

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

    # Application 2: Hilbert matrix
    A2 = np.array([[1/(i+j-1) for j in range(1,9)] for i in range(1,9)])

    # Application 3: Tridiagonal symmetric matrix
    A3 = np.zeros((8,8))
    for i in range(8):
        A3[i,i] = 4
        if i > 0:
            A3[i,i-1] = -1
            A3[i-1,i] = -3

    results = []
    for i, (A, name) in enumerate([
        (A1, "Application 1 (Given 8x8 Matrix)"),
        (A2, "Application 2 (Hilbert Matrix)"),
        (A3, "Application 3 (Tridiagonal Symmetric Matrix)")
    ], 1):
        print(f"\nProcessing {name}")
        print("=" * 80)

        # Compute b = Ax
        b = A @ x_exact

        try:
            # Solve system
            x_computed = jordan_solve(A, b, name)

            # Compute metrics
            metrics = compute_metrics(A, x_computed, x_exact, b, name)
            results.append((i, *metrics))

        except ValueError as e:
            print(f"Error solving {name}: {e}")
            results.append((i, np.inf, np.inf, np.inf))

    # Print final results table
    print("\nFinal Results Table")
    print("=" * 80)
    print(f"{'Application':<12} {'Error':<20} {'Residual':<20} {'Condition Number':<20}")
    print("-" * 80)
    for app, err, res, cond in results:
        print(f"{app:<12} {err:<20.6e} {res:<20.6e} {cond:<20.6e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
