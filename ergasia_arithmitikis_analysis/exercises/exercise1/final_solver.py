import numpy as np

def print_matrix(A, label="", precision=6):
    """Print matrix with aligned columns and specified precision.

    Args:
        A: numpy array to print
        label: Optional label to print before matrix
        precision: Number of decimal places to show
    """
    print(f"\n{label}")
    format_str = f"{{:>{precision+6}.{precision}e}}"
    print('\n'.join(['  '.join(format_str.format(x) for x in row) for row in A]))

def jordan_solve(A, b, application_name, tol=1e-14):
    """Solve linear system using Jordan method with partial pivoting.

    Implementation follows these steps:
    1. Create augmented matrix [A|b]
    2. For each column k:
       a. Find maximum element in column k (partial pivoting)
       b. Swap rows if necessary
       c. Normalize pivot row
       d. Eliminate in all other rows
    3. Extract and verify solution

    Args:
        A: Coefficient matrix (n×n)
        b: Right-hand side vector
        application_name: Name of current application for logging
        tol: Tolerance for numerical stability checks

    Returns:
        x: Solution vector

    Raises:
        ValueError: If matrix is numerically singular
    """
    n = len(A)
    # Create augmented matrix [A|b] using high precision
    augmented = np.concatenate((A.astype(np.float64).copy(),
                              b.astype(np.float64).reshape(-1,1)), axis=1)

    print(f"\n{'='*80}")
    print(f"Solving {application_name}")
    print(f"{'='*80}")

    # Store original matrix for verification
    A_orig = A.copy()
    b_orig = b.copy()

    for k in range(n):
        # Step 1: Find maximum element in column k (partial pivoting)
        pivot_idx = k + np.argmax(np.abs(augmented[k:, k]))
        max_val = np.abs(augmented[pivot_idx, k])

        # Check for numerical stability
        if max_val < tol:
            raise ValueError(f"Matrix is numerically singular, max value = {max_val}")

        # Step 2: Row swap if necessary
        if pivot_idx != k:
            print(f"\nPivot step {k+1}:")
            print(f"Swapping rows {k+1} and {pivot_idx+1}")
            print(f"Pivot element: {augmented[pivot_idx,k]:.6e}")
            augmented[[k, pivot_idx]] = augmented[[pivot_idx, k]]
            print_matrix(augmented, "After row swap:")

        # Step 3: Normalize pivot row
        pivot = augmented[k,k]
        augmented[k] = augmented[k] / pivot
        print(f"\nNormalization step {k+1}:")
        print(f"Dividing row {k+1} by {pivot:.6e}")
        print_matrix(augmented, "After normalization:")

        # Step 4: Eliminate in all other rows
        for i in range(n):
            if i != k:
                multiplier = augmented[i,k]
                if abs(multiplier) > tol:
                    augmented[i] = augmented[i] - multiplier * augmented[k]
                    print(f"\nElimination step {k+1}.{i+1}:")
                    print(f"Eliminating with multiplier {multiplier:.6e}")
                    print_matrix(augmented, "After elimination:")

    # Extract solution
    x = augmented[:, -1]

    # Verify solution
    computed_b = A_orig @ x
    max_residual = np.max(np.abs(computed_b - b_orig))
    if max_residual > tol * 100:
        print(f"\nWarning: Large residual detected: {max_residual:.6e}")

    return x

def compute_metrics(A, x_computed, x_exact, b, name=""):
    """Compute and display detailed error metrics for the solution.

    Calculates three key metrics:
    1. Relative Error: ‖δx‖∞/‖x‖∞
    2. Relative Residual: ‖δr‖∞/‖x‖∞
    3. Condition Number: κ(A) = ‖A‖∞ · ‖A^{-1}‖∞

    Args:
        A: Original coefficient matrix
        x_computed: Computed solution vector
        x_exact: Known exact solution
        b: Original right-hand side vector
        name: Application name for logging

    Returns:
        tuple: (relative_error, relative_residual, condition_number)
    """
    print(f"\nMetrics for {name}")
    print("=" * 60)

    # Compute residual vector r = b - Ax
    r = b - A @ x_computed

    # Compute infinity norms
    x_norm = np.max(np.abs(x_exact))
    b_norm = np.max(np.abs(b))

    # Compute component-wise and relative errors
    abs_error = np.abs(x_computed - x_exact)
    rel_error = np.max(abs_error) / x_norm

    # Compute component-wise and relative residuals
    abs_residual = np.abs(r)
    rel_residual = np.max(abs_residual) / b_norm

    # Compute condition number using infinity norm
    cond = np.linalg.cond(A, np.inf)

    # Display detailed results
    print("\nSolution verification:")
    print("Exact solution:     ", " ".join(f"{x:10.6f}" for x in x_exact))
    print("Computed solution:  ", " ".join(f"{x:10.6f}" for x in x_computed))
    print("Component-wise error:", " ".join(f"{x:10.6e}" for x in abs_error))

    print(f"\nError Analysis:")
    print(f"Maximum absolute error: {np.max(abs_error):.6e}")
    print(f"Relative error (∞-norm): {rel_error:.6e}")

    print(f"\nResidual Analysis:")
    print(f"Maximum absolute residual: {np.max(abs_residual):.6e}")
    print(f"Relative residual (∞-norm): {rel_residual:.6e}")

    print(f"\nCondition Number Analysis:")
    print(f"Matrix norm ‖A‖∞: {np.linalg.norm(A, np.inf):.6e}")
    print(f"Inverse norm ‖A^{-1}‖∞: {np.linalg.norm(np.linalg.inv(A), np.inf):.6e}")
    print(f"Condition number κ(A): {cond:.6e}")

    return rel_error, rel_residual, cond

def main():
    """Main execution function for solving all three applications.

    Solves linear systems for:
    1. Given 8×8 matrix
    2. Hilbert matrix
    3. Tridiagonal symmetric matrix

    Uses predetermined solution x = (-1, 1, -1, 1, -1, 1, -1, 1)^T
    """
    # Set higher precision for calculations
    np.set_printoptions(precision=15)

    # Predetermined solution vector
    x_exact = np.array([-1., 1., -1., 1., -1., 1., -1., 1.])

    # Define matrices for each application
    A1 = np.array([
        [10., -2., -1., 2., 3., 1., -4., 7.],
        [5., 11., 3., 10., -3., 3., 3., -4.],
        [7., 12., 1., 5., 3., -12., 2., 3.],
        [8., 7., -2., 1., 3., 2., 2., 4.],
        [2., -13., -1., 1., 4., -1., 8., 3.],
        [4., 2., 9., 1., 12., -1., 4., 1.],
        [-1., 4., -7., -1., 1., 1., -1., -3.],
        [-1., 3., 4., 1., 3., -4., 7., 6.]
    ])

    A2 = np.array([[1./(i+j-1) for j in range(1,9)] for i in range(1,9)])

    A3 = np.zeros((8,8))
    for i in range(8):
        A3[i,i] = 4.
        if i > 0:
            A3[i,i-1] = -1.
            A3[i-1,i] = -3.

    results = []
    for i, (A, name) in enumerate([
        (A1, "Application 1 (Given 8x8 Matrix)"),
        (A2, "Application 2 (Hilbert Matrix)"),
        (A3, "Application 3 (Tridiagonal Symmetric Matrix)")
    ], 1):
        print(f"\nProcessing {name}")
        print("=" * 80)

        try:
            # Compute b = Ax using high precision
            b = A @ x_exact

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
