import numpy as np

def print_step(msg, matrix=None, indent=0):
    """Print step with optional matrix"""
    print("\n" + " " * indent + msg)
    if matrix is not None:
        print("\n".join([" " * indent + "  ".join(f"{x:12.6f}" for x in row) for row in matrix]))

def jordan_solve_with_steps(A, b, name=""):
    """Solve system using Jordan method with partial pivoting, showing all steps"""
    n = len(A)
    # Create augmented matrix [A|b]
    augmented = np.concatenate((A, b.reshape(-1,1)), axis=1)

    print(f"\n{'='*80}\nSolving {name}\n{'='*80}")
    print_step("Initial matrix A:", A)
    print_step("Vector b:", b.reshape(-1,1))
    print_step("Augmented matrix [A|b]:", augmented)

    for i in range(n):
        print(f"\nStep {i+1}: Processing column {i+1}")

        # Partial pivoting
        pivot_row = i + np.argmax(abs(augmented[i:, i]))
        if pivot_row != i:
            print(f"\nPivot selection:")
            print(f"Max element in column {i+1}: |a_{pivot_row+1}{i+1}| = {abs(augmented[pivot_row,i]):.6f}")
            print(f"Swapping rows {i+1} and {pivot_row+1}")
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            print_step("After row swap:", augmented, indent=2)

        # Normalize pivot row
        pivot = augmented[i,i]
        print(f"\nNormalizing row {i+1} (dividing by {pivot:.6f})")
        augmented[i] = augmented[i] / pivot
        print_step("After normalization:", augmented, indent=2)

        # Eliminate column entries
        for j in range(n):
            if i != j:
                multiplier = augmented[j,i]
                if abs(multiplier) > 1e-10:
                    print(f"\nEliminating entry in row {j+1}:")
                    print(f"multiplier = L_{j+1}{i+1} = {multiplier:.6f}")
                    print(f"R_{j+1} = R_{j+1} - ({multiplier:.6f})R_{i+1}")
                    augmented[j] = augmented[j] - multiplier * augmented[i]
                    print_step("After elimination:", augmented, indent=2)

    return augmented[:, -1]

def compute_metrics(A, x_computed, x_exact, b, name=""):
    """Compute and display error metrics"""
    print(f"\nMetrics for {name}:")
    print("-" * 40)

    # Error calculation
    error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
    print(f"Error calculation:")
    print(f"‖δx‖∞ = {np.max(np.abs(x_computed - x_exact)):.6e}")
    print(f"‖x‖∞ = {np.max(np.abs(x_exact)):.6e}")
    print(f"Error = {error:.6e}")

    # Residual calculation
    residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
    print(f"\nResidual calculation:")
    print(f"‖δr‖∞ = {np.max(np.abs(A @ x_computed - b)):.6e}")
    print(f"‖b‖∞ = {np.max(np.abs(b)):.6e}")
    print(f"Residual = {residual:.6e}")

    # Condition number
    cond = np.linalg.cond(A, np.inf)
    print(f"\nCondition number:")
    print(f"κ(A) = {cond:.6e}")

    return error, residual, cond

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
        if i < 7:
            A3[i,i+1] = -3

    results = []
    for i, (A, name) in enumerate([
        (A1, "Application 1 (Given 8x8 Matrix)"),
        (A2, "Application 2 (Hilbert Matrix)"),
        (A3, "Application 3 (Tridiagonal Symmetric Matrix)")
    ], 1):
        b = A @ x_exact
        x_computed = jordan_solve_with_steps(A, b, name)
        metrics = compute_metrics(A, x_computed, x_exact, b, name)
        results.append((i, *metrics))

    # Print final results table
    print("\n\nFinal Results Table")
    print("=" * 80)
    print(f"{'Application':<12} {'Error':<20} {'Residual':<20} {'Condition Number':<20}")
    print("-" * 80)
    for app, err, res, cond in results:
        print(f"{app:<12} {err:<20.6e} {res:<20.6e} {cond:<20.6e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
