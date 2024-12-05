import numpy as np

def print_matrix(A, label=""):
    """Print matrix with aligned columns"""
    print(f"\n{label}")
    print('\n'.join(['  '.join(f'{x:12.6f}' for x in row) for row in A]))

def jordan_solve(A, b, application_name):
    """Solve system using Jordan method with partial pivoting"""
    n = len(A)
    # Create augmented matrix [A|b]
    augmented = np.concatenate((A.copy(), b.reshape(-1,1)), axis=1)

    print(f"\n{'='*80}")
    print(f"Solving {application_name}")
    print(f"{'='*80}")
    print_matrix(augmented, "Initial augmented matrix [A|b]:")

    for i in range(n):
        # Partial pivoting
        pivot_row = i + np.argmax(abs(augmented[i:, i]))
        if pivot_row != i:
            print(f"\nPivoting step {i+1}:")
            print(f"Max element in column {i+1}: |a_{pivot_row+1}{i+1}| = {abs(augmented[pivot_row,i]):.6f}")
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            print_matrix(augmented, "After row swap:")

        # Normalize pivot row
        pivot = augmented[i,i]
        print(f"\nNormalization step {i+1}:")
        print(f"Divide row {i+1} by pivot = {pivot:.6f}")
        augmented[i] = augmented[i] / pivot
        print_matrix(augmented, "After normalization:")

        # Eliminate column entries
        for j in range(n):
            if i != j:
                multiplier = augmented[j,i]
                if abs(multiplier) > 1e-10:
                    print(f"\nElimination step {i+1}.{j+1}:")
                    print(f"L_{j+1}{i+1} = {multiplier:.6f}")
                    print(f"R_{j+1} = R_{j+1} - ({multiplier:.6f})R_{i+1}")
                    augmented[j] = augmented[j] - multiplier * augmented[i]
                    print_matrix(augmented, "After elimination:")

    return augmented[:, -1]

def compute_metrics(A, x_computed, x_exact, b):
    """Compute error metrics"""
    error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
    residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
    condition_number = np.linalg.cond(A, np.inf)

    print("\nError Analysis:")
    print(f"‖δx‖∞ = {np.max(np.abs(x_computed - x_exact)):.6e}")
    print(f"‖x‖∞ = {np.max(np.abs(x_exact)):.6e}")
    print(f"Error = {error:.6e}")

    print("\nResidual Analysis:")
    print(f"‖δr‖∞ = {np.max(np.abs(A @ x_computed - b)):.6e}")
    print(f"‖b‖∞ = {np.max(np.abs(b)):.6e}")
    print(f"Residual = {residual:.6e}")

    print(f"\nCondition Number:")
    print(f"κ(A) = {condition_number:.6e}")

    return error, residual, condition_number

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
        print("\nMatrix A:")
        print_matrix(A)
        print("\nPredetermined solution x:")
        print(x_exact)
        print("\nComputed b = Ax:")
        print(b)

        # Solve system
        x_computed = jordan_solve(A, b, name)
        print("\nComputed solution:")
        print(x_computed)

        # Compute metrics
        metrics = compute_metrics(A, x_computed, x_exact, b)
        results.append((i, *metrics))

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
