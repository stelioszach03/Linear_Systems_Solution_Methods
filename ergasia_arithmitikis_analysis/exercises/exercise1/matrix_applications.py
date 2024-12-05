import numpy as np

def create_matrix_1():
    """Create the given 8x8 matrix from Application 1"""
    return np.array([
        [10, -2, -1, 2, 3, 1, -4, 7],
        [5, 11, 3, 10, -3, 3, 3, -4],
        [7, 12, 1, 5, 3, -12, 2, 3],
        [8, 7, -2, 1, 3, 2, 2, 4],
        [2, -13, -1, 1, 4, -1, 8, 3],
        [4, 2, 9, 1, 12, -1, 4, 1],
        [-1, 4, -7, -1, 1, 1, -1, -3],
        [-1, 3, 4, 1, 3, -4, 7, 6]
    ])

def create_matrix_2(n=8):
    """Create Hilbert matrix for Application 2"""
    return np.array([[1/(i+j-1) for j in range(1,n+1)] for i in range(1,n+1)])

def create_matrix_3(n=8):
    """Create tridiagonal symmetric matrix for Application 3"""
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = 4  # Main diagonal
        if i > 0:
            A[i,i-1] = -1  # Lower subdiagonal
        if i < n-1:
            A[i,i+1] = -3  # Upper subdiagonal
    return A

def jordan_solve(A, b, output_file=None):
    """Jordan method with partial pivoting, showing detailed steps"""
    n = len(A)
    # Create augmented matrix [A|b]
    augmented = np.concatenate((A, b.reshape(-1,1)), axis=1).astype(np.float64)
    steps = []

    def log_step(msg, matrix=None):
        if output_file:
            steps.append(f"{msg}\n")
            if matrix is not None:
                steps.append("Matrix state:\n")
                steps.append('\n'.join(['  '.join(f'{x:10.6f}' for x in row) for row in matrix]))
                steps.append("\n\n")

    log_step("Initial augmented matrix [A|b]:", augmented)

    for i in range(n):
        # Partial pivoting
        pivot_row = i + np.argmax(abs(augmented[i:, i]))
        if pivot_row != i:
            log_step(f"Step {i+1}.1: Swap rows {i+1} and {pivot_row+1} for partial pivoting")
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            log_step("After row swap:", augmented)

        # Normalize pivot row
        pivot = augmented[i,i]
        log_step(f"Step {i+1}.2: Normalize row {i+1} (divide by {pivot:.6f})")
        augmented[i] = augmented[i] / pivot
        log_step("After normalization:", augmented)

        # Eliminate column entries
        for j in range(n):
            if i != j:
                multiplier = augmented[j,i]
                if abs(multiplier) > 1e-10:
                    log_step(f"Step {i+1}.3: R{j+1} = R{j+1} - ({multiplier:.6f})R{i+1}")
                    augmented[j] = augmented[j] - multiplier * augmented[i]
                    log_step("After elimination:", augmented)

    if output_file:
        with open(output_file, 'w') as f:
            f.writelines(steps)

    return augmented[:, -1]

def compute_metrics(A, x_computed, x_exact, b):
    """Compute error, residual, and condition number with details"""
    error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
    residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
    condition_number = np.linalg.cond(A, np.inf)
    return error, residual, condition_number

def main():
    # Predetermined solution
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    # Process all three applications
    applications = [
        (create_matrix_1(), "Application 1 (Given 8x8 Matrix)"),
        (create_matrix_2(), "Application 2 (Hilbert Matrix)"),
        (create_matrix_3(), "Application 3 (Tridiagonal Symmetric Matrix)")
    ]

    results = []
    for i, (A, desc) in enumerate(applications, 1):
        print(f"\nProcessing {desc}")
        print("=" * 50)

        # Compute b = Ax
        b = A @ x_exact
        print(f"Computing b = Ax for predetermined solution x")

        # Solve system
        output_file = f"application{i}_detailed.txt"
        x_computed = jordan_solve(A, b, output_file)

        # Compute metrics
        error, residual, cond = compute_metrics(A, x_computed, x_exact, b)
        results.append({
            'Application': i,
            'Error': error,
            'Residual': residual,
            'Condition_Number': cond
        })

        print(f"Solution verification:")
        print(f"Error: {error:.6e}")
        print(f"Residual: {residual:.6e}")
        print(f"Condition Number: {cond:.6e}\n")

    # Print final results table
    print("\nFinal Results Table")
    print("=" * 80)
    print(f"{'Application':<12} {'Error':<20} {'Residual':<20} {'Condition Number':<20}")
    print("-" * 80)
    for r in results:
        print(f"{r['Application']:<12} {r['Error']:<20.6e} {r['Residual']:<20.6e} {r['Condition_Number']:<20.6e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
