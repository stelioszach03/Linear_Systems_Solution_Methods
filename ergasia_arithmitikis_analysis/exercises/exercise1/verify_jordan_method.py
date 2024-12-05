import numpy as np

def format_step(msg, matrix=None):
    """Format step output with optional matrix"""
    output = [msg]
    if matrix is not None:
        output.append('\n'.join(['  '.join(f'{x:12.6f}' for x in row) for row in matrix]))
    return '\n'.join(output)

def verify_jordan_method(A, x_exact, application_name):
    """Verify Jordan method implementation for given matrix"""
    n = len(A)
    output = []

    # Step 1: Initial setup
    output.append(f"\n{'='*80}")
    output.append(f"Verifying {application_name}")
    output.append(f"{'='*80}\n")

    output.append(format_step("Initial Matrix A:", A))
    output.append(format_step("\nPredetermined solution x:", x_exact.reshape(-1,1)))

    # Step 2: Compute b = Ax
    b = A @ x_exact
    output.append(format_step("\nComputed b = Ax:", b.reshape(-1,1)))

    # Step 3: Create augmented matrix
    augmented = np.concatenate((A.copy(), b.reshape(-1,1)), axis=1)
    output.append(format_step("\nAugmented matrix [A|b]:", augmented))

    # Step 4: Jordan elimination with partial pivoting
    x_computed = np.zeros(n)
    for i in range(n):
        # Partial pivoting
        pivot_row = i + np.argmax(abs(augmented[i:, i]))
        if pivot_row != i:
            output.append(f"\nStep {i+1}.1: Pivot Selection")
            output.append(f"Max element in column {i+1}: |a_{pivot_row+1}{i+1}| = {abs(augmented[pivot_row,i]):.6f}")
            output.append(f"Swapping rows {i+1} and {pivot_row+1}")
            augmented[[i, pivot_row]] = augmented[[pivot_row, i]]
            output.append(format_step("After row swap:", augmented))

        # Normalize pivot row
        pivot = augmented[i,i]
        output.append(f"\nStep {i+1}.2: Normalize row {i+1}")
        output.append(f"Divide row by pivot = {pivot:.6f}")
        augmented[i] = augmented[i] / pivot
        output.append(format_step("After normalization:", augmented))

        # Eliminate column entries
        for j in range(n):
            if i != j:
                multiplier = augmented[j,i]
                if abs(multiplier) > 1e-10:
                    output.append(f"\nStep {i+1}.3: Eliminate entry in row {j+1}")
                    output.append(f"multiplier L_{j+1}{i+1} = {multiplier:.6f}")
                    output.append(f"R_{j+1} = R_{j+1} - ({multiplier:.6f})R_{i+1}")
                    augmented[j] = augmented[j] - multiplier * augmented[i]
                    output.append(format_step("After elimination:", augmented))

    x_computed = augmented[:, -1]

    # Step 5: Compute metrics
    error = np.max(np.abs(x_computed - x_exact)) / np.max(np.abs(x_exact))
    residual = np.max(np.abs(A @ x_computed - b)) / np.max(np.abs(b))
    condition_number = np.linalg.cond(A, np.inf)

    output.append("\nResults:")
    output.append("-" * 40)
    output.append(f"Computed solution x:")
    output.append(format_step("", x_computed.reshape(-1,1)))
    output.append(f"\nError (‖δx‖∞/‖x‖∞): {error:.6e}")
    output.append(f"Residual (‖δr‖∞/‖x‖∞): {residual:.6e}")
    output.append(f"Condition Number κ(A): {condition_number:.6e}")

    return '\n'.join(output), (error, residual, condition_number)

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

    # Process all applications
    results = []
    for A, name in [
        (A1, "Application 1 (Given 8x8 Matrix)"),
        (A2, "Application 2 (Hilbert Matrix)"),
        (A3, "Application 3 (Tridiagonal Symmetric Matrix)")
    ]:
        output, metrics = verify_jordan_method(A, x_exact, name)
        print(output)
        results.append(metrics)

    # Print final results table
    print("\nFinal Results Table")
    print("=" * 80)
    print(f"{'Application':<12} {'Error':<20} {'Residual':<20} {'Condition Number':<20}")
    print("-" * 80)
    for i, (err, res, cond) in enumerate(results, 1):
        print(f"{i:<12} {err:<20.6e} {res:<20.6e} {cond:<20.6e}")
    print("=" * 80)

if __name__ == "__main__":
    main()
