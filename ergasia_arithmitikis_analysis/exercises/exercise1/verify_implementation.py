import numpy as np

def print_matrix(A, label=""):
    """Print matrix with proper formatting"""
    if label:
        print(f"\n{label}:")
    for row in A:
        print(" ".join(f"{x:10.6f}" for x in row))

def verify_application1():
    print("VERIFICATION OF APPLICATION 1 - Given 8x8 Matrix")
    print("=" * 80)

    # Given 8x8 matrix
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

    # Predetermined solution
    x_exact = np.array([-1, 1, -1, 1, -1, 1, -1, 1])

    print("Step 1: Computing b = Ax")
    b = A @ x_exact
    print_matrix(A, "Matrix A")
    print("\nVector x (predetermined):")
    print(x_exact)
    print("\nComputed vector b = Ax:")
    print(b)

    # Create augmented matrix [A|b]
    augmented = np.concatenate((A, b.reshape(-1,1)), axis=1)
    print_matrix(augmented, "\nAugmented matrix [A|b]")

    # Solve using numpy for verification
    x_numpy = np.linalg.solve(A, b)
    print("\nSolution verification using numpy.linalg.solve:")
    print("x_computed =", x_numpy)

    # Calculate metrics
    error = np.max(np.abs(x_numpy - x_exact)) / np.max(np.abs(x_exact))
    residual = np.max(np.abs(A @ x_numpy - b)) / np.max(np.abs(b))
    condition_number = np.linalg.cond(A, np.inf)

    print("\nError Analysis:")
    print(f"Error (‖δx‖∞/‖x‖∞): {error:.6e}")
    print(f"Residual (‖δr‖∞/‖x‖∞): {residual:.6e}")
    print(f"Condition Number κ(A): {condition_number:.6e}")

if __name__ == "__main__":
    verify_application1()
    print("\nVerification complete. Check the results above for accuracy.")
