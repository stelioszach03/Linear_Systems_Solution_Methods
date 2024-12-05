import numpy as np
import time
from typing import Tuple, List
import statistics

def create_tridiagonal(n: int) -> np.ndarray:
    """Create the specified tridiagonal matrix."""
    A = np.zeros((n, n))
    # Main diagonal (4)
    np.fill_diagonal(A, 4)
    # Upper diagonal (-3)
    np.fill_diagonal(A[:-1, 1:], -3)
    # Lower diagonal (-1)
    np.fill_diagonal(A[1:, :-1], -1)
    return A

def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm.
    a: lower diagonal (-1)
    b: main diagonal (4)
    c: upper diagonal (-3)
    d: right hand side
    """
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # Forward elimination
    c_prime[0] = c[0]/b[0]
    d_prime[0] = d[0]/b[0]

    for i in range(1, n-1):
        c_prime[i] = c[i]/(b[i] - a[i-1]*c_prime[i-1])
    for i in range(1, n):
        d_prime[i] = (d[i] - a[i-1]*d_prime[i-1])/(b[i] - a[i-1]*c_prime[i-1])

    # Back substitution
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i]*x[i+1]

    return x

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system using Gaussian elimination without pivoting."""
    n = len(b)
    # Create augmented matrix [A|b]
    Ab = np.column_stack([A.copy(), b.copy()])

    # Forward elimination
    for k in range(n-1):
        for i in range(k+1, n):
            factor = Ab[i,k] / Ab[k,k]
            Ab[i,k:n+1] -= factor * Ab[k,k:n+1]

    # Back substitution
    x = np.zeros(n)
    x[n-1] = Ab[n-1,n] / Ab[n-1,n-1]
    for i in range(n-2, -1, -1):
        x[i] = (Ab[i,n] - np.dot(Ab[i,i+1:n], x[i+1:])) / Ab[i,i]

    return x

def run_timing_test(n: int, num_trials: int = 10) -> Tuple[List[float], List[float], float, float]:
    """Run timing comparison for size n."""
    A = create_tridiagonal(n)
    x_exact = np.ones(n)
    b = A @ x_exact

    thomas_times = []
    gauss_times = []

    # Extract diagonals for Thomas method
    main_diag = np.diag(A)
    upper_diag = np.diag(A, 1)
    lower_diag = np.diag(A, -1)

    # Run trials
    for _ in range(num_trials):
        # Time Thomas method
        start = time.perf_counter()
        x_thomas = thomas_algorithm(lower_diag, main_diag, upper_diag, b)
        thomas_times.append(time.perf_counter() - start)

        # Time Gaussian elimination
        start = time.perf_counter()
        x_gauss = gaussian_elimination(A, b)
        gauss_times.append(time.perf_counter() - start)

    # Compute errors for last solutions
    thomas_error = np.linalg.norm(x_thomas - x_exact, np.inf)
    gauss_error = np.linalg.norm(x_gauss - x_exact, np.inf)

    return thomas_times, gauss_times, thomas_error, gauss_error

def main():
    test_sizes = [100, 1000]  # n = 10k, k = 2,3
    num_trials = 10

    print("Execution Time Comparison: Thomas Method vs Gaussian Elimination")
    print("========================================================\n")

    results = []
    for n in test_sizes:
        print(f"\nTesting with matrix size n = {n}")
        print("-" * 40)

        thomas_times, gauss_times, thomas_error, gauss_error = run_timing_test(n, num_trials)

        # Calculate statistics
        thomas_mean = statistics.mean(thomas_times)
        thomas_std = statistics.stdev(thomas_times)
        gauss_mean = statistics.mean(gauss_times)
        gauss_std = statistics.stdev(gauss_times)
        speedup = gauss_mean / thomas_mean

        results.append({
            'n': n,
            'thomas_mean': thomas_mean,
            'thomas_std': thomas_std,
            'gauss_mean': gauss_mean,
            'gauss_std': gauss_std,
            'speedup': speedup,
            'thomas_error': thomas_error,
            'gauss_error': gauss_error
        })

        # Print results for this size
        print(f"\nResults for n = {n}:")
        print(f"Thomas Method:")
        print(f"  Average time: {thomas_mean:.6f} ± {thomas_std:.6f} seconds")
        print(f"  Error: {thomas_error:.2e}")
        print(f"\nGaussian Elimination:")
        print(f"  Average time: {gauss_mean:.6f} ± {gauss_std:.6f} seconds")
        print(f"  Error: {gauss_error:.2e}")
        print(f"\nSpeedup factor: {speedup:.2f}x")

    # Save results to file
    with open('timing_analysis_results.txt', 'w') as f:
        f.write("Execution Time Comparison Analysis\n")
        f.write("================================\n\n")

        f.write("Test Configuration\n")
        f.write("-----------------\n")
        f.write(f"- Matrix sizes tested: {', '.join(map(str, test_sizes))}\n")
        f.write(f"- Number of trials per test: {num_trials}\n")
        f.write("- Matrix properties: Tridiagonal symmetric\n")
        f.write("  * Main diagonal: 4\n")
        f.write("  * Lower subdiagonal: -1\n")
        f.write("  * Upper subdiagonal: -3\n")
        f.write("- Test vector: x = (1,1,...,1)^T\n")
        f.write("- System: Ax = b where b = Ax\n\n")

        f.write("Detailed Results\n")
        f.write("---------------\n\n")

        for result in results:
            n = result['n']
            f.write(f"{n = }\n")
            f.write("-" * (len(str(n)) + 4) + "\n")

            f.write("Thomas Method:\n")
            f.write(f"- Average execution time: {result['thomas_mean']:.6f} seconds\n")
            f.write(f"- Standard deviation: ±{result['thomas_std']:.6f} seconds\n")
            f.write(f"- Solution error: {result['thomas_error']:.2e}\n\n")

            f.write("Gaussian Elimination:\n")
            f.write(f"- Average execution time: {result['gauss_mean']:.6f} seconds\n")
            f.write(f"- Standard deviation: ±{result['gauss_std']:.6f} seconds\n")
            f.write(f"- Solution error: {result['gauss_error']:.2e}\n\n")

            f.write(f"Speedup factor: {result['speedup']:.2f}×\n\n")

        f.write("Summary Table\n")
        f.write("------------\n")
        f.write("Size (n) | Thomas (s) | Gaussian (s) | Speedup\n")
        f.write("---------|-----------:|-------------:|--------\n")
        for result in results:
            f.write(f"{result['n']:8d} | {result['thomas_mean']:9.6f} | {result['gauss_mean']:11.6f} | {result['speedup']:6.2f}×\n")

if __name__ == "__main__":
    main()
