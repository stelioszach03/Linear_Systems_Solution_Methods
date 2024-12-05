# Exercise 2: Tridiagonal System Solver Implementation

## Overview
This implementation solves a tridiagonal symmetric linear system using two methods:
1. Thomas Algorithm (specialized for tridiagonal systems)
2. Gaussian Elimination without pivoting (general method)

## Matrix Specifications
- Size: n × n (tested with n = 100, 1000)
- Main diagonal: 4
- Lower subdiagonal: -1
- Upper subdiagonal: -3
- Structure: Symmetric tridiagonal

## Files Description

### Core Implementation
- `run_timing_comparison.py`: Main script for timing comparison
  - Creates test matrices
  - Implements both solution methods
  - Runs timing experiments
  - Generates detailed results

### Results and Analysis
- `timing_analysis_results.txt`: Detailed timing results
- `theoretical_analysis.txt`: Mathematical justification of performance differences

## Implementation Details

### Thomas Algorithm
```python
def thomas_algorithm(a, b, c, d):
    """
    Solves Ax = d where A is tridiagonal with diagonals a, b, c
    a: lower diagonal (-1)
    b: main diagonal (4)
    c: upper diagonal (-3)
    d: right-hand side vector
    Returns: solution vector x
    """
```

### Gaussian Elimination
```python
def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian elimination without pivoting
    A: full matrix
    b: right-hand side vector
    Returns: solution vector x
    """
```

## Verification Process
1. Create test vector x = (1, 1, ..., 1)ᵀ
2. Compute b = Ax
3. Solve Ax = b using both methods
4. Verify solution accuracy
5. Measure execution times

## Key Results

### Performance Comparison
Matrix Size | Thomas (s) | Gaussian (s) | Speedup
-----------|------------|--------------|--------
100        | 0.000168   | 0.012829     | 76.53×
1000       | 0.001756   | 1.539895     | 877.11×

### Error Analysis
Both methods maintain high accuracy:
- Thomas Method: Error ≈ 5.55e-16
- Gaussian Elimination: Error ≈ 6.66e-16

## Usage
```bash
python3 run_timing_comparison.py
```

## Implementation Notes
1. All matrices are created using NumPy for efficient operations
2. Timing uses high-precision performance counter
3. Multiple trials ensure reliable timing measurements
4. Error calculations use infinity norm
5. Results include standard deviation for timing reliability

## Theoretical Background
The Thomas method achieves O(n) complexity by:
1. Exploiting tridiagonal structure
2. Eliminating need for full matrix storage
3. Optimizing memory access patterns
4. Reducing operation count to 5n

See `theoretical_analysis.txt` for detailed mathematical justification.
