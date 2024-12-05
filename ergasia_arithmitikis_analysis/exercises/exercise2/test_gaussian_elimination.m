% Test script for Gaussian elimination implementation
% Tests the solver with tridiagonal symmetric matrix for different sizes

function test_gaussian_elimination()
    % Test sizes: n = 10k, k = 2,3
    sizes = [100, 1000];

    for n = sizes
        fprintf('\nTesting with matrix size n = %d\n', n);
        fprintf('=====================================\n');

        % Create tridiagonal symmetric matrix
        % Main diagonal: 4
        % Lower subdiagonal: -1
        % Upper subdiagonal: -3
        A = create_tridiagonal(n);

        % Create exact solution vector x = (1,1,...,1)^T
        x_exact = ones(n, 1);

        % Compute right-hand side b = Ax
        b = A * x_exact;

        % Solve using Gaussian elimination
        x_computed = gaussian_elimination(A, b);

        % Compute error
        error = norm(x_computed - x_exact, inf);

        % Compute residual
        residual = norm(b - A*x_computed, inf);

        % Display results
        fprintf('Error (infinity norm): %.2e\n', error);
        fprintf('Residual (infinity norm): %.2e\n', residual);

        % Verify solution components
        if n <= 100  % Only show components for smaller size
            fprintf('\nFirst few solution components:\n');
            for i = 1:min(5,n)
                fprintf('x(%d) = %.15f\n', i, x_computed(i));
            end
            fprintf('...\n');
            for i = max(n-4,6):n
                fprintf('x(%d) = %.15f\n', i, x_computed(i));
            end
        end
    end
end

function A = create_tridiagonal(n)
    % Creates n×n tridiagonal symmetric matrix with
    % main diagonal = 4, lower = -1, upper = -3

    % Initialize sparse matrix for efficiency
    A = sparse(n,n);

    % Set main diagonal
    A = A + 4*speye(n);

    % Set sub-diagonals
    for i = 1:n-1
        A(i,i+1) = -3;  % upper subdiagonal
        A(i+1,i) = -1;  % lower subdiagonal
    end

    % Convert to full matrix as Gaussian elimination expects dense matrix
    A = full(A);
end

% Run tests
test_gaussian_elimination();
