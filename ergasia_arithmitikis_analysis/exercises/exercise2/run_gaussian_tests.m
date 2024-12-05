% Test script for Gaussian elimination implementation with detailed output
% Tests both basic and detailed versions with tridiagonal symmetric matrix

function run_gaussian_tests()
    % Test for n = 10k, k = 2,3 as specified
    test_sizes = [100, 1000];

    for n = test_sizes
        fprintf('\n==================================================\n');
        fprintf('Testing Gaussian Elimination for n = %d\n', n);
        fprintf('==================================================\n');

        % Create tridiagonal symmetric matrix
        A = create_tridiagonal_matrix(n);

        % Create exact solution vector x = (1,1,...,1)^T
        x_exact = ones(n, 1);

        % Compute right-hand side b = Ax
        b = A * x_exact;

        % For smaller size, show detailed elimination steps
        if n == 100
            fprintf('\nRunning detailed elimination process...\n');
            x_detailed = gaussian_elimination_detailed(A, b);

            % Verify detailed solution
            error_detailed = norm(x_detailed - x_exact, inf);
            residual_detailed = norm(b - A*x_detailed, inf);

            fprintf('\nDetailed Solution Verification:\n');
            fprintf('Error (infinity norm): %.2e\n', error_detailed);
            fprintf('Residual (infinity norm): %.2e\n', residual_detailed);
        end

        % Run basic version for all sizes
        fprintf('\nRunning basic elimination...\n');
        x_basic = gaussian_elimination(A, b);

        % Verify basic solution
        error_basic = norm(x_basic - x_exact, inf);
        residual_basic = norm(b - A*x_basic, inf);

        fprintf('\nBasic Solution Verification:\n');
        fprintf('Error (infinity norm): %.2e\n', error_basic);
        fprintf('Residual (infinity norm): %.2e\n', residual_basic);

        % Display first and last few components for verification
        fprintf('\nSolution components (showing first and last 3):\n');
        fprintf('First three components:\n');
        for i = 1:3
            fprintf('x(%d) = %.15f\n', i, x_basic(i));
        end
        fprintf('...\n');
        fprintf('Last three components:\n');
        for i = n-2:n
            fprintf('x(%d) = %.15f\n', i, x_basic(i));
        end
    end
end

function A = create_tridiagonal_matrix(n)
    % Creates n×n tridiagonal symmetric matrix with specified structure:
    % - Main diagonal: 4
    % - Lower subdiagonal: -1
    % - Upper subdiagonal: -3

    % Initialize matrix
    A = zeros(n,n);

    % Set main diagonal (4)
    for i = 1:n
        A(i,i) = 4;
    end

    % Set upper subdiagonal (-3)
    for i = 1:n-1
        A(i,i+1) = -3;
    end

    % Set lower subdiagonal (-1)
    for i = 2:n
        A(i,i-1) = -1;
    end
end

% Run the tests
run_gaussian_tests();
