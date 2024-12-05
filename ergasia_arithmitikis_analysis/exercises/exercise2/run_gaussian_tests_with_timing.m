% Test script for Gaussian elimination implementation with detailed output and timing
% Tests both basic and detailed versions with tridiagonal symmetric matrix

function run_gaussian_tests_with_timing()
    % Test for n = 10k, k = 2,3 as specified
    test_sizes = [100, 1000];

    % Store timing results
    times = zeros(length(test_sizes), 1);

    for idx = 1:length(test_sizes)
        n = test_sizes(idx);
        fprintf('\n==================================================\n');
        fprintf('Testing Gaussian Elimination for n = %d\n', n);
        fprintf('==================================================\n');

        % Create tridiagonal symmetric matrix
        A = create_tridiagonal_matrix(n);
        x_exact = ones(n, 1);
        b = A * x_exact;

        % For n=100, show detailed elimination steps
        if n == 100
            fprintf('\nDetailed elimination process for n = 100:\n');
            x_detailed = gaussian_elimination_detailed(A, b);
            verify_solution(A, b, x_detailed, x_exact, 'Detailed');
        end

        % Time the basic version
        fprintf('\nTiming basic Gaussian elimination...\n');
        tic;
        x_basic = gaussian_elimination(A, b);
        times(idx) = toc;

        % Verify solution
        verify_solution(A, b, x_basic, x_exact, 'Basic');

        fprintf('\nExecution time: %.6f seconds\n', times(idx));

        % Display solution components
        display_solution_components(x_basic, n);
    end

    % Display timing summary
    fprintf('\nTiming Summary:\n');
    fprintf('==============\n');
    for idx = 1:length(test_sizes)
        fprintf('n = %d: %.6f seconds\n', test_sizes(idx), times(idx));
    end
end

function verify_solution(A, b, x_computed, x_exact, method_name)
    error = norm(x_computed - x_exact, inf);
    residual = norm(b - A*x_computed, inf);

    fprintf('\n%s Solution Verification:\n', method_name);
    fprintf('Error (infinity norm): %.2e\n', error);
    fprintf('Residual (infinity norm): %.2e\n', residual);
end

function display_solution_components(x, n)
    fprintf('\nSolution components (first and last 3):\n');
    fprintf('First three:\n');
    for i = 1:3
        fprintf('x(%d) = %.15f\n', i, x(i));
    end
    fprintf('...\n');
    fprintf('Last three:\n');
    for i = n-2:n
        fprintf('x(%d) = %.15f\n', i, x(i));
    end
end

function A = create_tridiagonal_matrix(n)
    % Creates n×n tridiagonal symmetric matrix
    A = diag(4*ones(n,1)) + ...
        diag(-3*ones(n-1,1), 1) + ...
        diag(-1*ones(n-1,1), -1);
end

% Run the tests
run_gaussian_tests_with_timing();
