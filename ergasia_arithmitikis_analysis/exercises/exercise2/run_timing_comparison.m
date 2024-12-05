% Complete timing comparison suite for Thomas method vs Gaussian elimination
% Includes all necessary functions and proper timing infrastructure

function run_timing_comparison()
    % Configuration
    test_sizes = [100, 1000];  % n = 10k, k = 2,3
    num_trials = 10;  % Multiple trials for statistical significance

    % Results table initialization
    fprintf('Execution Time Comparison: Thomas Method vs Gaussian Elimination\n');
    fprintf('========================================================\n\n');

    % Header for detailed results
    fprintf('Detailed Results:\n');
    fprintf('---------------\n');

    for n = test_sizes
        fprintf('\nMatrix size n = %d:\n', n);

        % Create test problem
        A = create_tridiagonal(n);
        x_exact = ones(n, 1);
        b = A * x_exact;

        % Extract diagonals for Thomas method
        main_diag = diag(A);
        upper_diag = diag(A, 1);
        lower_diag = diag(A, -1);

        % Initialize timing arrays
        thomas_times = zeros(num_trials, 1);
        gauss_times = zeros(num_trials, 1);

        % Run trials
        for trial = 1:num_trials
            % Time Thomas method
            tic;
            x_thomas = thomas_method(lower_diag, main_diag, upper_diag, b);
            thomas_times(trial) = toc;

            % Time Gaussian elimination
            tic;
            x_gauss = gaussian_elimination(A, b);
            gauss_times(trial) = toc;

            % Verify solutions on first trial
            if trial == 1
                thomas_error = norm(x_thomas - x_exact, inf);
                gauss_error = norm(x_gauss - x_exact, inf);
                fprintf('\nSolution verification:\n');
                fprintf('Thomas method error: %.2e\n', thomas_error);
                fprintf('Gaussian elimination error: %.2e\n', gauss_error);
            end
        end

        % Calculate statistics
        thomas_mean = mean(thomas_times);
        thomas_std = std(thomas_times);
        gauss_mean = mean(gauss_times);
        gauss_std = std(gauss_times);
        speedup = gauss_mean / thomas_mean;

        % Display results
        fprintf('\nTiming Results (averaged over %d trials):\n', num_trials);
        fprintf('Thomas Method:         %.6f ± %.6f seconds\n', thomas_mean, thomas_std);
        fprintf('Gaussian Elimination:  %.6f ± %.6f seconds\n', gauss_mean, gauss_std);
        fprintf('Speedup Factor:        %.2fx\n', speedup);
    end

    % Final summary table
    fprintf('\nSummary Table\n');
    fprintf('============\n');
    fprintf('Size (n) | Thomas (s) | Gaussian (s) | Speedup\n');
    fprintf('---------+-----------+-------------+--------\n');
    for n = test_sizes
        A = create_tridiagonal(n);
        b = A * ones(n, 1);

        % Single timing run for summary
        tic; thomas_method(diag(A,-1), diag(A), diag(A,1), b); t1 = toc;
        tic; gaussian_elimination(A, b); t2 = toc;

        fprintf('%8d | %9.6f | %11.6f | %7.2fx\n', n, t1, t2, t2/t1);
    end
end

function A = create_tridiagonal(n)
    % Creates tridiagonal symmetric matrix with specified structure
    A = diag(4*ones(n,1)) + ...
        diag(-3*ones(n-1,1), 1) + ...
        diag(-1*ones(n-1,1), -1);
end

function x = thomas_method(a, b, c, d)
    % Thomas algorithm for tridiagonal system
    % a: lower diagonal (-1)
    % b: main diagonal (4)
    % c: upper diagonal (-3)
    % d: right-hand side vector

    n = length(d);
    c_prime = zeros(n-1, 1);
    d_prime = zeros(n, 1);
    x = zeros(n, 1);

    % Forward elimination
    c_prime(1) = c(1)/b(1);
    d_prime(1) = d(1)/b(1);

    for i = 2:n-1
        c_prime(i) = c(i)/(b(i) - a(i-1)*c_prime(i-1));
    end

    for i = 2:n
        d_prime(i) = (d(i) - a(i-1)*d_prime(i-1))/(b(i) - a(i-1)*c_prime(i-1));
    end

    % Back substitution
    x(n) = d_prime(n);
    for i = n-1:-1:1
        x(i) = d_prime(i) - c_prime(i)*x(i+1);
    end
end

% Run the comparison
run_timing_comparison();
