% Script to compare execution times between Thomas method and Gaussian elimination
% for solving tridiagonal symmetric linear systems

function compare_methods()
    % Test sizes as specified: n = 10k, k = 2,3
    test_sizes = [100, 1000];
    num_trials = 5;  % Run multiple trials for more reliable timing

    % Initialize results storage
    results = struct();
    results.sizes = test_sizes;
    results.thomas_times = zeros(length(test_sizes), num_trials);
    results.gaussian_times = zeros(length(test_sizes), num_trials);
    results.thomas_errors = zeros(length(test_sizes), 1);
    results.gaussian_errors = zeros(length(test_sizes), 1);

    fprintf('Comparing Thomas Method vs Gaussian Elimination\n');
    fprintf('=============================================\n\n');

    for i = 1:length(test_sizes)
        n = test_sizes(i);
        fprintf('Testing with matrix size n = %d\n', n);
        fprintf('--------------------------------\n');

        % Create test problem
        A = create_tridiagonal_matrix(n);
        x_exact = ones(n, 1);
        b = A * x_exact;

        % Run multiple trials
        for trial = 1:num_trials
            % Time Thomas method
            tic;
            x_thomas = thomas_solver(diag(A,-1), diag(A), diag(A,1), b);
            results.thomas_times(i, trial) = toc;

            % Time Gaussian elimination
            tic;
            x_gaussian = gaussian_elimination(A, b);
            results.gaussian_times(i, trial) = toc;

            % Store errors (only from first trial)
            if trial == 1
                results.thomas_errors(i) = norm(x_thomas - x_exact, inf);
                results.gaussian_errors(i) = norm(x_gaussian - x_exact, inf);
            end
        end

        % Display results for this size
        fprintf('\nResults for n = %d:\n', n);
        fprintf('Thomas Method:\n');
        fprintf('  Average time: %.6f seconds\n', mean(results.thomas_times(i,:)));
        fprintf('  Error: %.2e\n', results.thomas_errors(i));

        fprintf('\nGaussian Elimination:\n');
        fprintf('  Average time: %.6f seconds\n', mean(results.gaussian_times(i,:)));
        fprintf('  Error: %.2e\n', results.gaussian_errors(i));

        fprintf('\nSpeedup ratio: %.2fx\n', ...
            mean(results.gaussian_times(i,:))/mean(results.thomas_times(i,:)));
    end

    % Display final summary table
    fprintf('\nExecution Time Summary\n');
    fprintf('=====================\n');
    fprintf('Matrix Size | Thomas Method | Gaussian Elim. | Speedup\n');
    fprintf('-----------+---------------+---------------+--------\n');
    for i = 1:length(test_sizes)
        fprintf('%10d | %11.6fs | %11.6fs | %7.2fx\n', ...
            test_sizes(i), ...
            mean(results.thomas_times(i,:)), ...
            mean(results.gaussian_times(i,:)), ...
            mean(results.gaussian_times(i,:))/mean(results.thomas_times(i,:)));
    end

    % Display error summary
    fprintf('\nError Summary\n');
    fprintf('============\n');
    fprintf('Matrix Size | Thomas Method | Gaussian Elim.\n');
    fprintf('-----------+---------------+--------------\n');
    for i = 1:length(test_sizes)
        fprintf('%10d | %11.2e | %11.2e\n', ...
            test_sizes(i), ...
            results.thomas_errors(i), ...
            results.gaussian_errors(i));
    end
end

function A = create_tridiagonal_matrix(n)
    % Creates n×n tridiagonal symmetric matrix with:
    % - Main diagonal: 4
    % - Lower subdiagonal: -1
    % - Upper subdiagonal: -3
    A = diag(4*ones(n,1)) + ...
        diag(-3*ones(n-1,1), 1) + ...
        diag(-1*ones(n-1,1), -1);
end

% Run the comparison
compare_methods();
