function [x] = gaussian_elimination_detailed(A, b)
    % Solves the linear system Ax = b using Gaussian elimination without pivoting
    % with detailed logging of intermediate steps
    % Input:
    %   A: n x n coefficient matrix
    %   b: n x 1 right-hand side vector
    % Output:
    %   x: n x 1 solution vector

    % Get system size
    n = size(A, 1);

    % Create augmented matrix [A|b]
    Ab = [A, b];

    % Display initial augmented matrix
    fprintf('\nInitial augmented matrix [A|b]:\n');
    display_matrix(Ab);

    % Forward elimination
    fprintf('\nForward Elimination Phase:\n');
    fprintf('========================\n');

    for k = 1:n-1
        fprintf('\nStep %d:\n', k);
        fprintf('-------\n');

        % For each row below diagonal
        for i = k+1:n
            % Compute and display multiplier
            multiplier = Ab(i,k) / Ab(k,k);
            fprintf('L_%d%d = a_%d%d/a_%d%d = %.6f\n', ...
                    i, k, i, k, k, k, multiplier);

            % Eliminate entry in column k
            Ab(i,:) = Ab(i,:) - multiplier * Ab(k,:);
        end

        % Display intermediate matrix after this step
        fprintf('\nMatrix after step %d:\n', k);
        display_matrix(Ab);
    end

    % Back substitution
    fprintf('\nBack Substitution Phase:\n');
    fprintf('======================\n');

    x = zeros(n, 1);
    x(n) = Ab(n,n+1) / Ab(n,n);
    fprintf('x_%d = %.6f\n', n, x(n));

    for i = n-1:-1:1
        x(i) = (Ab(i,n+1) - Ab(i,i+1:n) * x(i+1:n)) / Ab(i,i);
        fprintf('x_%d = %.6f\n', i, x(i));
    end
end

function display_matrix(A)
    % Helper function to display matrix with aligned columns
    [m, n] = size(A);
    for i = 1:m
        for j = 1:n
            fprintf('%10.4f ', A(i,j));
        end
        fprintf('\n');
    end
    fprintf('\n');
end
