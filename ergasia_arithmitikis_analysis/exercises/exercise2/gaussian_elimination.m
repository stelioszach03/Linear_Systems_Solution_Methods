function [x] = gaussian_elimination(A, b)
    % Solves the linear system Ax = b using Gaussian elimination without pivoting
    % Input:
    %   A: n x n coefficient matrix
    %   b: n x 1 right-hand side vector
    % Output:
    %   x: n x 1 solution vector

    % Get system size
    n = size(A, 1);

    % Create augmented matrix [A|b]
    Ab = [A, b];

    % Forward elimination
    for k = 1:n-1
        % For each row below diagonal
        for i = k+1:n
            % Compute multiplier
            multiplier = Ab(i,k) / Ab(k,k);

            % Eliminate entry in column k
            Ab(i,:) = Ab(i,:) - multiplier * Ab(k,:);
        end
    end

    % Back substitution
    x = zeros(n, 1);
    x(n) = Ab(n,n+1) / Ab(n,n);
    for i = n-1:-1:1
        x(i) = (Ab(i,n+1) - Ab(i,i+1:n) * x(i+1:n)) / Ab(i,i);
    end
end
