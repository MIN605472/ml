function [ error ] = reconstructionError(D, mu, c)
    [m, n] = size(D);
    error = 0;
    for i=1:m
        mu_i = mu(c(i), :);
        error = error + sum((D(i, :) - mu_i) .^ 2, 2);
    end
end

