function [refVectors] = randomCentroids(X, k)
[m, n] = size(X);
refVectors = X(randi(m, k, 1), :);
end

