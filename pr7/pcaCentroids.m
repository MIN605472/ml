function [centroids] = pcaCentroids(X, k)
    [Xn, mu, sig] = normalizar(X);
    Sigma = cov(Xn);
    [V, D] = eig(Sigma);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx);
    D = D(idx, idx);
    U = V(:, 1:1);
    Z = Xn * U;
    minRange = min(Z);
    maxRange = max(Z);
    pts = linspace(minRange, maxRange, 2*k)';
    pts = pts(1:2:2*k);
    centroids = pts * U';
    centroids = desnormalizar(centroids, mu, sig);
end
