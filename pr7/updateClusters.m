function Z = updateClusters(D,mu)
% D(m,n), m datapoints, n dimensions
% mu(K,n) final centroids
%
% c(m) assignment of each datapoint to a class
[m, n] = size(D);
Z = zeros(m, 1);
for i=1:m
    mu_i = sum((mu - repmat(D(i, :), size(mu, 1), 1)) .^ 2, 2);
    [~, min_idx] = min(mu_i);
    Z(i) = min_idx;
end
end