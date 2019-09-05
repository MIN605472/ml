function munew = updateCentroids(D,c,K)
% D((m,n), m datapoints, n dimensions
% c(m) assignment of each datapoint to a class
%
% munew(K,n) new centroids

[m, n] = size(D);
munew = zeros(K, n);
for i=1:K
    % sum_t bit * xt/ sum_t bit
    munew(i, :) = mean(D(c==i, :));
end