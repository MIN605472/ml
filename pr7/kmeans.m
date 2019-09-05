function [mu, c] = kmeans(D,mu0,K)
% D(m,n), m datapoints, n dimensions
% mu0(K,n) K initial centroids
%
% mu(K,n) final centroids
% c(m) assignment of each datapoint to a class

    function [d] = diff(mu0, mu1)
        d = sum(sqrt(sum((mu0 - mu1) .^ 2, 2)), 1);
    end

iter = 1;
while true
    c = updateClusters(D, mu0);
    mu = updateCentroids(D, c, K);
    fprintf('iter %d\n', iter);
    if diff(mu0, mu) <= 0.0001 || iter >= 100
        fprintf('finished at iter %d\n', iter);
        break;
    end
    mu0 = mu;
    iter = iter + 1;
end
end
