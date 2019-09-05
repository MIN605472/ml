function [scaledFeatures, mu, sigma] = scaleFeatures(trX)
%FEATURESCALING Summary of this function goes here
%   Detailed explanation goes here
N = size(trX, 1);
mu = sum(trX) ./ N;
sigma = (sum((trX - repmat(mu, N, 1)) .^ 2) ./ (N - 1)) .^ (1/2);
% maxs = max(trX);
% mins = min(trX);
% sigma = maxs - mins;
scaledFeatures = (trX - repmat(mu, N, 1)) ./ repmat(sigma, N, 1);
mu = mu';
sigma = sigma';
end
