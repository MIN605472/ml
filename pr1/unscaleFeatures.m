function [unscaledFeatures] = unscaleFeatures(scaledFeatures, mu, sigma)
%UNSCALEFEATURES Summary of this function goes here
%   Detailed explanation goes here
N = size(scaledFeatures, 1);
unscaledFeatures = scaledFeatures .* repMat(sigma, N, 1) + repMat(mu, N, 1);
end

