function [weights] = normalEquation(trX, trR)
N = size(trX, 1); % num. of instances
X = [ones(N,1) trX];
weights = X \ trR;
end

