function [cost] = costFun(theta, X, y)
N = size(X,1);
h = 1./(1+exp(-(X*theta)));
cost = (-y'*log(h) - (1-y')*log(1-h))/N;
end

