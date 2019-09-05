function [yHat] = predictOneVsAll(weights, X)
yHats = 1 ./ (1 + exp(-(X * weights)));
[~, yHat] = max(yHats, [], 2);
end

