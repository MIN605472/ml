function [prob] = logisticFun(theta, x)
prob = 1 / (1 + exp(-(theta' * x)));
end