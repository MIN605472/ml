function [ error ] = errorFun(k, model, X, y)
Z = X * model.u;
yHat = clasificacionBayesiana(model.m, Z);
error = sum(y ~= yHat) / length(yHat);
end

