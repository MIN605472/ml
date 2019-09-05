function [error] = errorReg(hyperParam, w, X, y)
error = RMSE(w, expandir(X, hyperParam(1:1, :)'), y);
end
