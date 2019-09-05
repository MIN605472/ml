function [error] = errorPol(hyperParam, w, X, y)
error = RMSE(w, expandir(X, hyperParam'), y);
end
