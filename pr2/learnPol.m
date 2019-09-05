function [w] = learnPol(hyperParam, X, y)
[Xn, mu, sig] = normalizar(expandir(X, hyperParam'));
w = Xn \ y;
w = desnormalizar(w, mu, sig);
end

