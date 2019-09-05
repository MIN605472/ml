function [w] = learnReg(hyperParam, X, y)
lambda = hyperParam(1, 1);
[Xn, mu, sig] = normalizar(X);
H = Xn' * Xn + lambda * diag([0 ones(1, size(Xn, 2) - 1)]);
w = H \ (Xn' * y);
w = desnormalizar(w, mu, sig);
end

