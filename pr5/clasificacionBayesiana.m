function yhat = clasificacionBayesiana( modelo, X)
% Con los modelos entrenados, predice la clase para cada muestra X
K = length(modelo);
N = size(X, 1);
gis = zeros(N, K);
for i = 1:K
    gis(:, i) = gaussLog(modelo{i}.mu, modelo{i}.Sigma, X);
end
[~, yhat] = max(gis, [], 2);
end