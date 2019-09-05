function [ model ] = learnFun(k, Xtr, ytr)
Sigma = cov(Xtr);
[V, D] = eig(Sigma);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
D = D(idx, idx);
U = V(:, 1:k);
Z = Xtr * U;
naive = 0;
bestLambda = 0;
nclasses = length(unique(ytr));
model = struct;
model.m = entrenarGaussianas(Z, ytr, nclasses, naive, bestLambda);
model.u = U;
end
