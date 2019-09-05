function [weights] = learnOneVsAll(lambda, X, y)
classes = unique(y);
K = length(classes);
D = size(X, 2);
weights = zeros(D, K);
options = [];
options.display = 'none';
options.Method = 'lbfgs';
% options.MaxFunEvals = 100;
options.MaxIter = 100;
% options.optTol = 1e-4;
% options.progTol = 1e-9;
options.Corr = 10000;
options.useMex = 1;
for i = classes'
    y_i = (y == i);
    w_i = minFunc(@costeLogisticoReg, zeros(size(X, 2), 1), options, X, ...
        y_i, lambda);
    weights(:, i) = w_i;
end
end

