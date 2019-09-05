function [bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(k, learnFun, errorFun, hyperParam, X, y)
%kFoldCrossValidation Apply k-fold cross validation.
%INPUT:
%   k: the number of folds
%   learnFun: the learning function
%   errorFun: the errro function
%   hyperParam: the hyper parameters values. Should be a matrix where each
%   column represents the values
%   X: the input training data
%   y: the outut training data
% OUTPUT:
%   bestHyperParamNDX: the index pointing to a column in hyperParam whose
%   values give the lowest validation error
nhyperparams = size(hyperParam, 2);
trainingErrors = zeros(1, nhyperparams);
validationErrors = zeros(1, nhyperparams);
bestErrorV = Inf;
for hyperParamNDX = 1:nhyperparams
    errorT = 0;
    errorV = 0;
    for fold = 1:k
        [Xcv, ycv, Xtr, ytr] = particion(fold, k, X, y);
        hyperParams = hyperParam(:, hyperParamNDX);
        w = learnFun(hyperParams, Xtr, ytr);
        errorT = errorT + errorFun(hyperParams, w, Xtr, ytr);
        errorV = errorV + errorFun(hyperParams, w, Xcv, ycv);
    end
    errorT = errorT / k;
    errorV = errorV / k;
    trainingErrors(1, hyperParamNDX) = errorT;
    validationErrors(1, hyperParamNDX) = errorV;
    if errorV < bestErrorV
        bestErrorV = errorV;
        bestHyperParamNDX = hyperParamNDX;
    end
end
end

