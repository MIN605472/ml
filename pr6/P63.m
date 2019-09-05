% load images 
% images size is 20x20. 
clear;
close all;

load('MNISTdata2.mat');
classes = unique(y);
nclasses = length(classes);

nrows=20;
ncols=20;

nimages = size(X,1);

%% Perform PCA over all numbers
[X, mu, sig] = normalizar(X);
[Xtest, mu, sig] = normalizar(Xtest);
Sigma = cov(X);
[V, D] = eig(Sigma);
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
D = D(idx, idx);
U = V(:, 1:2);
Z = X * U;
Ztest = Xtest * U;

% z should contain the projections over the first two PC
% now is just a random matrix
% z=rand(size(X,1),2);

% Muestra las dos componentes principales
figure(100)
clf, hold on
plotwithcolor(Z(:,1:2), y);

%% Use classifier from previous labs on the projected space
bestLambda = 0;
naive = 0;
modelo = entrenarGaussianas(Z, y, nclasses, naive, bestLambda);
yHat = clasificacionBayesiana(modelo, Ztest);
C = confusionMat(ytest, yHat);
plotConfMat(C, [1:9 0]);

%% Find out what the best k is
% bestLambda = 0;
% naive = 0;
% ks = 1:200;
% accs = zeros(size(ks, 1), size(ks, 2)); 
% for k=ks
%     U = V(:, 1:k);
%     Z = X * U;
%     Ztest = Xtest * U;
%     modelo = entrenarGaussianas(Z, y, nclasses, naive, bestLambda);
%     yHat = clasificacionBayesiana(modelo, Ztest);
%     C = confusionMat(ytest, yHat);
%     accs(k) = 100*trace(C)/sum(C(:));
% end
% figure();
% plot(ks, accs);
% [accsS, idx] = sort(accs, 'descend');

%% Apply k-fold cross validation to find the best k
% ks = 1:200;
% [bestHyperParamNDX, trainingErrors, validationErrors] = ...
%     kfoldCrossValidation(10, @learnFun, @errorFun, ks, X, y);
% bestK = k(bestHyperParamNDX);
% fprintf('Best lambda: %f\n', bestK);
% semilogx(lambda, trainingErrors, 'r-');
% hold on;
% semilogx(lambda, validationErrors, 'b-');
% title('Progresion errror')
% ylabel('J');
% xlabel('lambda');
% legend('Error entrenamiento', 'Error validacion');