%% Practica 4 
% Based on exercise 3 of Machine Learning Online Class by Andrew Ng

clear ; close all;
addpath(genpath('./minfunc'));

% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
Xnon = X;
X = expandir(X, ones(size(X(1, :))) * 2);
Xtestnon = Xtest;
Xtest = expandir(Xtest, ones(size(Xtest(1, :))) * 2);
y = y(p);

%% 2. Regresión logística regularizada.
lambda = logspace(-5, -1, 10);
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, @learnOneVsAll, @errorOneVsAll, lambda, X, y);
% Mostrar evolucion error entrenamiento y validacion segun lambda
bestLambda = lambda(1, bestHyperParamNDX);
fprintf('-----------------\n');
fprintf('Best lambda: %f\n', bestLambda);
fprintf('Training error: %f\n', trainingErrors(1, bestHyperParamNDX));
fprintf('Validation error: %f\n', validationErrors(1, bestHyperParamNDX));
figure();
% hold on; grid on;
semilogx(lambda, trainingErrors, 'r-');
hold on;
semilogx(lambda, validationErrors, 'b-');
title('Progresion errror')
ylabel('J');
xlabel('lambda');
legend('Error entrenamiento', 'Error validacion');


%% 3. Matriz de confusión y Precisión/Recall.
% bestLambda = 0;
weights = learnOneVsAll(bestLambda, X, y);
yHat = predictOneVsAll(weights, Xtest);
verConfusiones(Xtestnon, ytest, yHat);
C = confusionMat(ytest,yHat);
plotConfMat(C, [1:9 0]);
fprintf('-----------------\n');
testError = errorOneVsAll(bestLambda, weights, Xtest, ytest);
fprintf('Test error: %f\n', testError);
