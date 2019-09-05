%% Practica 5
% clear ; close all;
% Carga los datos y los permuta aleatoriamente
load('MNISTdata2.mat'); % Lee los datos: X, y, Xtest, ytest
rand('state',0);
p = randperm(length(y));
X = X(p,:);
y = y(p);
nc = length(unique(y));

%% 3. Bayes ingenuo.
lambda = logspace(-4, 0, 10);
naive = 1;
learnFun = @(lambda, Xtr, ytr) entrenarGaussianas(Xtr, ytr, nc, ...
    naive, lambda);
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, learnFun, @errorFun, lambda, X, y);
% Mostrar evolucion error entrenamiento y validacion segun lambda
bestLambda = lambda(1, bestHyperParamNDX);
fprintf('-----------------Bayes ingenuo\n');
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
modelo = entrenarGaussianas(X, y, nc, naive, bestLambda);
yHat = clasificacionBayesiana(modelo, Xtest);
verConfusiones(Xtest, ytest, yHat);
C = confusionMat(ytest, yHat);
plotConfMat(C, [1:9 0]);

%% 4. Covarianzas completas.
lambda = logspace(-4, 0, 10);
naive = 0;
learnFun = @(lambda, Xtr, ytr) entrenarGaussianas(Xtr, ytr, nc, ...
    naive, lambda);
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, learnFun, @errorFun, lambda, X, y);
% Mostrar evolucion error entrenamiento y validacion segun lambda
bestLambda = lambda(1, bestHyperParamNDX);
fprintf('-----------------Covarianzas completas\n');
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
modelo = entrenarGaussianas(X, y, nc, naive, bestLambda);
yHat = clasificacionBayesiana(modelo, Xtest);
verConfusiones(Xtest, ytest, yHat);
C = confusionMat(ytest, yHat);
plotConfMat(C, [1:9 0]);
fprintf('Test error: %f\n', errorFun(bestLambda, modelo, Xtest, ytest));
