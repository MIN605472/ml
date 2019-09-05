%% Based on exercise 2 of Machine Learning Online Class by Andrew Ng 
clear ; close all;
addpath(genpath(pwd))

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
data = load('exam_data.txt');
y = data(:, 3);
N = length(y);
X = data(:, [1, 2]);
X = [ones(N, 1) X];
[Xcv, ycv, Xtr, ytr] =  particion(1, 5, X, y);

%  The first two columns contains the X values and the third column
%  contains the label (y).
dataMchip = load('mchip_data.txt');
XMchip = dataMchip(:, [1, 2]); 
yMchip = dataMchip(:, 3);
NMchip = length(yMchip);
pMchip = randperm(NMchip); %reordena aleatoriamente los datos
XMchip = XMchip(pMchip, :);
yMchip = yMchip(pMchip);
XMchip = [ones(NMchip, 1) XMchip];
[XcvMchip, ycvMchip, XtrMchip, ytrMchip] = particion(5, 5, XMchip, yMchip);

%% 2. Regresión logística básica
fprintf('2----------\n');
options = [];
options.display = 'none';
options.useMex = 1;
theta = minFunc(@costeLogistico, zeros(size(Xtr, 2), 1), ...
    options, Xtr, ytr);
plotDecisionBoundary(theta, Xtr, ytr);
xlabel('Exam 1 score');
ylabel('Exam 2 score');
errroEntrenamiento = costFun(theta, Xtr, ytr);
errorTest = costFun(theta, Xcv, ycv);
fprintf('Error entrenamiento: %f\nError test: %f\n', errroEntrenamiento, errorTest);
[tp, fp, fn, tn] = calcConfMat(Xtr, ytr, @(x) (1 / (1 + exp(-(theta' * x)))) >= 0.5);
tasaFallosEntrenamiento = (fp + fn) / (tp + fp + fn + tn);
fprintf('Tasa de fallos entrenamiento: %f\n', tasaFallosEntrenamiento);
[tp, fp, fn, tn] = calcConfMat(Xcv, ycv, @(x) (1 / (1 + exp(-(theta' * x)))) >= 0.5);
tasaFallosTest = (fp + fn) / (tp + fp + fn + tn);
fprintf('Tasa de fallos test: %f\n', tasaFallosTest);
x2 = (0:1:100)';
newX = [ones(length(x2), 1) ones(length(x2), 1) * 45 x2];
newY = arrayfun(@(rowNDX) logisticFun(theta, newX(rowNDX, :)'), (1:size(newX, 1))');
figure();
hold on; grid on;
title('P(A|X1=45,X2=x)')
xlabel('Exam 2 score')
ylabel('P(A|X1=45,X2=x)')
plot(x2, newY)

%% 3. Regularizacion
fprintf('3----------\n');
XtrMchip = mapFeature(XtrMchip(:, 2), XtrMchip(:, 3));
XcvMchip = mapFeature(XcvMchip(:, 2), XcvMchip(:, 3));
lambda = 0:1e-4:1e-1;
learnFun = @(lambda, x, y) minFunc(@costeLogisticoReg, ...
    zeros(size(x, 2), 1), options, x, y, lambda);
errorFun = @(lambda, w, x, y) calcMissClass(x, y, @(x) logisticFun(w, x));
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, learnFun, errorFun, lambda, XtrMchip, ytrMchip);
bestLambda = lambda(1, bestHyperParamNDX);
fprintf('Best lambda: %f\n', bestLambda);
figure();
title('Progresion errror')
ylabel('J');
xlabel('lambda');
hold on; grid on;
plot(lambda, trainingErrors, 'r-');
plot(lambda, validationErrors, 'b-');
legend('Error entrenamiento', 'Error validacion');
% Plot best lambda
thetaBestLambda = minFunc(@costeLogisticoReg, zeros(size(XtrMchip, 2), 1), options, XtrMchip, ytrMchip, bestLambda);
% [tp, fp, fn, tn] = calcConfMat(XcvMchip, ycvMchip, @(x) (1 / (1 + exp(-(thetaBestLambda' * x)))) >= 0.5);
% errorTest = (fp + fn) / (tp + fp + fn + tn);
errorTest = costFun(thetaBestLambda, XcvMchip, ycvMchip);
fprintf('Error validacion lambda=%f: %f \n', bestLambda, validationErrors(1, bestHyperParamNDX));
fprintf('Error test lambda=%f: %f \n', bestLambda, errorTest);
[tp, fp, fn, tn] = calcConfMat(XcvMchip, ycvMchip, @(x) (1 / (1 + exp(-(thetaBestLambda' * x)))) >= 0.5);
tasaFallosEntrenamiento = (fp + fn) / (tp + fp + fn + tn);
fprintf('Tasa de fallos test: %f\n', tasaFallosEntrenamiento);
plotDecisionBoundary(thetaBestLambda, XtrMchip, ytrMchip);
title(sprintf('lambda = %f', bestLambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
% Plot lambda = 0, i.e., no regularization or penalty applied
lambdaZero = 0;
theta = minFunc(@costeLogisticoReg, zeros(size(XtrMchip, 2), 1), options, XtrMchip, ytrMchip, lambdaZero);
errorTest = costFun(theta, XcvMchip, ycvMchip);
fprintf('Error validacion lambda=%f: %f \n', lambdaZero, validationErrors(1, 1));
fprintf('Error test lambda=%f: %f \n', lambdaZero, errorTest);
[tp, fp, fn, tn] = calcConfMat(XcvMchip, ycvMchip, @(x) (1 / (1 + exp(-(theta' * x)))) >= 0.5);
tasaFallosTest = (fp + fn) / (tp + fp + fn + tn);
fprintf('Tasa de fallos test: %f\n', tasaFallosTest);
plotDecisionBoundary(theta, XtrMchip, ytrMchip);
title(sprintf('lambda = %f', lambdaZero))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

%% 4. Precision/Recall
fprintf('4----------\n');
XcvMchip =  mapFeature(XcvMchip(:, 2), XcvMchip(:, 3));
[tp, fp, fn, tn] = calcConfMat(XcvMchip, ycvMchip, @(x) (1 / (1 + exp(-(thetaBestLambda' * x)))) >= 0.5);
fprintf('%2d | %2d\n%2d | %2d\n', tp, fp, fn, tn);
fprintf('Precision: %f\n', tp / (tp + fp));
fprintf('Recall: %f\n', tp / (tp + fn));
newTheta = thetaBestLambda + [0; zeros(length(thetaBestLambda) - 1, 1)];
[tp, fp, fn, tn] = calcConfMat(XcvMchip, ycvMchip, @(x) (1 / (1 + exp(-(newTheta' * x)))) >= 0.8);
fprintf('Cambiando theta_0\n');
fprintf('%2d | %2d\n%2d | %2d\n', tp, fp, fn, tn);
fprintf('Precision: %f\n', tp / (tp + fp));
fprintf('Recall: %f\n', tp / (tp + fn));
plotDecisionBoundary(newTheta, XcvMchip, ycvMchip);
title(sprintf('lambda = %f', bestLambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
plotDecisionBoundary(thetaBestLambda, XcvMchip, ycvMchip);
title(sprintf('lambda = %f', bestLambda))
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')

