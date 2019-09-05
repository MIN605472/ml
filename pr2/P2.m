close all;
%% Cargar los datos
datos = load('CochesTrain.txt');
ydatos = datos(:, 1);   % Precio en Euros
Xdatos = datos(:, 2:4); % Años, Km, CV
x1dibu = linspace(min(Xdatos(:,1)), max(Xdatos(:,1)), 100)'; %para dibujar

datos2 = load('CochesTest.txt');
ytest = datos2(:,1);  % Precio en Euros
Xtest = datos2(:,2:4); % Años, Km, CV
Ntest = length(ytest);

%% 2. Selección del grado del polinomio para la antigüedad del coche
hyperParam = [1:10; ones(1, 10); ones(1, 10)];
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, @learnPol, @errorPol, hyperParam, Xdatos, ydatos);
figure();
title('Progresion errror')
ylabel('RMSE');
xlabel('Grado del polinomio para la antigüedad');
hold on; grid on;
plot(1:length(trainingErrors), trainingErrors, 'r-');
plot(1:length(validationErrors), validationErrors, 'b-');
legend('Error entrenamiento', 'Error validacion');
bestDegreeAge = hyperParam(1, bestHyperParamNDX);

%% 3. Selección del grado del polinomio para los kilómetros
hyperParam = [ones(1, 10) * bestDegreeAge; 1:10; ones(1, 10)];
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, @learnPol, @errorPol, hyperParam, Xdatos, ydatos);
figure();
title('Progresion errror')
ylabel('RMSE');
xlabel('Grado del polinomio para los kilómetros');
hold on; grid on;
plot(1:length(trainingErrors), trainingErrors, 'r-');
plot(1:length(validationErrors), validationErrors, 'b-');
legend('Error entrenamiento', 'Error validacion');
bestDegreeKm = hyperParam(2, bestHyperParamNDX);
w = learnPol([bestDegreeAge, bestDegreeKm, 1], Xdatos, ydatos);
errorTest3 = RMSE(w, expandir(Xtest, [bestDegreeAge, bestDegreeKm, 1]), ...
    ytest);
disp(['Error test: ', num2str(errorTest3)]);
bestErrorValidation3 = validationErrors(1, bestHyperParamNDX);

%% 4. Regularización
delta = -1E-7:1e-7:1e-6;
errorFun = @(hyperParam, w, X, y) RMSE(w, X,y);
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, @learnReg, errorFun, delta, ...
    expandir(Xdatos, [10 5 5]), ydatos);
figure();
title('Progresion errror')
ylabel('RMSE');
xlabel('Lambda');
hold on; grid on;
plot(delta, trainingErrors, 'r-');
plot(delta, validationErrors, 'b-');
legend('Error entrenamiento', 'Error validacion');
w = learnReg(delta(1, bestHyperParamNDX), expandir(Xdatos, [10 5 5]), ydatos);
errorTest4 = RMSE(w, expandir(Xtest, [10 5 5]), ytest);
disp(['Error test: ', num2str(errorTest4)]);
bestErrorValidation4 = validationErrors(1, bestHyperParamNDX);
fprintf('**\nTest error 3: %f Test error 4: %f\n', errorTest3, errorTest4);
fprintf('Validation error 3: %f Validation error 4: %f\n', ...
    bestErrorValidation3, bestErrorValidation4);

%% 42. Algunas probatinas extras
P = npermutek(1:10, 3);
B = P(:, 3) <= 10;
P = P(B, :)';
hyperParam = [P];
[bestHyperParamNDX, trainingErrors, validationErrors] = ...
    kfoldCrossValidation(5, @learnPol, @errorPol, hyperParam, Xdatos, ydatos);
figure();
title('Progresion errror')
ylabel('RMSE');
xlabel('Grado del polinomio para la antigüedad');
hold on; grid on;
plot(1:length(trainingErrors), trainingErrors, 'r-');
plot(1:length(validationErrors), validationErrors, 'b-');
legend('Error entrenamiento', 'Error validacion');

w = learnPol(hyperParam(:, bestHyperParamNDX), Xdatos, ydatos);
errorTest = RMSE(w, expandir(Xtest, hyperParam(:, bestHyperParamNDX)), ...
    ytest);
disp(['Error test: ', num2str(errorTest)]);