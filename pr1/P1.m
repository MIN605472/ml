close all;
%% Cargar los datos
% Cargar los datos de entrenamiento
datos = load('PisosTrain.txt');
y = datos(:,3);  % Precio en Euros
x1 = datos(:,1); % m^2
x2 = datos(:,2); % Habitaciones
N = length(y);
% Cargar los datos de test
datos_test = load('PisosTest.txt');
y_test = datos_test(:,3);  % Precio en Euros
x1_test = datos_test(:,1); % m^2
x2_test = datos_test(:,2); % Habitaciones
N_test = length(y_test);
%% 2 Regresion monovariable utilizando la ecuacion normal
% Dibujar los puntos de entrenamiento
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on;
% Calcular recta 
wMonoNormal = normalEquation(x1, y);
% Dibujar recta
Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * wMonoNormal;
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos EntrenamiSento', 'Prediccion')
% Mostrar error entrenamiento y test
errorTrainMonoNormal = sqrt(sum((y - [ones(N, 1) x1] * wMonoNormal) ...
    .^ 2) / N);
errorTestMonoNormal = sqrt(sum((y_test - [ones(N_test, 1) x1_test] ...
    * wMonoNormal) .^ 2) / N_test);
%% 3 Regresion multivariable utilizando la ecuacion normal
% Calcular plano 
wMultiNormal = normalEquation([x1 x2], y);
% Dibujo de un Ajuste con dos Variables
X = [ones(N,1) x1 x2];
yest = X * wMultiNormal;
% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');
% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);
% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];   
yg = Xg * wMultiNormal;
% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
% Mostrar error entrenamiento y test
errorTrainMultiNormal = sqrt(sum((y - [ones(N, 1) x1 x2] ...
    * wMultiNormal) .^ 2) / N);
errorTestMultiNormal = sqrt(sum((y_test - [ones(N_test, 1) ...
    x1_test x2_test] * wMultiNormal) .^ 2) / N_test);
% 3b
precioMono = [1 100] * wMonoNormal;
precioMulti = [repmat([1 100], 4, 1) (2:5)'] * wMultiNormal;
%% 4 Regresion monovariable utilizando descenso del gradiente
% Calcular recta 
wMonoGradient = gradientDescent(x1, y, @leastSqrsError, 0.0001);
% Dibujar los puntos de entrenamiento
figure;
plot(x1, y, 'bx');
title('Precio de los Pisos')
ylabel('Euros'); xlabel('Superficie (m^2)');
grid on; hold on;
% Dibujar recta
Xextr = [1 min(x1)  % Predicción para los valores extremos
         1 max(x1)];
yextr = Xextr * wMonoGradient;
plot(Xextr(:,2), yextr, 'r-'); % Dibujo la recta de predicción
legend('Datos Entrenamiento', 'Prediccion')
% Show error
errorTrainMonoGradient = sqrt(sum((y - [ones(N,1) x1] * wMonoGradient) ...
    .^ 2) / N);
errorTestMonoGradient = sqrt(sum((y_test - [ones(N_test,1) x1_test] ...
    * wMonoGradient) .^ 2) / N_test);
%% 5 Regresion multivariable utilizando descenso del gradiente
% Calcular plano
wMultiGradient = gradientDescent([x1 x2], y, @leastSqrsError, 0.0001);
% Dibujo de un Ajuste con dos Variables
X = [ones(N,1) x1 x2];
yest = X * wMultiGradient;
% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');
% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);
% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * wMultiGradient;
% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
% Mostrar error entrenamiento y test
errorTrainMultiGradient = sqrt(sum((y - [ones(N, 1) x1 x2] ...
    * wMultiGradient) .^ 2) / N);
errorTestMultiGradient = sqrt(sum((y_test - [ones(N_test, 1) ...
    x1_test x2_test] * wMultiGradient) .^ 2) / N_test);
%% 6 Regresion multivariable utilzando descenso del gradiente con el coste de Huber
% Calcular plano
fun = @(theta, X, y) hubersCost(theta, X, y, 50000);
wMultiGradientHuber = gradientDescent([x1 x2], y, fun, 0.0001);
% Dibujo de un Ajuste con dos Variables
X = [ones(N,1) x1 x2];
yest = X * wMultiGradientHuber;
% Dibujar los puntos de entrenamiento y su valor estimado 
figure;  
plot3(x1, x2, y, '.r', 'markersize', 20);
axis vis3d; hold on;
plot3([x1 x1]' , [x2 x2]' , [y yest]', '-b');
% Generar una retícula de np x np puntos para dibujar la superficie
np = 20;
ejex1 = linspace(min(x1), max(x1), np)';
ejex2 = linspace(min(x2), max(x2), np)';
[x1g,x2g] = meshgrid(ejex1, ejex2);
x1g = x1g(:); %Los pasa a vectores verticales
x2g = x2g(:);
% Calcula la salida estimada para cada punto de la retícula
Xg = [ones(size(x1g)), x1g, x2g];
yg = Xg * wMultiGradientHuber;
% Dibujar la superficie estimada
surf(ejex1, ejex2, reshape(yg,np,np)); grid on; 
title('Precio de los Pisos')
zlabel('Euros'); xlabel('Superficie (m^2)'); ylabel('Habitaciones');
% Mostrar error entrenamiento y test
errorTrainMultiGradientHuber = sqrt(sum((y - [ones(N, 1) x1 x2] ...
    * wMultiGradientHuber) .^ 2) / N);
errorTestMultiGradientHuber = sqrt(sum((y_test - [ones(N_test, 1) ...
    x1_test x2_test] * wMultiGradientHuber) .^ 2) / N_test);

% it = 100000;
% figure;
% ylabel('error test'); xlabel('delta');
% grid on; hold on;
% while it > 0
% fun = @(theta, X, y) hubersCost(theta, X, y, it);
% wMultiGradientHuber = gradientDescent([x1 x2], y, fun, 0.0001);
% errorTestMultiGradientHuber = sqrt(sum((y_test - [ones(N_test, 1) ...
%     x1_test x2_test] * wMultiGradientHuber) .^ 2) / N_test);
% plot(it, errorTestMultiGradientHuber, 'r-'); % Dibujo la recta de predicción
% it = it - 10000;
% end