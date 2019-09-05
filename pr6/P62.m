%% Practica 6.2: PCA 

clear all
close all

% Leer la imagen
I = imread('turing.png');

% Convertirla a blanco y negro
BW = rgb2gray(I);

% Convertir los datos a double
X=im2double(BW);

% graficar la imagen
figure();
colormap(gray);
imshow(X);
axis off;
% pause

% Aplicar PCA
[U,S,V] = svd(X);
%% Graficar las primeras 5 componentes
for k = 1:5
    figure();
    Xhat = U(:, k) * S(k, k) * V(:, k)';
    imshow(Xhat);
    colormap(gray);
    axis off;
%     pause
end

%% Graficar la reconstrucci�n con las primeras 1, 2, 5, 10, 20, y total
% de componentes
for k = [1 2 5 10 20 rank(X)]
    figure();
    Xhat = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
    imshow(Xhat);
    colormap(gray);
    axis off;
%     pause
end

% Encontrar el valor de k que mantenga al menos el 90% de la variabilidad
dia = diag(S);
d = length(dia);
var = zeros(d, 1);
for i=1:d
    var(i) = sum(dia(1:i)) / sum(dia);
end
figure();
grid on;
hold on;
plot(var, 'xb-');
xlabel('k');
ylabel('var');
k = find(var >= 0.9, 1);

% Graficar la reconsrtucci�n con las primeras k componentes
figure();
Xhat = U(:, 1:k) * S(1:k, 1:k) * V(:, 1:k)';
imshow(Xhat);
colormap(gray);
axis off;

% Calcular y mostrar el ahorro en espacio
figure();
[m, n] = size(X);
plot((m * n) ./ ((1:n) * (m + n + 1)));
xlabel('k');
ylabel('compression ratio');