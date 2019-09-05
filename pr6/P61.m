%% Practica 6.1: PCA 

clear all;
close all;

% Leer los datos originales en la variable X
load P61
[m, n] = size(X);

% Graficar los datos originales
figure();
axis equal;
grid on;
hold on;
plot3(X(:,1),X(:,2),X(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
% pause

% Estandarizar los datos (solo hace falta centrarlos)
[Xn, mu, sig] = normalizar(X);

% Graficar los datos centrados
figure();
axis equal;
grid on;
hold on;
plot3(Xn(:,1), Xn(:,2), Xn(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');

% Calcular la matrix de covarianza muestral de los datos centrados
Sigma = cov(Xn);

% Aplicar PCA para obtener los vectores propios y valores propios
[V, D] = eig(Sigma);

% Ordenar los vectores y valores preprios de mayor a menor valor propio
[~, idx] = sort(diag(D), 'descend');
V = V(:, idx);
D = D(idx, idx);

% Graficar en color rojo cada vector propio * 3 veces la raiz de su 
% correspondiente valor propio
c = ['r', 'g', 'b'];
for i=1:n
    l = [zeros(3, 1) V(:, i) * 3 * sqrt(D(i, i))];
    plot3(l(1, :), l(2, :), l(3, :), c(i));
end

% Graficar la variabilidad que se mantiene si utilizas los tres primeros
% vectores propios, los dos primeros, o solo el primer vector propio
% var = zeros(n, 1);
% for i=n:-1:1
%     U = V(:, idx(1:i, :));
%     Z = Xn * U;
%     Xn_hat = Z * U';
%     a = sum(sum((Xn - Xn_hat) .^ 2, 2)) / m;
%     b = sum(sum(Xn .^ 2, 2)) / m;
%     var(n - i + 1) = a / b;
% end
% x = (n:-1:1)';
% figure();
% grid on;
% hold on;
% plot(x, 1 - var,'ob-');
% xlabel('k');
% ylabel('var');

figure();
grid on;
hold on;
var = zeros(n, 1);
dia = diag(D);
for i=1:n
    var(i) = sum(dia(1:i)) / sum(dia);
end
plot((1:n)', var,'ob-');
xlabel('k');
ylabel('var');

% Aplicar PCA para reducir las dimensiones de los datos y mantener al menos
% el 90% de la variabilidad
U = V(:, 1:2);
Z = Xn * U;

% Graficar aparte los datos z proyectados seg�n el resultado anterior
figure();
axis equal;
grid on;
hold on;
plot(Z(:,1), Z(:,2),'b.');
xlabel ('X');
ylabel ('Y');

% Graficar en verde los datos reproyectados \hat{x} en la figura original
Xn_hat = Z * U';
figure();
axis equal;
grid on;
hold on;
plot3(Xn(:,1), Xn(:,2), Xn(:,3),'b.');
xlabel ('X');
ylabel ('Y');
zlabel ('Z');
plot3(Xn_hat(:,1), Xn_hat(:,2), Xn_hat(:,3),'g.');
