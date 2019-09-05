% close; clear;
figure()
im = imread('smallparrot.jpg');
imshow(im)
%% datos
D = double(reshape(im,size(im,1)*size(im,2),3));
% [D, mu, sig] = normalizar(D);
% figure(1);
% axis equal;
% grid on;
% hold on;
% plot3(D(:,1),D(:,2),D(:,3),'b.');
% xlabel ('X');
% ylabel ('Y');
% zlabel ('Z');
% Sigma = cov(D);
% [V, U] = eig(Sigma);
% [~, idx] = sort(diag(U), 'descend');
% V = V(:, idx);
% U = U(idx, idx);
% c = ['r', 'g', 'y'];
% for i=1:3
%     l = [zeros(3, 1) V(:, i) * 3 * sqrt(U(i, i))];
%     plot3(l(1, :), l(2, :), l(3, :), c(i));
% end
%% dimensiones
m = size(D,1);
n = size(D,2);

%% Kmeans 
K = 16;

%% Inicializar centroides
mu0 = pcaCentroids(D, K);

%% bucle kmeans
[mu, c] = kmeans(D, mu0, K);

%% reconstruir imagen
qIM=zeros(length(c),3);
for h=1:K
    ind=find(c==h);
    qIM(ind,:)=repmat(mu(h,:),length(ind),1);
end
qIM=reshape(qIM,size(im,1),size(im,2),size(im,3));
figure()
imshow(uint8(qIM));
%% mostrar ratio de compresion
figure();
loglog((24 * m * n) ./ (log2(1:m) * m * n + 24 * (1:m)));
xlabel('k');
ylabel('compression ratio');

%% error reconstruccion
error = reconstructionError(D, mu, c);