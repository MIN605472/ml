function [error] = errorFun(lambda, modelo, X, y)
% K = length(modelo);
% N = size(X, 1);
% gis = zeros(N, K);
% error = 1;
% for i = 1:K
%     gis(:, i) = gaussLog(modelo{i}.mu, modelo{i}.Sigma, X);
% end
% % classes = unique(y);
% for i = 1:N
%     error = error + gis(i, y(i));
% %     error = error + 1 - gis(i, 1);
% %     for j = 2:K
% %         error = error + gis(i, j);
% %     end
% end
% error = -1 * error / N;
% 
% Missclasification rate
yHat = clasificacionBayesiana(modelo, X);
error = sum(y ~= yHat) / length(yHat);
end

