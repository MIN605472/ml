function [weights] = gradientDescent(trX, trR, costFun, alpha)
N = size(trX, 1); % num. of instances
F = size(trX, 2); % num. of features
[scaledTrX, mu, sigma] = scaleFeatures(trX);
w = zeros(F + 1, 1);
X = [ones(N, 1) scaledTrX];
% alpha = 0.0001;
it = 0;
figure;
title('Progreso del error')
ylabel('Error'); xlabel('Iteracion');
grid on;
hold on;
while true
%     nabla = (X' * X * w - X' * trR);
%     e0 = 1 / 2 * sum(((X * w  - trR) .^ 2));
    [e0, nabla] = costFun(w, X, trR);
    w = w - alpha * nabla;
%     e1 = 1 / 2 * sum(((X * w  - trR) .^ 2));
    e1 = costFun(w, X, trR);
    plot(it, e1, 'r.-');
    it = it + 1;
    if e1 / e0 >= 1
        break;
    end
end
w0 = w(1,1) - sum(w(2:end, :) .* mu ./ sigma);
weights = [w0; (w(2:end, :) ./ sigma)];
end