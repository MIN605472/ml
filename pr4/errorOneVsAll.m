function [error] = errorOneVsAll(~, weights, X, y)
[N, ~] = size(X);
classes = unique(y);
% K = length(classes);
error = 0;
for i = classes'
    theta = weights(:, i);
    h = 1./(1 + exp(-(X * theta)));
    y_i = y == i;
%     J = (-y_i(h ~= 0 & h ~= 1)' * log(h(h ~= 0 & h ~= 1)) - (1 - y_i(h ~= 0 & h ~= 1)') * log(1 - h(h ~= 0 & h ~= 1)));
%     J = J + (-y_i(h == 1)' * log(h(h == 1)));
%     J = J + (-(1 - y_i(h == 0)') * log(1 - h(h == 0)));
    J =(-y_i'*log(h) - (1-y_i')*log(1-h))/N;
    error = error + J;
end
% error = error / N;
end