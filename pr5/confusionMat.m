function [confMat] = confusionMat(y, yhat)
classes = unique(y);
K = length(classes);
confMat = zeros(K, K);
for t = classes'
    tPos = y == t;
    pValues = yhat(tPos, :);
    for c = classes'
        confMat(t, c) = confMat(t, c) + sum(pValues == c);
    end
end
end

