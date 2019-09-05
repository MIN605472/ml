function [tp, fp, fn, tn] = calcConfMat(X, y, fun)
tp = 0;
fp = 0;
fn = 0;
tn = 0;
for rowNDX = 1:size(X, 1)
    row = X(rowNDX, :)';
    expectedOutput = y(rowNDX, 1);
    predictedOutput = fun(row);
    if expectedOutput == predictedOutput
        if expectedOutput == 1
            tp = tp + 1;
        else 
            tn = tn + 1;
        end
    elseif predictedOutput == 1
        fp = fp + 1;
    else
        fn = fn + 1;
    end
end

end

