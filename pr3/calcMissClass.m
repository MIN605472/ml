function [nmissclass] = calcMissClass(X, y, fun)
nmissclass = 0;
for rowNDX = 1:size(X, 1)
    row = X(rowNDX, :)';
    out = y(rowNDX, 1);
    prob = fun(row);
    if (prob >= 0.5 && out == 0) || (prob < 0.5 && out == 1)
        nmissclass = nmissclass + 1; 
    end
end
nmissclass = nmissclass / length(y);
end

