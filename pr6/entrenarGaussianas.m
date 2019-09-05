function modelo = entrenarGaussianas( Xtr, ytr, nc, NaiveBayes, landa )
% Entrena una Gaussana para cada clase y devuelve:
% modelo{i}.N     : Numero de muestras de la clase i
% modelo{i}.mu    : Media de la clase i
% modelo{i}.Sigma : Covarianza de la clase i
% Si NaiveBayes = 1, las matrices de Covarianza serán diagonales
% Se regularizarán las covarianzas mediante: Sigma = Sigma + landa*eye(D)
for i = 1:nc
    ytr_i = ytr == i;
    modelo{i}.N = sum(ytr_i);
    modelo{i}.mu = mean(Xtr(ytr_i, :))';
    modelo{i}.Sigma = cov(Xtr(ytr_i, :));
%     modelo{i}.Sigma = (1 - landa) * modelo{i}.Sigma + landa * diag(diag(modelo{i}.Sigma));
    modelo{i}.Sigma = (1-landa) * modelo{i}.Sigma + landa * eye(size(modelo{i}.Sigma));
    if NaiveBayes == 1 
        modelo{i}.Sigma = diag(diag(modelo{i}.Sigma));
    end
end
end