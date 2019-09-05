function [ wdes ] = desnormalizar( w, mu, sig )
% Desnormaliza los pesos de la regresion
% wdes = w(2:end)./sig';
% wdes = [w(1)-(mu*wdes); wdes];

wdes = w .* repmat(sig, size(w, 1), 1) + repmat(mu, size(w, 1), 1);

end

