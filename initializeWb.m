function [W, b, cost] = initializeWb(W, b, data, H, stepGradW, maxiter)
%
%
%
%
%
% Shu Kong (Aimery)
% aimerykong@gmail.com
% Sep. 5, 2013
% 

%%
if nargin < 6
    maxiter = 600;
end

if nargin < 5
    stepGradW = 0.0001;
end

%%
lambdaW = 0.0000001;
M = size(data,2);

cost = zeros(1, maxiter);
Actv = sigmoid( W*data+repmat(b, 1, size(data, 2)) );

%%
for i = 1:maxiter
    if mod(i, 200) == 0
        fprintf('\t\t\t%d  cost = %.20f \n', i, cost(i-1));
    end
    
    Wgrad = 2*( ...
                (Actv-H).*(Actv.*(1-Actv))...
                ) * data';
    bgrad = 2*sum( (Actv-H) .* ( Actv.*(1-Actv) ), 2 );

    W = W - stepGradW*( Wgrad/M + 2*lambdaW*W);
    b = b - stepGradW/M*bgrad;
%    W = W - (8^floor(i/100))*stepGradW*Wgrad;
%    b = b - (8^floor(i/100))*stepGradW*bgrad;

    Actv = sigmoid( W*data+repmat(b, 1, size(data, 2)) );
    cost(i) = sum(sum((H - Actv).^2)) + lambdaW*sum(sum(W.^2));
end

fprintf('finished!\n')


