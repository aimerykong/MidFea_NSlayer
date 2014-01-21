function [W, b, cost] = updateWb(W, b, data, H, stepGradW, maxiter)
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
    stepGradW = 0.001;
end

%%
epsilon = 0.000001;
lambdaW = 0.00001;
M = size(data,2);

cost = zeros(1, maxiter);
Actv = sigmoid( W*data+repmat(b, 1, size(data, 2)) );

Wgrad = 2*( (Actv-H).*(Actv.*(1-Actv)) ) * data';
Wgrad = Wgrad/M + 2*lambdaW*W;
a = max(abs(Wgrad(:)));
b = mean(abs(Wgrad(:)));
%stepGradW = 1/a/100;

%%
for i = 1:maxiter    
%     Wgrad = 2* ( (Actv-H).*(Actv.*(1-Actv))) * data';
%     bgrad = 2*sum( (Actv-H) .* ( Actv.*(1-Actv) ), 2 );
% 
%     W = W - stepGradW*(Wgrad + 2*lambdaW*W);
%     b = b - stepGradW*bgrad;  
    
    %W = W - (10^floor(i/100))*stepGradW*Wgrad;
    %b = b - (10^floor(i/100))*stepGradW*bgrad; 
    
    Wpre = W;
    
    Wgrad = 2*( ...
                (Actv-H).*(Actv.*(1-Actv))...
                ) * data';
    bgrad = 2*sum( (Actv-H) .* ( Actv.*(1-Actv) ), 2 );

    W = W - stepGradW*( Wgrad/M + 2*lambdaW*W);
    b = b - stepGradW/M*bgrad;
    
    Actv = sigmoid( W*data+repmat(b, 1, size(data, 2)) );
    cost(i) = sum(sum((H - Actv).^2)) + lambdaW*sum(sum(W.^2));
    
%     if mod(i, 200) == 0
%         fprintf('\t\t\t%d  cost = %.20f \n', i, cost(i-1));
%         t = norm(Wpre-W,'fro');
%     end
    
    if norm(Wpre-W,'fro') <= epsilon
        fprintf('!!!\n');
        cost = cost(1:i);
        i = maxiter+1;
        return;
    end
end
