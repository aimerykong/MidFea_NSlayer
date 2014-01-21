function ProjX = RandProj(p, d)
ProjX = rand(p, d);

ProjX = ProjX - ones(p,1)*mean(ProjX);

B = ones(p,1)*sum(ProjX.^2);
B(B==0) = 1;
ProjX = ProjX ./ sqrt(B);


ProjX = ProjX';




