% readC.m

Xc = load('Xc.txt');

Nx = length(Xc);

NumEq = 3;

Qbig = load('Q.txt');

Q = zeros(Nx,NumEq);
Q1 = zeros(1,Nx);
Q2 = zeros(1,Nx);
Q3 = zeros(1,Nx);

count = 1;
for i = 1:Nx
    for n = 1:NumEq
        Q(i,n) = Qbig(count);
        count = count + 1;
    end
end

Q1 = Q(:,1)';
Q2 = Q(:,2)';
Q3 = Q(:,3)';