% first attempt at registration using optimization and monomial features
close all;

% Generate point cloud
s = 50;
X1 = zeros(3,s);

% X(1,:) = randn(1,s);
% X(2,:) = randn(1,s);

X1(1,:) = 1*(rand(1,s)-0.5*ones(1,s));
X1(2,:) = 1*(rand(1,s)-0.5*ones(1,s));

X1(3,:) = -X1(1,:).^2 + -X1(2,:).^2 ; % 3D parabola z = x^2 + y^2
% X(3,:) = X(1,:).^3 + X(2,:).^2 + X(1,:).*X(2,:) ; % surface z = x^3 + y^2 + xy

A = orth(randn(3,3));
a = 1*rand(3,1)*ones(1,s);

X2 = A*X1+a;

XX = [X1 X2];


figure;
plot3(X1(1,:), X1(2,:), X1(3,:),'.b', X2(1,:), X2(2,:), X2(3,:),'.r');
title('Original data and Transformed version');
legend('Original X','A*X+a');

d = 2;
n = 3;
N = nchoosek(n+d,d);
r = N - 1;

phiXX = monomials(XX,d);
phiX2 = monomials(X2,d);
phiX1 = monomials(X1,d);

rankXX = rank(phiXX);
rankX1 = rank(phiX1);
rankX2 = rank(phiX2);
fprintf('rank(phi XX) = %d\n', rankXX);
fprintf('rank(phi X1) = %d\n', rankX1);
fprintf('rank(phi X2) = %d\n', rankX2);

[s_min,coeff] = sigma_min(X1,d);

% Optimize over a and A
[AA,aa] = registration_manopt(X1,X2,d);

distA = norm(AA-A,'fro')
dista = norm(a - aa,'fro')

[s_min,coeff] = sigma_min([X1 AA*X2+aa*ones(1,s)],d)
