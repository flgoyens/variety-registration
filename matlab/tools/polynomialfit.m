function [s_min,coeff] = polynomialfit(Y,d)
%{
Finds the coefficients of the degree d polynomial that best fits the point cloud Y.  
Y is n x s and the polynomial has n variables. The coefficients refer to
the monomial basis built in 'alphabetical' order. For 3 variables:
1,x,y,z,x^2,xy,xz,y^2,yz,z^2, x^3, x^2y,x^2z,
xy^2,xyz,xz^2,y^3,y^2z,z^3,... 

The coefficients are taken as the last singular vector of phi(Y)'. 
If the last singular value is zero then the polynomial fits the points
cloud exaclty. With noise this is basically never the case and s_min
returns the singular value associated with the coefficients. The smaller
s_min, the better the polynomial fits the point cloud.
%}
[n,~] = size(Y);
N = nchoosek(n + d,d);
phi = monomials(Y,d);
[~,Sschatten,V] = svd(phi');
s_min = Sschatten(N,end);
coeff = V(:,end);
end