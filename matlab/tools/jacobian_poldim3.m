function jacobian = jacobian_poldim3(a,d)
%{
returns the gradient of phi_d(X) as a Nx3 matrix with the first column
being the gradient in x, second column in y then z.
%}
[n,~] = size(a);
N = nchoosek(n+d,d);
jacobian = zeros(N,d);
x = a(1);
y = a(2);
z = a(3);
if(d==1)
    jacobian = [0 1 0 0; %D_x phi 
                0 0 1 0; %D_y phi
                0 0 0 1]'; %D_z phi
elseif(d==2)
    jacobian = [ 0 1 0 0 2*x y z 0 0 0;
                 0 0 1 0 0 x 0 2*y z 0;
                 0 0 0 1 0 0 x 0 y 2*z;]';
elseif(d==3)
%     phi = [1 x y z...
%         x*x x*y x*z y*y y*z z*z...
%         x^3 x^2*y x^2*z x*y^2 x*y*z x*z^2 y^3 y^2*z y*z^2 z^3]';
    
    jacobian = [ 0 1 0 0 2*x y z 0 0 0 3*x^2 2*x*y 2*x*z y^2 y*z z^2 0 0 0 0;
                 0 0 1 0 0 x 0 2*y z 0 0 x^2 0 2*x*y x*z 0 3*y^2 2*y*z z^2 0;
                 0 0 0 1 0 0 x 0 y 2*z 0 0 x^2 0 x*y 2*z*x 0 y^2 2*y*z 3*z^2;]';
elseif(d==4)
%     phi = [1 x y z...
%         x*y x*z y*z x*x y*y z*z...
%         x^3 y^3 z^3 x*y^2 x*y*z x*z^2 x^2*y x^2*z y^2*z z^2*y...
%         x^4 x^3*y x^3*z x^2*y^2 x^2*y*z x^2*z^2 x*y^3 x*y^2*z x*y*z^2 x*z^3 y^4  y^3*z y*z^3 y^2*z^2 z^4]';
    fprintf('degree  4 and above not yet umplemented, ERROR\n');
 elseif(d==5)
%     phi = [1 x y z...
%         x*y x*z y*z x*x y*y z*z...
%         x^3 y^3 z^3 x*y^2 x*y*z x*z^2 x^2*y x^2*z y^2*z z^2*y...
%         x^4 x^3*y x^3*z x^2*y^2 x^2*y*z x^2*z^2 x*y^3 x*y^2*z x*y*z^2 x*z^3 y^4  y^3*z y*z^3 y^2*z^2 z^4 ...
%         x^5 x^4*y x^4*z x^3*y^2 x^3*y*z x^3*z^2 x^2*y^3 x^2*y^2*z x^2*y*z^2 x^2*z^3 x*y^4 x*y^3*z x*y^2*z^2 x*y*z^3 ...
%         x*z^4 y^5 y^4*z y^3*z^2 y^2*z^3 y*z^4 z^5]';
    fprintf('degree  4 and above not yet umplemented, ERROR\n');
else
    fprintf('degree  4 and above not yet umplemented, ERROR\n');
end
end