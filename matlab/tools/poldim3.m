function pol = poldim3(X,d)
pol = 0;
x = X(1);
y = X(2);
z = X(3);
if(d==1)
    pol = [1 x y z]';
elseif(d==2)
    pol = [1 x y z...
        x*x x*y x*z y*y y*z z*z]';
elseif(d==3)
    pol = [1 x y z...
            x*x x*y x*z y*y y*z z*z...
            x^3 x^2*y x^2*z x*y^2 x*y*z x*z^2 y^3 y^2*z y*z^2 z^3]';
elseif(d==4)
    pol = [1 x y z...
            x*y x*z y*z x*x y*y z*z...
            x^3 y^3 z^3 x*y^2 x*y*z x*z^2 x^2*y x^2*z y^2*z z^2*y...
            x^4 x^3*y x^3*z x^2*y^2 x^2*y*z x^2*z^2 x*y^3 x*y^2*z x*y*z^2 x*z^3 y^4  y^3*z y*z^3 y^2*z^2 z^4]';
elseif(d==5)
    pol = [1 x y z...
            x*y x*z y*z x*x y*y z*z...
            x^3 y^3 z^3 x*y^2 x*y*z x*z^2 x^2*y x^2*z y^2*z z^2*y...
            x^4 x^3*y x^3*z x^2*y^2 x^2*y*z x^2*z^2 x*y^3 x*y^2*z x*y*z^2 x*z^3 y^4  y^3*z y*z^3 y^2*z^2 z^4 ...
            x^5 x^4*y x^4*z x^3*y^2 x^3*y*z x^3*z^2 x^2*y^3 x^2*y^2*z x^2*y*z^2 x^2*z^3 x*y^4 x*y^3*z x*y^2*z^2 x*y*z^3 ...
            x*z^4 y^5 y^4*z y^3*z^2 y^2*z^3 y*z^4 z^5]';
else
    fprintf('degree  6 and above not yet implemented, ERROR\n');
end
end