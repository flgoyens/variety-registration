function pol = poldim2(x,d)
pol = 0;
if(d==1)
    pol = [1 x y z]';
elseif(d==2)
    pol = [1 x(1) x(2)...
        x(1)*x(2) x(1)*x(1) x(2)*x(2)]';
elseif(d==3)
    pol = [1 x(1) x(2)...
        x(1)*x(1) x(1)*x(2) x(2)*x(2)...
        x(1)*x(1)*x(1) x(1)*x(1)*x(2) x(1)*x(2)*x(2) x(2)*x(2)*x(2) ]';
elseif(d==4)
    pol = [1 x(1) x(2)...
        x(1)*x(1) x(1)*x(2) x(2)*x(2)...
        x(1)*x(1)*x(1) x(1)*x(1)*x(2) x(1)*x(2)*x(2) x(2)*x(2)*x(2)...
        x(1)*x(1)*x(1)*x(1) x(1)*x(1)*x(1)*x(2) x(1)*x(1)*x(2)*x(2) x(2)*x(2)*x(2)*x(2)]';
else
    fprintf('degree above 4  not yet implemented, ERROR\n');
end
end