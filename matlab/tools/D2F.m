function HessF = D2F(a,c,d)
HessF = zeros(3,3);
x = a(1); y = a(2); z = a(3);
if(d==1)
    %     jacobian = [0 1 0 0; %D_x phi
    %                 0 0 1 0; %D_y phi
    %                 0 0 0 1]'; %D_z phi
    HessF = zeros(3,3);
elseif(d==2)
    %     jacobian = [ 0 1 0 0 2*x y z 0 0 0;    %D_x phi
    %                  0 0 1 0 0 x 0 2*y z 0;    %D_y phi
    %                  0 0 0 1 0 0 x 0 y 2*z;]'; %D_z phi

    HessF(1,1) = c'*[0 0 0 0 2 0 0 0 0 0]'; %D_x D_x phi
    HessF(2,1) = c'*[0 0 0 0 0 1 0 0 0 0]'; %D_y D_x phi
    HessF(3,1) = c'*[0 0 0 0 0 0 1 0 0 0]'; %D_z D_x phi
    
    HessF(1,2) = c'*[0 0 0 0 0 1 0 0 0 0]'; %D_x D_y phi
    HessF(2,2) = c'*[0 0 0 0 0 0 0 2 0 0]'; %D_y D_y phi
    HessF(3,2) = c'*[0 0 0 0 0 0 0 0 1 0]'; %D_z D_y phi
    
    HessF(1,3) = c'*[0 0 0 0 0 0 1 0 0 0]'; %D_x D_z phi
    HessF(2,3) = c'*[0 0 0 0 0 0 0 0 1 0]'; %D_y D_z phi
    HessF(3,3) = c'*[0 0 0 0 0 0 0 0 0 2]'; %D_z D_z phi
    
elseif(d==3)
    %     phi = [1 x y z...
    %         x*x x*y x*z y*y y*z z*z...
    %         x^3 x^2*y x^2*z x*y^2 x*y*z x*z^2 y^3 y^2*z y*z^2 z^3]';
    
    %     jacobian = [ 0 1 0 0 2*x y z 0 0 0    3*x^2 2*x*y 2*x*z y^2 y*z z^2 0 0 0 0; %D_x phi
    %                  0 0 1 0 0 x 0 2*y z 0    0 x^2 0 2*x*y x*z 0 3*y^2 2*y*z z^2 0; %D_y phi
    %                  0 0 0 1 0 0 x 0 y 2*z    0 0 x^2 0 x*y 2*z*x 0 y^2 2*y*z 3*z^2;]'; %D_z phi
    
    HessF(1,1) = c'*[0 0 0 0 2 0 0 0 0 0   6*x 2*y 2*z 0 0 0 0 0 0 0]'; %D_x D_x phi
    HessF(2,1) = c'*[0 0 0 0 0 1 0 0 0 0   0 2*x 0 2*y z 0 0 0 0 0  ]'; %D_y D_x phi
    HessF(3,1) = c'*[0 0 0 0 0 0 1 0 0 0   0 0 2*x 0 y 2*z 0 0 0 0  ]'; %D_z D_x phi ok
    
    HessF(1,2) = c'*[0 0 0 0 0 1 0 0 0 0   0 2*x 0 2*y z 0 0 0 0 0  ]'; %D_x D_y phi
    HessF(2,2) = c'*[0 0 0 0 0 0 0 2 0 0   0 0 0 2*x 0 0 6*y 2*z 0 0]'; %D_y D_y phi
    HessF(3,2) = c'*[0 0 0 0 0 0 0 0 1 0   0 0 0 0 x 0 0 2*y 2*z 0  ]'; %D_z D_y phi ok
    
    HessF(1,3) = c'*[0 0 0 0 0 0 1 0 0 0   0 0 2*x 0 y 2*z 0 0 0 0  ]'; %D_x D_z phi
    HessF(2,3) = c'*[0 0 0 0 0 0 0 0 1 0   0 0 0 0 x 0 0 2*y 2*z 0  ]'; %D_y D_z phi
    HessF(3,3) = c'*[0 0 0 0 0 0 0 0 0 2   0 0 0 0 0 2*x 0 0 2*y 6*z]'; %D_z D_z phi ok
    
end