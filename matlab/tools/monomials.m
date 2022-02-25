function [phi,N] = monomials(X,d)

[n,s] = size(X);
N = nchoosek(n + d,d);
phi = zeros(N,s);

if(n==2)
    for i =1:s
        phi(:,i) = poldim2(X(:,i),d);
    end
elseif(n==3)
    for i =1:s
        phi(:,i) = poldim3(X(:,i),d);
    end
end



end