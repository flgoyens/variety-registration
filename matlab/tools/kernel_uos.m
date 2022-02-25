function out = kernel_uos(x,d,c)
if nargin == 2
    c = 1; 
end
out = (x'*x+c).^d;
end