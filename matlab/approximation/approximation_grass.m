function [Y,xcost, error_rank] = approximation_grass(X,d, lambda,r,verbose)
% X is a cloud of data point that (possibly approximately/noise) belongs to
% a manifold.
% This function solves the problem
% min f(Y,U) = trace( k_d(Y) - P_U k_d(Y) ) + lambda*||X-Y||^2_F
%             st. U in Grass(s,r)
%                 Y in R^{3 x s}
%
% note that P_U = U*((U'*U)\U' wich simplifies to U*U' when U is orthogonal

[n,s] = size(X);

problem.M = productmanifold(struct('Y', euclideanfactory(n,s),...
    'U', grassmannfactory(s, r)));

kernel = @kernel_uos;
% Define the problem cost function, gradient and action of Hessian

problem.cost  =  @(x) trace( (eye(s,s) - x.U*x.U')*kernel_uos(x.Y,d)) ...
                        + lambda*norm(x.Y-X,'fro')^2;

problem.egrad = @(x) struct('Y',2*d*x.Y*(kernel_uos(x.Y,d-1).* (eye(s,s)- x.U*x.U'))...
                                + 2*lambda*(x.Y - X),...
                                'U',-2*kernel(x.Y,d)*x.U );

problem.ehess = @(x, xdot) struct('Y',2*(d-1)*d*x.Y*(kernel_uos(x.Y,d-2).* (eye(s,s)- x.U*x.U').* ( x.Y'*xdot.Y + xdot.Y'*x.Y) )...
    +2*d*xdot.Y*(kernel_uos(x.Y,d-1).*(eye(s,s)- x.U*x.U'))...
    -2*d*x.Y*(kernel_uos(x.Y,d-1).*(x.U*xdot.U' + xdot.U*x.U' ))...
    + 2*lambda* xdot.Y,...
    'U',-2*kernel(x.Y,d)*xdot.U  -2*d*(kernel_uos(x.Y,d-1).*( x.Y'*xdot.Y + xdot.Y'*x.Y  ))*x.U);

% Numerically check gradient and Hessian consistency.
% figure;
% checkgradient(problem);
% figure;
% checkhessian(problem);

   x0.Y = X;
%  x0.Y = Xreal;
x0.U = svd_r(kernel_uos(x0.Y,d),r);

options.tolgradnorm = 1e-6;
options.maxiter = 50;
options.verbosity = verbose;
% Solve.

%  [x, xcost, info] = steepestdescent(problem,x0,options);
  [x, xcost, info] = trustregions(problem,x0,options);
%    [x, xcost, info] = trustregions(problem,[],options);
 Y = x.Y;

error_rank = trace( (eye(s,s) - x.U*x.U')*kernel_uos(x.Y,d));

end
