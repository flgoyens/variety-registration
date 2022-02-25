function [AA,aa] = registration_manopt(X1,X2,d)
% X1 and A2 are clouds of data point that (possibly approximately/noise) belongs to
% the same manifold but they have been measured from a different angle, we have to match them and find
% rotation A and shift a sucg that X1 = A*X2 + a.
% This function solves the problem
% min f(A,a,U) = trace(P_Up k_d([X1 x.A*X2+x.a*ones(1,s2)]))
%             st. U in Grass(s,r)
%                 A in St^{3 x 3} (orthogonal group)
%                 a in R^3.
%
% note that P_U = U*((U'*U)\U' wich simplifies to U*U' when U is orthogonal
% and P_Up is Id - P_U, the projection on the orthogonal complement of U.

[~,s1] = size(X1);
[n,s2] = size(X2);
if s2 ~= s1
    fprintf('ATTENTION: the two point clouds do not have the same dimension, is this ok ?\n');
    s = min(s1,s2);
end
s = s1+s2;

N = nchoosek(n+d,d);
r = N - 1;


problem.M = productmanifold(struct('A', stiefelfactory(3,3),'a', euclideanfactory(3,1),...
    'U', grassmannfactory(s, r)));


% Define the problem cost function, gradient and action of Hessian

function store = prepare(x, store)
        if ~isfield(store, 'PUorth')
            store.PUorth = eye(s,s)- x.U*x.U';
        end
    end

kernel = @kernel_uos;
problem.cost  =  @cost;
    function [fx,store] = cost(x,store)
        store = prepare(x, store);
        fx = trace( store.PUorth*kernel_uos([X1 x.A*X2+x.a*ones(1,s2)],d));
        store = incrementcounter(store, 'costcalls');
    end

problem.egrad = @egrad;
    function [gx,store] = egrad(x,store)
        store = prepare(x, store);
        grad = [X1 x.A*X2+x.a*ones(1,s2)]*(store.PUorth.*kernel_uos([X1 x.A*X2+x.a*ones(1,s2)],d));
        grad = grad(:,s1+1:end);
        gx = struct('A', (grad)*X2',...
                            'a', (grad)*ones(s2,1),...
                            'U',-2*kernel([X1 x.A*X2+x.a*ones(1,s2)],d)*x.U );
        store = incrementcounter(store, 'gradcalls');
    end

% problem.ehess = @(x, xdot) struct('Y',2*(d-1)*d*x.Y*(kernel_uos(x.Y,d-2).* (eye(s,s)- x.U*x.U').* ( x.Y'*xdot.Y + xdot.Y'*x.Y) )...
%     +2*d*xdot.Y*(kernel_uos(x.Y,d-1).*(eye(s,s)- x.U*x.U'))...
%     -2*d*x.Y*(kernel_uos(x.Y,d-1).*(x.U*xdot.U' + xdot.U*x.U' ))...
%     + 2*lambda* xdot.Y,...
%     'U',-2*kernel(x.Y,d)*xdot.U  -2*d*(kernel_uos(x.Y,d-1).*( x.Y'*xdot.Y + xdot.Y'*x.Y  ))*x.U);

% Numerically check gradient and Hessian consistency.
figure;
checkgradient(problem);
% figure;
% checkhessian(problem);

  x0.A = eye(3,3);
  x0.a = ones(s2,1);
  x0.U = svd_r(randn(s,s),r);

options.tolgradnorm = 1e-5;
options.maxiter = 2000;
options.verbosity = 1;
% Solve.

%   [x, xcost, info] = trustregions(problem,x0,options);
    [x, xcost, info] = steepestdescent(problem,[],options);
%   [x, xcost, info] = trustregions(problem,[],options);
 AA = x.A;
 aa = x.a;


% Display some statistics.
figure;
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration #');
ylabel('Gradient norm');
title('Convergence of the trust-regions algorithm');
end
