clear all; close all;format long;
% Generate point cloud
% this script is to test different values of the penalty parameter lambda

% s = 100;
n = 10;
sigma = 0.001;
% X = zeros(n,s);
% a = linspace(-1, 1, 0.5*s);
% X(1,:) = [a a];
% X(2,:) = [a -a];
dim_subspace = 2;
n_subspace = 2;
pts_per_sub = 35;
[M,s] = generate_uos(n,dim_subspace,n_subspace,pts_per_sub);

Mhat = M + sigma*randn(size(M));

% figure;
% plot(Xnoise(1,:), Xnoise(2,:),'.r', X(1,:), X(2,:), '*b');
%   leg = legend({'$\hat{M}$ noisy data';'$M$ background truth'},'FontSize',14);
%     set(leg,'Interpreter','latex');
%     return

d = 2; % degree of polynomial
N = nchoosek(n+d,d);
r = N - 1;


% figure;
% plot3(X(1,:), X(2,:), X(3,:),'.b', Xnoise(1,:), Xnoise(2,:), Xnoise(3,:),'.r');
% title('Original data and noisy version');
% legend('Original X','Noisy X');


%      lambda = 0.5*1e-6;
%     verbose = 1;
%    [Y_grass,cost] = approximation_grass(Xnoise,d,lambda,r,verbose);
    [M,error_rank_array, error_noise_array, error_solution, lambda] = increase_lambda(Mhat,d,M);

    error_rank_array
    error_noise_array
    error_solution
    figure(1);
    loglog(lambda, error_noise_array,'r',lambda, error_solution, 'g');
    leg = legend({'$||\hat{M} - X^*||$';'$||M - X^*||$'},'FontSize',14);
    set(leg,'Interpreter','latex');

    figure(2);
    loglog(lambda, error_rank_array, 'b');
    legend('rank error');  %  ;'$||\Phi(X^*) - P_U \Phi(X^*)||$'

%     fprintf('|| M - Y_grass ||_F = %d \n',norm(Y_grass - X,'fro'));
%     fprintf('|| M_hat - Y_grass ||_F = %d \n',norm(Y_grass - Xnoise,'fro'));

%     phi_grass = monomials(Y_grass,d);
%     remainder_grass = norm(surface_coefficients'*phi_grass)/s;
%     fprintf("norm(c'*phi(Y_grass))= %d\n",remainder_grass);


%     hold on;
%     figure;
%     plot(Xnoise(1,:), Xnoise(2,:), '.r',Y_grass(1,:), Y_grass(2,:),'.b');
% %     legend('X original point cloud','Y grass opti');
%     leg = legend({'$\hat{M}$';'$X^*$'},'FontSize',14);
%     set(leg,'Interpreter','latex');
%     title('Grassmannian optimization');
