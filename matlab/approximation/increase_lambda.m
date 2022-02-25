function [X,error_rank_array, error_noise_array, error_solution, lambda] = increase_lambda(Mhat,d,M)
% Solves the approximation problem using the Grassmanian formulation for a decreasing sequence of the penalty
% parameter lambda using the previous solution as a warm start. The aim is
% to find the best value of lambda.
[n,s] = size(Mhat);
N = nchoosek(n+d,d);
r = N - 1;
lambda = [1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1 10 100 1000 1e4 1e5 1e6 1e7];
trials = length(lambda);
error_rank_array = zeros(trials,1);
error_noise_array = zeros(trials,1);
error_solution = zeros(trials,1);

Y_grass = Mhat;

verbose = 1;
for i = 1:trials
    [Y_grass,~, error_rank] = approximation_grass(Y_grass,d,lambda(i),r,verbose);
%     [Y_grass,~, error_rank] = approximation_grass(randn(n,s),d,lambda(i),r,verbose);
    error_rank_array(i) = error_rank;
    error_noise_array(i) = norm(Y_grass - Mhat, 'fro')^2;
    error_solution(i) = norm(Y_grass - M,'fro')^2;
    fprintf("lambda = %d and rank_error = %d\n",lambda(i), error_rank);
end

X = Y_grass;
end
