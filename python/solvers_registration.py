#!/usr/bin/python3
import pymanopt
import autograd.numpy as np
# import numpy as np
from numpy import random as rnd
from numpy import linalg as la
from numpy.linalg import svd

from pymanopt.manifolds import Product, Grassmann, SpecialOrthogonalGroup, Euclidean
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions
from registration_tools import ncr, monomial_features_n2_d2, monomial_features_n3_d2, monomial_kernel, truncate_svd


def registration_n2(M1hat, M2hat, d=2, mu=1e-5):
    # Finds a rotation Q and translation a such that
    # M2 = Q*M1 + a
    n, s1 = M1hat.shape
    n, s2 = M2hat.shape
    N = ncr(n+d, d)
    r = N-1
    x01 = (M1hat, truncate_svd(monomial_features_n2_d2(M1hat), r))
    x02 = (M2hat, truncate_svd(monomial_features_n2_d2(M2hat), r))
    Xopt1, flag = smoothing_n2(M1hat, d, mu, x01)
    X1 = Xopt1[0]
    U1 = Xopt1[1]
    Xopt2, flag = smoothing_n2(M2hat, d, mu, x02)
    X2 = Xopt2[0]
    U2 = Xopt2[1]
    P_U2perp = np.identity(N) - U2@U2.T

    manifold = Product((SpecialOrthogonalGroup(n), Euclidean(n)))
    @pymanopt.function.Autograd
    def cost_registration2(Q, a):
        A = P_U2perp@monomial_features_n2_d2(Q@X1 + np.array([a, ]*s1).T)
        return np.trace(A.T@A)
    problem = Problem(manifold=manifold, cost=cost_registration2)
    problem.verbosity = 1
    solver = TrustRegions(maxiter=1000, mingradnorm=1e-07, logverbosity=2)

    f_best = 1000
    Xbest = None
    i = 0
    check = 1
    while(i < 10 and check):  # this loop just tries 20 random starting points
        Xopt, optlog = solver.solve(problem)
        final_log = optlog['final_values']
        if(final_log['f(x)'] <= 1e-2):
            print("Registration SOLVED: Solver found cost f(Q,a)<= 1e-2 (attempts: "+str(i) + ")")
            print('f_end = ', final_log['f(x)'])
            Qend = Xopt[0]
            aend = Xopt[1]
            return (Qend, aend, U2, X1, X2)
            check = 0
        else:
            print('registration restart: f = ', final_log['f(x)'])
            if(final_log['f(x)'] <= f_best):
                Xbest = Xopt
                f_best = final_log['f(x)']
            i = i + 1
    print("Registration FAILED: Optimizer did not find solution (attempts: "+str(i) + ")")
    Qend = Xbest[0]
    aend = Xbest[1]
    return (Qend, aend, U2, X1, X2)


def smoothing_n2(Mhat, d, mu=0, x0=None):
    n, s = Mhat.shape
    N = ncr(n + d, d)
    Phi_M = monomial_features_n2_d2(Mhat)
    r = min(la.matrix_rank(Phi_M, 1e-10), N-1)
    manifold = Product((Euclidean(n, s), Grassmann(N, r)))
    @pymanopt.function.Autograd
    def cost_smoothing2(X, U):
        Phi = monomial_features_n2_d2(X)
        A = Phi - U@(U.T@Phi)
        return np.trace(A.T@A) + mu*np.trace((X-Mhat).T@(X-Mhat))
    problem = Problem(manifold=manifold, cost=cost_smoothing2)
    problem.verbosity = 2
    # XU0 = (M, svd_r(Phi_M))
    solver = TrustRegions(maxiter=1000, mingradnorm=1e-07, logverbosity=2)
    XUopt, optlog = solver.solve(problem, x0)
    final_log = optlog['final_values']
    Xend = XUopt[0]
    Uend = XUopt[1]
    Phi = monomial_features_n2_d2(Xend)
    A = Phi - Uend@(Uend.T@Phi)
    rank_residual = np.trace(A.T@A)
    # fitting_residual = np.trace((Xend-Mhat).T@(Xend-Mhat))
    if(rank_residual < 1e-5):
        flag = 1
        print('Smoothing seems to have suceeded')
    else:
        flag = 0

    if(final_log['f(x)'] <= 1e-5):
        print("Smoothing Suceeded: Solver found cost f(X,U)<= 1e-5")
    else:
        print("Smoothing FAILED: The variety was not correctly identified. Change mu ? ")

    return (XUopt, flag)


def smoothing_n3(Mhat, d=2, mu=1e-5, x0=None):
    n, s = Mhat.shape
    N = ncr(n + d, d)
    Phi_M = monomial_features_n3_d2(Mhat)
    r = min(la.matrix_rank(Phi_M, 1e-10), N-1)
    manifold = Product((Euclidean(n, s), Grassmann(N, r)))
    @pymanopt.function.Autograd
    def cost_smoothing3(X, U):
        Phi = monomial_features_n3_d2(X)
        A = Phi - U@(U.T@Phi)
        return np.trace(A.T@A) + mu*np.trace((X-Mhat).T@(X-Mhat))
    problem = Problem(manifold=manifold, cost=cost_smoothing3)
    problem.verbosity = 2
    # XU0 = (M, svd_r(Phi_M))
    solver = TrustRegions(maxiter=100, mingradnorm=1e-07, logverbosity=2)
    XUopt, optlog = solver.solve(problem, x0)
    final_log = optlog['final_values']
    Xend = XUopt[0]
    Uend = XUopt[1]
    Phi = monomial_features_n3_d2(Xend)
    A = Phi - Uend@(Uend.T@Phi)
    rank_residual = np.trace(A.T@A)
    # fitting_residual = np.trace((Xend-Mhat).T@(Xend-Mhat))
    if(rank_residual < 1e-5):
        flag = 1
        print('Smoothing seems to have suceeded')
    else:
        flag = 0

    if(final_log['f(x)'] <= 1e-5):
        print("Smoothing Suceeded: Solver found cost f(X,U)<= 1e-5")
    else:
        print("Smoothing FAILED: The variety was not correctly identified. Change mu ? ")

    return (XUopt, flag)


def registration_n3(M1hat, M2hat, d=2, mu=1e-5, x0=None):
    # Finds a rotation Q and translation a such that
    # M2 = Q*M1 + a
    n, s1 = M1hat.shape
    n, s2 = M2hat.shape
    N = ncr(n+d, d)
    r = N-1
    x01 = (M1hat, truncate_svd(monomial_features_n3_d2(M1hat), r))
    x02 = (M2hat, truncate_svd(monomial_features_n3_d2(M2hat), r))
    Xopt1, flag = smoothing_n3(M1hat, d, mu, x01)
    X1 = Xopt1[0]
    U1 = Xopt1[1]
    Xopt2, flag = smoothing_n3(M2hat, d, mu, x02)
    X2 = Xopt2[0]
    U2 = Xopt2[1]
    P_U2perp = np.identity(N) - U2@U2.T

    manifold = Product((SpecialOrthogonalGroup(n), Euclidean(n)))
    @pymanopt.function.Autograd
    def cost_registration3(Q, a):
        A = P_U2perp@monomial_features_n3_d2(Q@X1 + np.array([a, ]*s1).T)
        return np.trace(A.T@A)
    problem = Problem(manifold=manifold, cost=cost_registration3)
    problem.verbosity = 2
    solver = TrustRegions(maxiter=10, mingradnorm=1e-07, logverbosity=2)

    f_best = 1000
    Xbest = None
    Xopt, optlog = solver.solve(problem, x0)
    final_log = optlog['final_values']
    if(final_log['f(x)'] <= 1e-1):
        print("Registration SOLVED: Solver found cost f(Q,a)<= 1e-1 (attempts: 1) ")
        print('f_end = ', final_log['f(x)'])
        Qend = Xopt[0]
        aend = Xopt[1]
        return (Qend, aend, U2, X1, X2)
    if(final_log['f(x)'] <= f_best):
        print('f = ', final_log['f(x)'])
        Xbest = Xopt
        f_best = final_log['f(x)']
    i = 0
    while(i < 2):  # this loop just tries 10 random starting points
        Xopt, optlog = solver.solve(problem, x0)
        final_log = optlog['final_values']
        if(final_log['f(x)'] <= 1e-1):
            print("Registration SOLVED: Solver found cost f(Q,a)<= 1e-1 (attempts: "+str(i) + ")")
            print('f_end = ', final_log['f(x)'])
            Qend = Xopt[0]
            aend = Xopt[1]
            return (Qend, aend, U2, X1, X2)
        else:
            print('registration restart: f = ', final_log['f(x)'])
            if(final_log['f(x)'] <= f_best):
                Xbest = Xopt
                f_best = final_log['f(x)']
            i = i + 1
    print("Registration FAILED: Optimizer did not find solution (attempts: "+str(i) + ")")
    Qend = Xbest[0]
    aend = Xbest[1]
    return (Qend, aend, U2, X1, X2)


def smoothing_n2_d2_increase_mu(Mhat, M, d, mu0, sigma, x0=None):
    tol1 = 1e-7
    error = tol1 + 1
    i = 0
    mu = 1e-6
    gamma = 10
    n_iter = 10
    mu_array = np.zeros([n_iter+1, 1])
    error_rank_array = np.zeros([n_iter+1, 1])
    error_noise_array = np.zeros([n_iter+1, 1])
    # error_feasibility_array = np.zeros([n_iter+1, 1])
    while(error >= tol1 and i <= n_iter):
        # Xopt, optlog = solver.solve(problem, x0)
        XUopt, flag = smoothing_n2_d2_noisy(Mhat, d, mu, sigma, x0=None)
        # x0 = XUopt
        x, u = XUopt
        Phi = monomial_features_n2_d2(x)
        A = Phi - u@(u.T@Phi)
        error_rank = np.trace(A.T@A)
        error_noise = la.norm(x-Mhat)
        error_rank_array[i] = error_rank
        error_noise_array[i] = error_noise
        # error_feasibility_array[i] = infeasibility
        mu_array[i] = mu
        print('Iteration: '+str(i)+', mu = '+str(mu)+', error_rank = '+str(error_rank)+', error_noise = '+str(error_noise)+', infeasibility = '+str(infeasibility))
        error = error_rank + error_noise
        # if(error_feasibility > 1e-6):
        mu = gamma*mu
        i = i + 1
    plt.figure()
    titre = 'Noisy problem: sigma = '+str(sigma)
    plt.title(titre)
    plt.semilogy(error_rank_array[0:i], 'bx-', label='rank error')
    plt.semilogy(error_noise_array[0:i], 'rx-', label='noise error')
    plt.semilogy(error_feasibility_array[0:i], 'gx-', label='infeasibility')
    plt.grid(True)
    plt.legend([r'$|| \Phi(X) - P_{U}\Phi(X)||^2$', r'$||A(X)-\tilde{b}||$', r'$||A(X)-b||$'])
    x = np.arange(0, len(mu_array))
    plt.xticks(x, mu_array)
    # dir = 'Noisy problem: sigma = '+str(sigma)
    plt.savefig('./results/change_mu_smoothing.eps', format='eps')
    plt.show()
    return Xopt


def kernel_smoothing(M, d):
    n, s = M.shape
    monomial_ker = monomial_kernel(M, d, 1)
    r = la.matrix_rank(monomial_ker, 1e-10)
    manifold = Grassmann(s, r)
    @pymanopt.function.Autograd
    def cost2(U):
        K = monomial_kernel(M, d, 1)
        return np.trace(K - U@U.T@K)

    problem = Problem(manifold=manifold, cost=cost2)
    problem.verbosity = 2
    solver = TrustRegions(maxiter=5000, mingradnorm=1e-05, logverbosity=2)
    Xopt, optlog = solver.solve(problem)
    return Xopt[0]
