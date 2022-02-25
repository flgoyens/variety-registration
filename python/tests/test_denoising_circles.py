#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
from numpy import random as rnd
import registration_tools
import solvers_registration


def main():
    n = 2
    s = 60

    # circle
    theta = np.linspace(0, 2*np.pi, s)
    x = np.cos(theta)
    y = np.sin(theta)
    sigma = 0.01

    M = np.array([x, y])
    d = 2
    mu = 1e-5
    N = registration_tools.lifted_dimension(n, d)
    r = N-1
    plt.figure(1)
    ax = plt.gca()


    ynoise = y + sigma*rnd.randn(s,)
    xnoise = x + sigma*rnd.randn(s,)
    Mhat = np.array([xnoise, ynoise])

    x0 = (Mhat, registration_tools.truncate_svd(registration_tools.monomial_features_n2_d2(Mhat), r))
    (XUopt, flag) = solvers_registration.smoothing_n2(Mhat, d, mu, x0)
    Xopt = XUopt[0]

    ax.plot(M[0, :], M[1, :], c='k', label=r'$x^2 + y^2 = 1$')
    ax.scatter(Mhat[0, :], Mhat[1, :], c='r', marker='.', label=r'$\hat M$')
    ax.scatter(Xopt[0, :], Xopt[1, :], c='b', marker='.', label=r'$X^*$')
    ax.axis('equal')
    ax.legend()
    plt.savefig('circle_denoising_sigma001.eps', format='eps')
    plt.show(block=True)
    print('done')


if __name__ == '__main__':
    main()
