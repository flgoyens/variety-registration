#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
import math
from numpy import random as rnd
from numpy import linalg as la
# from numpy.linalg import svd
import registration_tools
import solvers_registration


def main():
    n = 2
    s = 150

    # Generate data (circle)
    theta = np.linspace(0, 2*np.pi, s)
    x = np.cos(theta)
    y = np.sin(theta)
    sigma = 0.2

    ynoise = y + sigma*rnd.randn(s,)
    xnoise = x + sigma*rnd.randn(s,)
    Mhat = np.array([xnoise, ynoise])
    M = np.array([x, y])

    # Method parameters
    d = 2
    mu = 1e-5
    N = registration_tools.lifted_dimension(n, d)
    r = N-1
    x0 = (Mhat, registration_tools.truncate_svd(registration_tools.monomial_features_n2_d2(Mhat), r))

    # Run algorithm
    (XUopt, flag) = solvers_registration.smoothing_n2(Mhat, d, mu, x0)
    Xopt = XUopt[0]

    # Display results
    print('RMSE = ', la.norm(M-Xopt)/math.sqrt(n*s))

    plt.figure(1)
    ax = plt.gca()  # or any other way to get an axis object
    ax.plot(M[0, :], M[1, :], c='k', label=r'$x^2 + y^2 = 1$')
    ax.scatter(Mhat[0, :], Mhat[1, :], c='r', marker='.', label=r'$\hat M$')
    ax.scatter(Xopt[0, :], Xopt[1, :], c='b', marker='.', label=r'$X^*$')
    ax.axis('equal')
    # ax.plot(x, y, label=r'$\sin (x)$')
    ax.legend()
    # plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('circle-python_sigma02-again.eps', format='eps')
    plt.show(block=True)
    print('done')


if __name__ == '__main__':
    main()
