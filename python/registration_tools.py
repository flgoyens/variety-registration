from functools import reduce
import operator as op
import autograd.numpy as np
from numpy import random as rnd
from numpy import linalg as la
from numpy.linalg import svd


def truncate_svd(K, r):
    u, s, v = svd(K)
    return u[:, 0:r]


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def lifted_dimension(n, d):
    return ncr(n + d, d)


def monomial_kernel(X, d=2, c=1):
    return (X.T@X + c)**d


def monomial_features_n2_d2(X):
    # only n = 2, and d = 2
    # n = xi.shape
    # if(n != 2):
    #     print("ERROR: n must be 2")
    n, s = X.shape
    return np.array([np.ones(s), X[0, :], X[1, :], X[0, :]*X[1, :], X[0, :]*X[0, :], X[1, :]*X[1, :]])


def monomial_features_n3_d2(X):
    # only n = 3, and d = 2
    n, s = X.shape
    return np.array([np.ones(s), X[0, :], X[1, :], X[2, :],
                    X[0, :]*X[0, :], X[0, :]*X[1, :], X[0, :]*X[2, :],
                     X[1, :]*X[2, :], X[1, :]*X[1, :], X[2, :]*X[2, :]])


def monomial_features_n2_d3(X):
    # only n = 2, and d = 3
    n, s = X.shape
    return np.array([np.ones(s), X[0, :], X[1, :],
                    X[0, :]*X[0, :], X[0, :]*X[1, :], X[1, :]*X[1, :],
                    X[0, :]*X[0, :]*X[0, :], X[1, :]*X[1, :]*X[1, :],
                    X[0, :]*X[1, :]*X[1, :], X[0, :]*X[0, :]*X[1, :],
                     ])


def monomial_features_n3_d3(X):
    # only n = 3, and d = 3
    n, s = X.shape
    return np.array([np.ones(s), X[0, :], X[1, :], X[2, :],
                    X[0, :]*X[0, :], X[0, :]*X[1, :], X[0, :]*X[2, :],
                    X[1, :]*X[2, :], X[1, :]*X[1, :], X[2, :]*X[2, :],
                    X[0, :]*X[0, :]*X[0, :], X[1, :]*X[1, :]*X[1, :],
                    X[2, :]*X[2, :]*X[2, :], X[0, :]*X[1, :]*X[2, :],
                    X[0, :]*X[1, :]*X[1, :], X[0, :]*X[0, :]*X[1, :],
                    X[0, :]*X[0, :]*X[2, :], X[0, :]*X[2, :]*X[2, :],
                    X[1, :]*X[1, :]*X[2, :], X[1, :]*X[2, :]*X[2, :],
                     ])


def generate_transformation(n, translation=1):
    Q, RR = la.qr(rnd.randn(n, n))
    Q = np.dot(Q, np.diag(np.sign(np.diag(RR))))
    if la.det(Q) < 0:
        Q[:, [0, 1]] = Q[:, [1, 0]]
    if(translation):
        a = np.array(2*rnd.randn(n))
    else:
        a = np.zeros(n)
    return (Q, a)
