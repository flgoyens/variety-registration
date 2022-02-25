#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from numpy import random as rnd
from numpy import linalg as la
# from numpy.linalg import svd
from registration_tools import generate_transformation
# from registration_tools import ncr, monomial_features_2dim
from solvers_registration import registration_n2

def main():
    n = 2
    s = 50
    s1 = s
    s2 = s

# Define transformation : Q and a
    (Q, aa) = generate_transformation(n, 1)

# Data generation
# A quadratic
    M2 = np.zeros([n, s2])
    M2[0, :] = 2.5*np.abs(np.array(rnd.rand(1, s2)))
    M2[1, :] = M2[0, :]*M2[0, :]

    Mtemp = np.zeros([n, s1])
    Mtemp[0, :] = -2.5*np.abs(np.array(rnd.rand(1, s1)))
    Mtemp[1, :] = Mtemp[0, :]*Mtemp[0, :]
    M1 = Q@Mtemp + np.array([aa, ]*s1).transpose()

    sigma = 1e-2
    M1hat = M1 + sigma*rnd.randn(n, s1)
    M2hat = M2 + sigma*rnd.randn(n, s2)

    # d = 2
    # mu = 1e-5
    Qend, aend, U1, X1, X2 = registration_n2(M1hat, M2hat)
    M4 = Qend@X1 + np.array([aend, ]*s1).T

    plt.figure(1)
    plt.subplot(211)
    plt.scatter(M1hat[0, :], M1hat[1, :], c='r', marker='.', label=r"$\hat{M}_1$")
    plt.scatter(M2hat[0, :], M2hat[1, :], c='b', marker='.', label=r"$\hat{M}_2$")
    plt.axis('equal')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend()

    plt.subplot(212)
    plt.scatter(M4[0, :], M4[1, :], c='g', marker='.', label=r"$Q X_1+a$")
    plt.scatter(X2[0, :], X2[1, :], c='b', marker='.', label=r"$X_2$")
    plt.axis('equal')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('registration-no-overlap-sigma2.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    main()
