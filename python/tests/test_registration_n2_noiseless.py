#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from numpy import random as rnd
# from numpy import linalg as la
# from numpy.linalg import svd
from registration_tools import generate_transformation
# from registration_tools import ncr, monomial_features_2dim
from solvers_registration import registration_n2


def main():
    n = 2
    s = 20
    s1 = s
    s2 = 30

# Define transformation : Q and a
    (Q, aa) = generate_transformation(n, 1)

# Data generation
# A quadratic
    M2 = np.zeros([n, s2])
    M2[0, :] = np.array(rnd.randn(1, s2))
    M2[1, :] = M2[0, :]*M2[0, :]

    Mtemp = np.zeros([n, s1])
    Mtemp[0, :] = np.array(rnd.randn(1, s1))
    Mtemp[1, :] = Mtemp[0, :]*Mtemp[0, :]
    M1 = Q@Mtemp + np.array([aa, ]*s1).transpose()

    sigma = 0
    M1hat = M1 + sigma*rnd.randn(n, s1)
    M2hat = M2 + sigma*rnd.randn(n, s2)

    # d = 2
    # mu = 1e-5
    Qend, aend, U1, X1, X2 = registration_n2(M1hat, M2hat)
    M4 = Qend@X1 + np.array([aend, ]*s1).T

    plt.figure(1)
    plt.subplot(211)
    plt.scatter(M1hat[0, :], M1hat[1, :], c='r', marker='.', label=r"$M_1$")
    plt.scatter(M2hat[0, :], M2hat[1, :], c='b', marker='.', label=r"$M_2$")
    plt.axis('equal')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.legend()

    plt.subplot(212)
    plt.scatter(M4[0, :], M4[1, :], c='g', marker='.', label=r"$Q M_1+a$")
    plt.scatter(X2[0, :], X2[1, :], c='b', marker='.', label=r"$M_2$")
    plt.axis('equal')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('registration_n2_noiseless.eps', format='eps')
    plt.show()


if __name__ == '__main__':
    main()
