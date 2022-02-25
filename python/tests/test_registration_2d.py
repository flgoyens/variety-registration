#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from numpy import random as rnd
from numpy import linalg as la
# from numpy.linalg import svd
from registration_tools import generate_transformation
# from registration_tools import ncr, monomial_features_2dim
from solvers_registration import noiseless_registration


def main():
    n = 2
    s = 20
    s1 = s
    s2 = s + 10

# Define transformation : Q and a
    (Q, aa) = generate_transformation(n, 1)

# Data generation
# #   A simple straight line
    # x = np.array(rnd.randn(1, s))
    # M1 = np.zeros([n, s])
    # M1[0, :] = x
    # M1[1, :] = 2*x
    # M2 = Q@M1

# A quadratic
    M1 = np.zeros([n, s1])
    M1[0, :] = np.array(rnd.randn(1, s1))
    M1[1, :] = M1[0, :]*M1[0, :]

    Mtemp = np.zeros([n, s2])
    Mtemp[0, :] = np.array(rnd.randn(1, s2))
    Mtemp[1, :] = Mtemp[0, :]*Mtemp[0, :]
    M2 = Q@Mtemp + np.array([aa, ]*s2).transpose()

    Qend, aend, Uend = noiseless_registration(M1, M2, 2)
    M4 = Qend@M1 + np.array([aend, ]*s1).T

    plt.figure(1)
    plt.subplot(211)
    plt.scatter(M1[0, :], M1[1, :], c='r', marker='.', label="M1")
    plt.scatter(M2[0, :], M2[1, :], c='b', marker='.', label="M2")
    # plt.title('Initial plot')
    plt.axis('scaled')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.subplot(212)
    plt.scatter(M2[0, :], M2[1, :], c='b', marker='.', label="M2")
    plt.scatter(M4[0, :], M4[1, :], c='g', marker='.', label="QM1+a")
    # plt.title('M2 and Q*M1 + a')
    plt.axis('scaled')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    main()
