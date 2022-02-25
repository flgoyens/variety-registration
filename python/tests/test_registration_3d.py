#!/usr/bin/python3
import autograd.numpy as np
import matplotlib.pyplot as plt
# import numpy as np
from numpy import random as rnd
from numpy import linalg as la
# from numpy.linalg import svd
# from mpl_toolkits.mplot3d import Axes3D
from solvers_registration import registration_n3
from registration_tools import generate_transformation


def main():
    n = 3
    s = 200
    s1 = s
    s2 = s

    (Q, aa) = generate_transformation(n, 1)
    print("||a|| = " + str(la.norm(aa)))

# #   A simple straight line
    # x = np.array(rnd.randn(1, s))
#     M1 = np.zeros([n, s])
#     M1[0, :] = x
#     M1[1, :] = 2*x
#     M2 = Q@M1


# A quadratic
    x = np.array(rnd.randn(2, s))
    M2 = np.zeros([n, s])
    M2[0:2, :] = x
    M2[2, :] = M2[0, :]*M2[0, :] + M2[1, :]*M2[1, :]
    M1 = Q@M2 + np.array([aa, ]*s1).T

    Qend, aend, Uend, X1, X2 = registration_n3(M1, M2, 2)
    M4 = Qend@M1 + np.array([aend, ]*s2).T

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(M1[0, :], M1[1, :], M1[2, :], c='r', marker='.', label=r"$M_1$")
    ax.scatter(M2[0, :], M2[1, :], M2[2, :], c='b', marker='.', label=r"$M_2$")
    # plt.axis('equal')
    plt.legend()
    # ax.set_legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(M4[0, :], M4[1, :], M4[2, :], c='g', marker='.', label=r"$Q M_1+a$")
    ax.scatter(M2[0, :], M2[1, :], M2[2, :], c='b', marker='.', label=r"$M_2$")
    # plt.axis('equal')
    plt.legend()
    # ax.set_legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('registration-n3-noiseless_testt.eps', format='eps')
    plt.show(block=True)


if __name__ == '__main__':
    main()
