"""Welded beam problem.


References
----------
.. [1] C. Sharpe, T. Wiest, P. Wang, and C. C. Seepersad, “A Comparative Evaluation of Supervised Machine
        Learning Classification Techniques for Engineering Design Applications,” Journal of Mechanical Design,
        vol. 141, no. 12, Oct. 2019, doi: 10.1115/1.4044524.
"""
import numpy as np


def welded_beam_cont(x):
    f = 1.10471 * ((x[:, 0] ** 2) * x[:, 1]) + 0.04811 * (
        x[:, 2] * x[:, 3] * (14 + x[:, 1])
    )
    return f


def welded_beam_constraints_cont(x):
    p = 6000
    L = 14
    E = 30e6
    G = 12e6
    taumax = 13600
    sigmamax = 30e3
    deltamax = 0.25
    tau1 = p / (np.sqrt(2) * x[:, 0] * x[:, 1])
    m = p * (L + (x[:, 1] / 2))
    r = np.sqrt((x[:, 1] ** 2) / 4 + ((x[:, 0] + x[:, 2]) / 2) ** 2)
    j = 2 * (
        np.sqrt(2)
        * x[:, 0]
        * x[:, 1]
        * ((x[:, 1] ** 2) / 12 + ((x[:, 0] + x[:, 2]) / 2) ** 2)
    )
    sigma = 6 * p * L / (x[:, 3] * (x[:, 2] ** 2))
    delta = 4 * p * L**3 / (E * (x[:, 2] ** 3) * x[:, 3])
    pc = (4.013 * E * np.sqrt((x[:, 2] ** 2) * (x[:, 3] ** 6) / 36) / L**2) * (
        1 - (x[:, 2] / (2 * L)) * np.sqrt(E / (4 * G))
    )
    tau11 = m * r / j
    tau = np.sqrt(tau1**2 + 2 * tau1 * tau11 * x[:, 1] / (2 * r) + tau11**2)
    g1 = tau - taumax
    g2 = sigma - sigmamax
    g3 = x[:, 0] - x[:, 3]
    g4 = 0.10471 * x[:, 0] ** 2 + 0.04811 * x[:, 2] * x[:, 3] * (14 + x[:, 1]) - 5
    g5 = 0.125 - x[:, 0]
    g6 = delta - deltamax
    g7 = p - pc
    #    pdb.set_trace()
    gval = np.transpose(np.array([g1, g2, g3, g4, g5, g6, g7]))
    gbool = np.array(gval < 0)
    return gval, gbool


def welded_beam(x):
    f = welded_beam_cont(x)
    gval, gbool = welded_beam_constraints_cont(x)
    return f, gval, gbool
