"""Solar heat exchange problem.

Based on the problem described in [1]_.

References
----------
.. [1] C. Sharpe, T. Wiest, P. Wang, and C. C. Seepersad, “A Comparative Evaluation of Supervised Machine Learning Classification Techniques for Engineering Design Applications,” Journal of Mechanical Design, vol. 141, no. 12, Oct. 2019, doi: 10.1115/1.4044524.
"""
import numpy as np


def solar_hex(X):
    m_dot = 0.2
    q = 1000.0
    friction = 0.4  # Changed from 0.04
    rho_water = 997.1
    cp_water = 4200.0
    clearance = 0.02
    g = 9.81

    d_p, n_p = X[:, 0], X[:, 1]  # pipe diameter and number of pipes

    s = 8 * friction / (np.pi**2 * g) * (m_dot / rho_water) ** 2 / d_p**5  # Checked

    h_enc = d_p + clearance  # Checked
    w_enc = d_p * n_p + d_p * (n_p - 1) + clearance  # Checked

    l_s = (
        3 * m_dot * cp_water / q
        - (
            np.pi * d_p * (3 * d_p / 2 + clearance / 2)
            + (n_p - 1) * np.pi**2 / 4 * ((3 * d_p / 2) ** 2 - (d_p / 2) ** 2)
        )
    ) / (
        np.pi * d_p / 2 * n_p
    )  # Changed

    l_eff = l_s * n_p + (3 * d_p + clearance) + 50 * d_p * (n_p - 1)  # Checked

    h_l = s * l_eff  # Checked
    l_enc = l_s + 3 * d_p + clearance  # Checked
    volume = l_enc * w_enc * h_enc  # Checked

    return np.column_stack([h_l, volume, l_enc])
