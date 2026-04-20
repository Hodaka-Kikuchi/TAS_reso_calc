# RL_calc.py
import math
import numpy as np

def RL_calc(lc_param):

    a = lc_param['a']
    b = lc_param['b']
    c = lc_param['c']
    alpha = lc_param['alpha']
    beta = lc_param['beta']
    gamma = lc_param['gamma']

    V = a * b * c * math.sqrt(
        1 - math.cos(math.radians(alpha))**2
        - math.cos(math.radians(beta))**2
        - math.cos(math.radians(gamma))**2
        + 2 * math.cos(math.radians(alpha))
        * math.cos(math.radians(beta))
        * math.cos(math.radians(gamma))
    )

    V0 = math.sqrt(
        1 - math.cos(math.radians(alpha))**2
        - math.cos(math.radians(beta))**2
        - math.cos(math.radians(gamma))**2
        + 2 * math.cos(math.radians(alpha))
        * math.cos(math.radians(beta))
        * math.cos(math.radians(gamma))
    )

    def acosd(x):
        return math.degrees(math.acos(x))

    alpha_star = acosd(
        (math.cos(math.radians(beta)) * math.cos(math.radians(gamma))
         - math.cos(math.radians(alpha))) /
        (math.sin(math.radians(beta)) * math.sin(math.radians(gamma)))
    )

    beta_star = acosd(
        (math.cos(math.radians(alpha)) * math.cos(math.radians(gamma))
         - math.cos(math.radians(beta))) /
        (math.sin(math.radians(alpha)) * math.sin(math.radians(gamma)))
    )

    gamma_star = acosd(
        (math.cos(math.radians(alpha)) * math.cos(math.radians(beta))
         - math.cos(math.radians(gamma))) /
        (math.sin(math.radians(alpha)) * math.sin(math.radians(beta)))
    )

    n_a = 2 * math.pi / V * b * c * math.sin(math.radians(alpha))
    n_b = 2 * math.pi / V * a * c * math.sin(math.radians(beta))
    n_c = 2 * math.pi / V * a * b * math.sin(math.radians(gamma))

    astar = n_a * np.array([1, 0, 0])

    bstar = n_b * np.array([
        math.cos(math.radians(gamma_star)),
        math.sin(math.radians(gamma_star)),
        0
    ])

    cstar = n_c * np.array([
        math.cos(math.radians(beta_star)),
        (math.cos(math.radians(alpha_star))
         - math.cos(math.radians(beta_star)) * math.cos(math.radians(gamma_star)))
        / math.sin(math.radians(gamma_star)),
        V0 / math.sin(math.radians(gamma_star))
    ])

    astar[np.abs(astar) <= 1e-6] = 0
    bstar[np.abs(bstar) <= 1e-6] = 0
    cstar[np.abs(cstar) <= 1e-6] = 0

    return {
        "astar": astar,
        "bstar": bstar,
        "cstar": cstar,
        "alpha_star": alpha_star,
        "beta_star": beta_star,
        "gamma_star": gamma_star,
        "n_a": n_a,
        "n_b": n_b,
        "n_c": n_c,
        "V": V,
        "V0": V0
    }