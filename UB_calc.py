import math
import numpy as np

def UB_calc(sv1, sv2, astar, bstar, cstar,
            alpha_star, beta_star, gamma_star,
            n_a, n_b, n_c, a, b, c, alpha, beta, gamma):

    u1 = sv1[0]*astar + sv1[1]*bstar + sv1[2]*cstar
    U1 = u1 / np.linalg.norm(u1)

    u2 = sv2[0]*astar + sv2[1]*bstar + sv2[2]*cstar
    uu2 = u2 - np.dot(U1, u2) * U1
    U2 = uu2 / np.linalg.norm(uu2)

    U3 = np.cross(U1, U2)
    U = np.vstack([U1, U2, U3])

    B = 1/(2*math.pi) * np.array([
        [n_a, n_b*math.cos(math.radians(gamma_star)), n_c*math.cos(math.radians(beta_star))],
        [0, n_b*math.sin(math.radians(gamma_star)), -n_c*math.sin(math.radians(beta_star))*math.cos(math.radians(alpha))],
        [0, 0, 2*math.pi/c]
    ])

    B[np.abs(B) <= 1e-6] = 0

    UB = U @ B
    UB[np.abs(UB) <= 1e-6] = 0

    return {
        "U": U,
        "B": B,
        "UB": UB
    }
