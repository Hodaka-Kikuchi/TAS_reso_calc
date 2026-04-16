import streamlit as st
import numpy as np
import math

# =========================
# 逆格子計算
# =========================
def RL_calc(a, b, c, alpha, beta, gamma):

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


# =========================
# UB計算
# =========================
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


# =========================
# Streamlit UI
# =========================
st.title("TAS UB Matrix Calculator")

st.subheader("Lattice parameters")

a = st.number_input("a (Å)", value=5.0)
b = st.number_input("b (Å)", value=5.0)
c = st.number_input("c (Å)", value=5.0)

alpha = st.number_input("alpha (deg)", value=90.0)
beta  = st.number_input("beta (deg)", value=90.0)
gamma = st.number_input("gamma (deg)", value=90.0)

st.subheader("Reflection vectors (hkl)")

h1 = st.number_input("sv1 h", value=1)
k1 = st.number_input("sv1 k", value=0)
l1 = st.number_input("sv1 l", value=0)

h2 = st.number_input("sv2 h", value=0)
k2 = st.number_input("sv2 k", value=1)
l2 = st.number_input("sv2 l", value=0)

if st.button("Calc"):

    rl = RL_calc(a, b, c, alpha, beta, gamma)

    UB = UB_calc(
        np.array([h1, k1, l1]),
        np.array([h2, k2, l2]),
        rl["astar"], rl["bstar"], rl["cstar"],
        rl["alpha_star"], rl["beta_star"], rl["gamma_star"],
        rl["n_a"], rl["n_b"], rl["n_c"],
        a, b, c, alpha, beta, gamma
    )

    st.subheader("Results")

    st.write("### astar")
    st.write(rl["astar"])

    st.write("### bstar")
    st.write(rl["bstar"])

    st.write("### cstar")
    st.write(rl["cstar"])

    st.write("### U matrix")
    st.write(UB["U"])

    st.write("### B matrix")
    st.write(UB["B"])

    st.write("### UB matrix")
    st.write(UB["UB"])