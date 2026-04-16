import streamlit as st
import numpy as np
import math
import pandas as pd

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
col1, col2, col3 = st.columns(3)
with col1:
    a = st.number_input("a (Å)", value=5.0)
    alpha = st.number_input("alpha (deg)", value=90.0)
with col2:
    b = st.number_input("b (Å)", value=5.0)
    beta = st.number_input("beta (deg)", value=90.0)
with col3:
    c = st.number_input("c (Å)", value=5.0)
    gamma = st.number_input("gamma (deg)", value=90.0)

st.subheader("Reflection vectors (hkl)")

st.subheader("sv1 (hkl)")
col1, col2, col3 = st.columns(3)
with col1:
    h1 = st.number_input("h1", value=1)
with col2:
    k1 = st.number_input("k1", value=0)
with col3:
    l1 = st.number_input("l1", value=0)

st.subheader("sv2 (hkl)")
col1, col2, col3 = st.columns(3)
with col1:
    h2 = st.number_input("h2", value=0)
with col2:
    k2 = st.number_input("k2", value=1)
with col3:
    l2 = st.number_input("l2", value=0)

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

    st.subheader("Reciprocal lattice vectors")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        a = rl["astar"]
        st.markdown("### astar")
        st.write(f"({a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f})")
    with col2:
        b = rl["bstar"]
        st.markdown("### bstar")
        st.write(f"({b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f})")
    with col3:
        c = rl["cstar"]
        st.markdown("### cstar")
        st.write(f"({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})")

    st.subheader("Matrices (U, B, UB)")
    df_U = pd.DataFrame(UB["U"])
    df_B = pd.DataFrame(UB["B"])
    df_UB = pd.DataFrame(UB["UB"])
    st.dataframe(df_U, header=False, index=False)
    st.dataframe(df_B, header=False, index=False)
    st.dataframe(df_UB, header=False, index=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### U matrix")
        df_U = pd.DataFrame(UB["U"], columns=["x", "y", "z"])
        st.dataframe(df_U)

    with col2:
        st.markdown("### B matrix")
        df_B = pd.DataFrame(UB["B"], columns=["x", "y", "z"])
        st.dataframe(df_B)

    with col3:
        st.markdown("### UB matrix")
        df_UB = pd.DataFrame(UB["UB"], columns=["x", "y", "z"])
        st.dataframe(df_UB)