import streamlit as st
import numpy as np
import math
import pandas as pd

# 逆格子計算
from RL_calc import RL_calc

# UB計算
from UB_calc import UB_calc



#################################################################################

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="TAS Resolution Calculator", layout="wide")
st.title("TAS Resolution Calculator")

with st.container(border=True):
    st.markdown("<h5>Lattice information</h5>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<h5>Lattice paramter</h5>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6  = st.columns(6)
        with col1:
            a = st.number_input("a (Å)", value=5.0, format="%.4f")
        with col2:
            b = st.number_input("b (Å)", value=5.0, format="%.4f")
        with col3:
            c = st.number_input("c (Å)", value=5.0, format="%.4f")
        with col4:
            alpha = st.number_input("alpha (deg)", value=90.0)
        with col5:
            beta = st.number_input("beta (deg)", value=90.0)
        with col6:
            gamma = st.number_input("gamma (deg)", value=90.0)

    with st.container(border=True):
        st.markdown("<h5>scattering plane</h5>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container(border=True):
                st.markdown("<h5>axis 1 (in plane)</h5>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    h1 = st.number_input("h1", value=1.0)
                with c2:
                    k1 = st.number_input("k1", value=0.0)
                with c3:
                    l1 = st.number_input("l1", value=0.0)

        with col2:
            with st.container(border=True):
                st.markdown("<h5>axis 2 (in plane)</h5>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    h2 = st.number_input("h2", value=0.0)
                with c2:
                    k2 = st.number_input("k2", value=1.0)
                with c3:
                    l2 = st.number_input("l2", value=0.0)
        
        with col3:
            with st.container(border=True):
                st.markdown("<h5>axis 3 (out of plane)</h5>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    h3 = st.number_input("h3", value=0.0)
                with c2:
                    k3 = st.number_input("k3", value=0.0)
                with c3:
                    l3 = st.number_input("l3", value=1.0)

    lc_param = {
        "a":a,
        "b":b,
        "c":c,
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma,
        "sv1":np.array([h1, k1, l1]),
        "sv2":np.array([h2, k2, l2]),
        "sv3":np.array([h3, k3, l3]),
    }

    if st.button("Calc"):

        rl = RL_calc(lc_param)
        UB = UB_calc(lc_param,rl)

        st.subheader("Reciprocal lattice vectors")
        
        def safe_matrix(M):
            return np.array(M, dtype=float)

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

with st.container(border=True):
    st.subheader("Collimator conditions (unit:min)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("### 1st")
            gm_1st = st.checkbox("supermirror", key="gm_1st")

            if gm_1st:
                div_1st_m = st.number_input(
                    "m-value",
                    value=1.2,
                    key="div_1st_m"
                )

                st.number_input("horizontal (disabled)", value=80, disabled=True, key="div_1st_h_disabled")
                st.number_input("vertical (disabled)", value=240, disabled=True, key="div_1st_v_disabled")

                div_1st_h = None
                div_1st_v = None

            else:
                div_1st_h = st.number_input("horizontal", value=80, key="div_1st_h")
                div_1st_v = st.number_input("vertical", value=240, key="div_1st_v")

                st.number_input("m-value (disabled)", value=1.2, disabled=True, key="div_1st_m_disabled")

                div_1st_m = None

    with col2:
        with st.container(border=True):
            st.markdown("### 2nd")
            div_2nd_h = st.number_input("horizontal", value=80, key="div_2nd_h")
            div_2nd_v = st.number_input("vertical", value=240, key="div_2nd_v")

    with col3:
        with st.container(border=True):
            st.markdown("### 3rd")
            div_3rd_h = st.number_input("horizontal", value=80, key="div_3rd_h")
            div_3rd_v = st.number_input("vertical", value=240, key="div_3rd_v")

    with col4:
        with st.container(border=True):
            st.markdown("### 4th")
            div_4th_h = st.number_input("horizontal", value=80, key="div_4th_h")
            div_4th_v = st.number_input("vertical", value=240, key="div_4th_v")

    col_param = {
        "div_1st_m":div_1st_m,
        "div_1st_h":div_1st_h,
        "div_1st_v":div_1st_v,
        "div_2nd_h":div_2nd_h,
        "div_2nd_v":div_2nd_v,
        "div_3rd_h":div_3rd_h,
        "div_3rd_v":div_3rd_v,
        "div_4th_h":div_4th_h,
        "div_4th_v":div_4th_v,
    }

d_options = {
    "PG(002)": 3.355,
    "PG(004)": 1.677,
    "Heusler": 3.362,
    "CoFe": 1.771,
    "Ge(111)": 3.266,
    "Ge(311)": 1.714,
    "Ge(511)": 1.089,
    "Ge(533)": 0.863,
    "Si(111)": 3.135,
    "Cu(111)": 2.087,
    "Cu(002)": 1.807,
    "Cu(220)": 1.278,
}

with st.container(border=True):
    st.subheader("Crystal & Mosaic conditions (unit:min)")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        with st.container(border=True):
            st.markdown("<h5>monochromator</h5>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                mono_choice = st.selectbox(
                    "crystal",
                    list(d_options.keys()) + ["Other"],
                    key="mono"
                )

                if mono_choice == "Other":
                    d_mono = st.number_input("d (Å)", value=3.0, format="%.3f", key="d_mono_manual")
                else:
                    d_mono = d_options[mono_choice]
                    st.number_input(
                        "d (Å)",
                        value=d_mono,
                        format="%.3f",
                        disabled=True,
                        key="d_mono_auto"
                    )

                st.write("d =", d_mono)
            with c2:
                mos_mono_h= st.number_input("horizontal", value=30, key="mos_mono_h")
                mos_mono_v = st.number_input("vertical", value=30, key="mos_mono_v")

    with col2:
        with st.container(border=True):
            st.markdown("<h5>sample</h5>", unsafe_allow_html=True)
            mos_sam_h = st.number_input("horizontal", value=60, key="mos_sam_h")
            mos_sam_v = st.number_input("vertical", value=60, key="mos_sam_v")

    with col3:
        with st.container(border=True):
            st.markdown("<h5>analyzer</h5>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                ana_choice = st.selectbox(
                    "crystal",
                    list(d_options.keys()) + ["Other"],
                    key="ana"
                )

                if ana_choice == "Other":
                    d_ana = st.number_input("d (Å)", value=3.0, format="%.3f", key="d_ana_manual")
                else:
                    d_ana = d_options[ana_choice]
                    st.number_input(
                        "d (Å)",
                        value=d_ana,
                        format="%.3f",
                        disabled=True,
                        key="d_ana_auto"
                    )

                st.write("d =", d_ana)

            with c2:
                mos_ana_h = st.number_input("horizontal", value=60, key="mos_ana_h")
                mos_ana_v = st.number_input("vertical", value=60, key="mos_ana_v")

    mos_param = {
        "d_mono":d_mono,
        "mos_mono_h":mos_mono_h,
        "mos_mono_v":mos_mono_v,
        "mos_sam_h":mos_sam_h,
        "mos_sam_v":mos_sam_v,
        "d_ana":d_ana,
        "mos_ana_h":mos_ana_h,
        "mos_ana_h":mos_ana_h,
    }

with st.container(border=True):
    st.subheader("Instrument setting")

    col1, col2, col3 = st.columns([3, 2, 4])
    with col1:
        with st.container(border=True):
            st.markdown("<h5>Configuration</h5>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                energy_mode = st.radio(
                    "Mode",
                    ["Ei fixed", "Ef fixed"]
                )

            with c2:
                geometry = st.radio(
                    "Geometry",
                    ["W", "anti-W"]
                )

            with c3:
                sign_config = st.radio(
                    "Sign",
                    ["+-+", "-+-"]
                )
    with col2:
        with st.container(border=True):
            st.markdown("<h5>Approximation</h5>", unsafe_allow_html=True)

            method = st.radio(
                "Method",
                ["Cooper-Nathans", "Popovici"]
            )
    with col3:
        with st.container(border=True):
            st.markdown("<h5>Focusing Conditions</h5>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            # ===== Monochromator =====
            with col1:
                st.markdown("**Monochromator**")

                c1, c2 = st.columns(2)

                with c1:
                    mono_h = st.checkbox("Horizontal", key="mono_h")
                with c2:
                    mono_h_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        value=10,
                        disabled=not mono_h,
                        key="mono_h_blade"
                    )

                c3, c4 = st.columns(2)

                with c3:
                    mono_v = st.checkbox("Vertical", key="mono_v")
                with c4:
                    mono_v_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        value=10,
                        disabled=not mono_v,
                        key="mono_v_blade"
                    )

            # ===== Analyzer =====
            with col2:
                st.markdown("**Analyzer**")

                c1, c2 = st.columns(2)

                with c1:
                    ana_h = st.checkbox("Horizontal", key="ana_h")
                with c2:
                    ana_h_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        value=10,
                        disabled=not ana_h,
                        key="ana_h_blade"
                    )

                c3, c4 = st.columns(2)

                with c3:
                    ana_v = st.checkbox("Vertical", key="ana_v")
                with c4:
                    ana_v_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        value=10,
                        disabled=not ana_v,
                        key="ana_v_blade"
                    )
    # ===== 下：コンポーネントサイズ入力 =====
    with st.container(border=True):
        st.markdown("<h5>Components size</h5>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        # ===== Distances =====
        with col1:
            with st.container(border=True):
                st.markdown("**Distance**")

                L0 = st.number_input("L0 (source→mono)", value=10.0, step=0.001,format="%.3f")
                L1 = st.number_input("L1 (mono→sample)", value=1.60, step=0.001,format="%.3f")
                L2 = st.number_input("L2 (sample→ana)", value=1.02, step=0.001,format="%.3f")
                L3 = st.number_input("L3 (ana→det)", value=0.49, step=0.001,format="%.3f")

        # ===== Beam =====
        with col2:
            with st.container(border=True):
                st.markdown("**Beam**")

                beam_width = st.number_input("B_Width", value=0.140, step=0.001,format="%.3f")
                beam_height = st.number_input("B_Height", value=0.200, step=0.001,format="%.3f")

        # ===== Monochromator =====
        with col3:
            with st.container(border=True):
                st.markdown("**Monochromator(per piece)**")

                mono_width = st.number_input("M_Width", value=0.020, step=0.001,format="%.3f")
                mono_height = st.number_input("M_Height", value=0.020, step=0.001,format="%.3f")
                mono_thickness = st.number_input("M_Thickness", value=0.002, step=0.001,format="%.3f")

        # ===== Analyzer =====
        with col4:
            with st.container(border=True):
                st.markdown("**Analyzer(per piece)**")

                ana_width = st.number_input("A_Width", value=0.020, step=0.001,format="%.3f")
                ana_height = st.number_input("A_Height", value=0.020, step=0.001,format="%.3f")
                ana_thickness = st.number_input("A_Thickness", value=0.002, step=0.001,format="%.3f")

        # ===== Detector =====
        with col5:
            with st.container(border=True):
                st.markdown("**Detector**")

                det_width = st.number_input("D_Width", value=0.032, step=0.001,format="%.3f")
                det_height = st.number_input("D_Height", value=0.120, step=0.001,format="%.3f")

    geom = {
        "L0": L0,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "beam_width": beam_width,
        "beam_height": beam_height,
        "mono_width": mono_width,
        "mono_height": mono_height,
        "mono_thickness": mono_thickness,
        "ana_width": ana_width,
        "ana_height": ana_height,
        "ana_thickness": ana_thickness,
        "det_width": det_width,
        "det_height": det_height,
    }

with st.container(border=True):
    st.subheader("Calculation")

    mode = st.radio(
        "Mode",
        ["single", "scan"],
        horizontal=True,
        key="calc_mode"
    )

    # ======================
    # single mode
    # ======================
    if mode == "single":

        st.markdown("### Single calculation")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            hw = st.number_input("hw (meV)", value=0.0, step=0.001, key="single_hw",format="%.3f")

        with c2:
            h = st.number_input("H", value=0.0, step=0.001, key="single_h",format="%.3f")

        with c3:
            k = st.number_input("K", value=0.0, step=0.001, key="single_k",format="%.3f")

        with c4:
            l = st.number_input("L", value=0.0, step=0.001, key="single_l",format="%.3f")

        if st.button("Calculate single"):
            st.write("single mode selected")
            st.write(hw, h, k, l)

        calc_params = {
                "mode": "single",
                "hw": hw,
                "h": h,
                "k": k,
                "l": l
            }
    
    # ======================
    # scan mode
    # ======================
    elif mode == "scan":

        st.markdown("### Scan calculation")

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            hw_i = st.number_input("hw initial (meV)", value=0.00, step=0.001, key="hw_i",format="%.3f")
            hw_f = st.number_input("hw final (meV)", value=0.00, step=0.001, key="hw_f",format="%.3f")

        with c2:
            h_i = st.number_input("H initial", value=0.0, step=0.001, key="h_i",format="%.3f")
            h_f = st.number_input("H final", value=0.001, step=0.001, key="h_f",format="%.3f")

        with c3:
            k_i = st.number_input("K initial", value=0.0, step=0.001, key="k_i",format="%.3f")
            k_f = st.number_input("K final", value=0.0, step=0.001, key="k_f",format="%.3f")

        with c4:
            l_i = st.number_input("L initial", value=0.0, step=0.001, key="l_i",format="%.3f")
            l_f = st.number_input("L final", value=0.0, step=0.001, key="l_f",format="%.3f")

        with c5:
            npts = st.number_input("Number of scan points", value=11, step=1, key="scan_npts")

        calc_params = {
            "mode": "scan",
            "hw": {
                "initial": hw_i,
                "final": hw_f
            },
            "h": {
                "initial": h_i,
                "final": h_f
            },
            "k": {
                "initial": k_i,
                "final": k_f
            },
            "l": {
                "initial": l_i,
                "final": l_f
            },
            "npts": npts
        }

        if st.button("calc scan"):
            st.write("scan mode selected")
            st.write(hw_i,hw_f,h_i, h_f, k_i, k_f, l_i, l_f, npts)

##################################################################################

if st.button("Calc"):

    rl = RL_calc(lc_param)
    UB = UB_calc(lc_param,rl)

    st.subheader("Reciprocal lattice vectors")
    
    def safe_matrix(M):
        return np.array(M, dtype=float)

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
