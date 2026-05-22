import streamlit as st
import numpy as np
import math
import pandas as pd
import json

# 逆格子計算
from RL_calc import RL_calc

# UB計算
from UB_calc import UB_calc

# single QE positionでの計算
from QEresolution_scan_dev import calcresolution_scan3 # スライダー形式、Qz方向にも拡張
from QEresolution_scan_dev import make_resolution_fig

# 使用方法
# powershellで　cd C:\Users\h34\Documents\Python\TAS_reso_calc_web
# 続けて　streamlit run app_dev.py

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="TAS Resolution Calculator", layout="wide")
st.title("TAS Resolution Calculator [debug mode]")
st.warning("Development version")

#################################################################################

# development note & default value

with open("default_instr_val.json", "r") as f:
    INSTRUMENTS = json.load(f)

INSTRUMENT_LABELS = {
    "arbitrary": "ARBITRARY",
    "CTAX": "CTAX@HFIR",
    "HER": "HER@JRR3"
}

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        "📒 [Development Notes](https://tasresocalc-4mzh7b5efdx5qsyzkztcaf.streamlit.app/)"
    )

with col2:
    instrument_display = st.selectbox(
    "Instrument",
    list(INSTRUMENT_LABELS.values()),
    key="instrument_select"
)

label_to_key = {v: k for k, v in INSTRUMENT_LABELS.items()}
instrument = label_to_key[instrument_display]

config = INSTRUMENTS[instrument]

# 初回 or instrument変更時だけ反映
if (
    "instrument_loaded" not in st.session_state
    or st.session_state.instrument_loaded != instrument
):
    config = INSTRUMENTS[instrument]

    st.session_state.instrument_loaded = instrument

    # checkbox（bool）
    st.session_state.gm_1st = config.get("supermirror", {}).get("enabled")
    st.session_state.fc_mono_h = config.get("monochromator", {}).get("hfocus")
    st.session_state.fc_mono_v = config.get("monochromator", {}).get("vfocus")
    st.session_state.fc_ana_h = config.get("analyzer", {}).get("hfocus")
    st.session_state.fc_ana_v = config.get("analyzer", {}).get("vfocus")

    # radio（文字列）
    st.session_state.Method_config = config.get("approximation", {}).get("method")

    st.session_state.energy_mode = config.get("configuration", {}).get("energy_mode")
    st.session_state.geometry = config.get("configuration", {}).get("geometry")
    st.session_state.sign_config = config.get("configuration", {}).get("sign")

    # combobox
    st.session_state.mono = config.get("monochromator", {}).get("crystal")
    st.session_state.ana = config.get("analyzer", {}).get("crystal")

    # number_input（float / int）
    st.session_state.div_2nd_h = config.get("collimator", {}).get("2nd_h")
    st.session_state.div_2nd_v = config.get("collimator", {}).get("2nd_v")
    st.session_state.div_3rd_h = config.get("collimator", {}).get("3rd_h")
    st.session_state.div_3rd_v = config.get("collimator", {}).get("3rd_v")
    st.session_state.div_4th_h = config.get("collimator", {}).get("4th_h")
    st.session_state.div_4th_v = config.get("collimator", {}).get("4th_v")
    if st.session_state.gm_1st:
        st.session_state.div_1st_m = config.get("supermirror", {}).get("m_value")
    else:
        st.session_state.div_1st_h = config.get("collimator", {}).get("1st_h")
        st.session_state.div_1st_v = config.get("collimator", {}).get("1st_v")

    st.session_state.Ef = config.get("configuration", {}).get("Ef")

    st.session_state.mono_h_blade = config.get("monochromator", {}).get("blade_h")
    st.session_state.mono_v_blade = config.get("monochromator", {}).get("blade_v")
    st.session_state.ana_h_blade = config.get("analyzer", {}).get("blade_h")
    st.session_state.ana_v_blade = config.get("analyzer", {}).get("blade_v")

    st.session_state.mos_mono_h = config.get("monochromator", {}).get("mosaic_h")
    st.session_state.mos_mono_v = config.get("monochromator", {}).get("mosaic_v")
    st.session_state.mos_ana_h = config.get("analyzer", {}).get("mosaic_h")
    st.session_state.mos_ana_v = config.get("analyzer", {}).get("mosaic_v")

    st.session_state.L0 = config.get("distance", {}).get("L0")
    st.session_state.L1 = config.get("distance", {}).get("L1")
    st.session_state.L2 = config.get("distance", {}).get("L2")
    st.session_state.L3 = config.get("distance", {}).get("L3")

    st.session_state.beam_width = config.get("beam", {}).get("width")
    st.session_state.beam_height = config.get("beam", {}).get("height")

    st.session_state.mono_width = config.get("monochromator", {}).get("width")
    st.session_state.mono_height = config.get("monochromator", {}).get("height")
    st.session_state.mono_thickness = config.get("monochromator", {}).get("thickness")

    st.session_state.ana_width = config.get("analyzer", {}).get("width")
    st.session_state.ana_height = config.get("analyzer", {}).get("height")
    st.session_state.ana_thickness = config.get("analyzer", {}).get("thickness")

    st.session_state.det_width = config.get("detector", {}).get("width")
    st.session_state.det_height = config.get("detector", {}).get("height")

    # 記録
    st.session_state.instrument_loaded = instrument

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

    if st.button("Calc UB matrix"):

        rl = RL_calc(lc_param)
        UB = UB_calc(lc_param,rl)

        st.markdown("Reciprocal lattice vectors", unsafe_allow_html=True)
        
        def safe_matrix(M):
            return np.array(M, dtype=float)

        col1, col2, col3 = st.columns(3)

        with col1:
            a = rl["astar"]
            st.markdown("a*", unsafe_allow_html=True)
            st.write(f"({a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f})")
        with col2:
            b = rl["bstar"]
            st.markdown("b*", unsafe_allow_html=True)
            st.write(f"({b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f})")
        with col3:
            c = rl["cstar"]
            st.markdown("c*", unsafe_allow_html=True)
            st.write(f"({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})")

        st.markdown("U,B,UB matrix", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("U matrix", unsafe_allow_html=True)
            df_U = pd.DataFrame(UB["U"], columns=["x", "y", "z"])
            st.dataframe(df_U)

        with col2:
            st.markdown("B matrix", unsafe_allow_html=True)
            df_B = pd.DataFrame(UB["B"], columns=["x", "y", "z"])
            st.dataframe(df_B)

        with col3:
            st.markdown("UB matrix", unsafe_allow_html=True)
            df_UB = pd.DataFrame(UB["UB"], columns=["x", "y", "z"])
            st.dataframe(df_UB)

with st.container(border=True):
    st.subheader("Collimator conditions (unit:min)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("### 1st")
            gm_1st = st.checkbox("supermirror", value=True, key="gm_1st")

            if gm_1st:
                div_1st_m = st.number_input(
                    "m-value",
                    key="div_1st_m"
                )

                st.number_input("horizontal (disabled)", disabled=True, key="div_1st_h_disabled")
                st.number_input("vertical (disabled)", disabled=True, key="div_1st_v_disabled")

                div_1st_h = None
                div_1st_v = None

            else:
                div_1st_h = st.number_input("horizontal", key="div_1st_h")
                div_1st_v = st.number_input("vertical", key="div_1st_v")

                st.number_input("m-value (disabled)", disabled=True, key="div_1st_m_disabled")

                div_1st_m = None

    with col2:
        with st.container(border=True):
            st.markdown("### 2nd")
            div_2nd_h = st.number_input("horizontal", key="div_2nd_h")
            div_2nd_v = st.number_input("vertical", key="div_2nd_v")

    with col3:
        with st.container(border=True):
            st.markdown("### 3rd")
            div_3rd_h = st.number_input("horizontal", key="div_3rd_h")
            div_3rd_v = st.number_input("vertical", key="div_3rd_v")

    with col4:
        with st.container(border=True):
            st.markdown("### 4th")
            div_4th_h = st.number_input("horizontal", key="div_4th_h")
            div_4th_v = st.number_input("vertical", key="div_4th_v")

    col_param = {
        'gm_1st':gm_1st,
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
                mos_mono_h= st.number_input("horizontal", key="mos_mono_h")
                mos_mono_v = st.number_input("vertical", key="mos_mono_v")

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
                mos_ana_h = st.number_input("horizontal", key="mos_ana_h")
                mos_ana_v = st.number_input("vertical", key="mos_ana_v")

    mos_param = {
        "d_mono":d_mono,
        "mos_mono_h":mos_mono_h,
        "mos_mono_v":mos_mono_v,
        "mos_sam_h":mos_sam_h,
        "mos_sam_v":mos_sam_v,
        "d_ana":d_ana,
        "mos_ana_h":mos_ana_h,
        "mos_ana_v":mos_ana_v,
    }

with st.container(border=True):
    st.subheader("Instrument setting")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        with st.container(border=True):
            st.markdown("<h5>Configuration</h5>", unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1])

            # ===== Left column =====
            with c1:

                energy_mode = st.radio(
                    "Mode",
                    ["Ei fixed", "Ef fixed"],
                    key="energy_mode"
                )

                if energy_mode == "Ei fixed":
                    Ei = st.number_input(
                        "Ei (meV)",
                        step=0.001,
                        format="%.3f",
                        key="Ei"
                    )
                    Ef = None

                else:
                    Ef = st.number_input(
                        "Ef (meV)",
                        step=0.1,
                        format="%.3f",
                        key="Ef"
                    )
                    Ei = None

            # ===== Right column =====
            with c2:

                geometry = st.radio(
                    "Geometry",
                    ["W", "anti-W"],
                    key="geometry"
                )

                sign_config = st.radio(
                    "Sign",
                    ["+-+", "-+-"],
                    key="sign_config"
                )
    with col2:
        with st.container(border=True):
            st.markdown("<h5>Approximation</h5>", unsafe_allow_html=True)

            method = st.radio(
                "Method",
                ["Cooper-Nathans", "Popovici"],
                key="Method_config"
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
                    fc_mono_h = st.checkbox("Horizontal", key="fc_mono_h")
                with c2:
                    mono_h_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        key="mono_h_blade"
                    )

                c3, c4 = st.columns(2)

                with c3:
                    fc_mono_v = st.checkbox("Vertical", value=True, key="fc_mono_v")
                with c4:
                    mono_v_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        key="mono_v_blade"
                    )

            # ===== Analyzer =====
            with col2:
                st.markdown("**Analyzer**")

                c1, c2 = st.columns(2)

                with c1:
                    fc_ana_h = st.checkbox("Horizontal", key="fc_ana_h")
                with c2:
                    ana_h_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        key="ana_h_blade"
                    )

                c3, c4 = st.columns(2)

                with c3:
                    fc_ana_v = st.checkbox("Vertical", value=True, key="fc_ana_v")
                with c4:
                    ana_v_blade = st.number_input(
                        "Blade number",
                        min_value=1,
                        key="ana_v_blade"
                    )

    config = {
        "energy_mode": energy_mode,
        "Ei": Ei,
        "Ef": Ef,
        "geometry": geometry,
        "sign_config": sign_config
    }
    approximation = {
        "method": method
    }
    focusing = {
        "monochromator": {
            "horizontal": {
                "enabled": fc_mono_h,
                "blades": mono_h_blade
            },
            "vertical": {
                "enabled": fc_mono_v,
                "blades": mono_v_blade
            }
        },
        "analyzer": {
            "horizontal": {
                "enabled": fc_ana_h,
                "blades": ana_h_blade
            },
            "vertical": {
                "enabled": fc_ana_v,
                "blades": ana_v_blade
            }
        }
    }
    # ===== 下：コンポーネントサイズ入力 =====

    with st.container(border=True):
        st.markdown("<h5>Components size (unit:m)</h5>", unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        # ===== Distances =====
        with col1:
            with st.container(border=True):
                st.markdown("**Distance**")

                L0 = st.number_input("L0 (source→mono)", step=0.1, format="%.3f", key="L0")
                L1 = st.number_input("L1 (mono→sample)", step=0.01, format="%.3f", key="L1")
                L2 = st.number_input("L2 (sample→ana)", step=0.01, format="%.3f", key="L2")
                L3 = st.number_input("L3 (ana→det)", step=0.01, format="%.3f", key="L3")

        # ===== Beam =====
        with col2:
            with st.container(border=True):
                st.markdown("**Beam**")

                beam_width = st.number_input(
                    "B_Width",
                    step=0.001,
                    format="%.3f",
                    key="beam_width"
                )

                beam_height = st.number_input(
                    "B_Height",
                    step=0.001,
                    format="%.3f",
                    key="beam_height"
                )

        # ===== Monochromator =====
        with col3:
            with st.container(border=True):
                st.markdown("**Monochromator(per piece)**")

                mono_width = st.number_input(
                    "M_Width",
                    step=0.001,
                    format="%.3f",
                    key="mono_width"
                )

                mono_height = st.number_input(
                    "M_Height",
                    step=0.001,
                    format="%.3f",
                    key="mono_height"
                )

                mono_thickness = st.number_input(
                    "M_Thickness",
                    step=0.001,
                    format="%.3f",
                    key="mono_thickness"
                )

        # ===== Analyzer =====
        with col4:
            with st.container(border=True):
                st.markdown("**Analyzer(per piece)**")

                ana_width = st.number_input(
                    "A_Width",
                    step=0.001,
                    format="%.3f",
                    key="ana_width"
                )

                ana_height = st.number_input(
                    "A_Height",
                    step=0.001,
                    format="%.3f",
                    key="ana_height"
                )

                ana_thickness = st.number_input(
                    "A_Thickness",
                    step=0.001,
                    format="%.3f",
                    key="ana_thickness"
                )

        # ===== Detector =====
        with col5:
            with st.container(border=True):
                st.markdown("**Detector**")

                det_width = st.number_input(
                    "D_Width",
                    step=0.001,
                    format="%.3f",
                    key="det_width"
                )

                det_height = st.number_input(
                    "D_Height",
                    step=0.001,
                    format="%.3f",
                    key="det_height"
                )

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

        calc_params = {
                "hw": hw,
                "h": h,
                "k": k,
                "l": l
            }

        if st.button("Calc single"):
            rl = RL_calc(lc_param)
            #mat1,mat2 = calcresolution_scan3(lc_param,rl,col_param,mos_param,config,approximation,focusing,geom,calc_params)
            #st.text(np.array2string(mat1, precision=18, suppress_small=False))
            #st.text(np.array2string(mat2, precision=18, suppress_small=False))
            #st.text(np.array2string(mat3, precision=18, suppress_small=False))
            RM, fig = calcresolution_scan3(lc_param,rl,col_param,mos_param,config,approximation,focusing,geom,calc_params)
            #st.write("A_sets", A_sets)
            #st.write("QE_sets", QE_sets)
            st.pyplot(fig)
            st.write("RM matrix:")
            st.text(np.array2string(RM, precision=6, suppress_small=False))
    
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
            h_f = st.number_input("H final", value=0.00, step=0.001, key="h_f",format="%.3f")

        with c3:
            k_i = st.number_input("K initial", value=0.0, step=0.001, key="k_i",format="%.3f")
            k_f = st.number_input("K final", value=0.0, step=0.001, key="k_f",format="%.3f")

        with c4:
            l_i = st.number_input("L initial", value=0.0, step=0.001, key="l_i",format="%.3f")
            l_f = st.number_input("L final", value=0.0, step=0.001, key="l_f",format="%.3f")

        with c5:
            npts = st.number_input("Scan points",min_value=2, value=2, step=1, key="scan_npts")

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

        # scan arrays
        hw_vals = np.linspace(hw_i, hw_f, int(npts))
        h_vals  = np.linspace(h_i,  h_f,  int(npts))
        k_vals  = np.linspace(k_i,  k_f,  int(npts))
        l_vals  = np.linspace(l_i,  l_f,  int(npts))

        if "scan_results" not in st.session_state:
            st.session_state.scan_results = None

        if "scan_slider" not in st.session_state:
            st.session_state.scan_slider = 1

        # ------------------------------------------------------------
        # Calc scanボタン
        # ------------------------------------------------------------
        if st.button("Calc scan"):

            rl = RL_calc(lc_param)

            results = []

            for i in range(int(npts)):

                calc_params_i = {
                    "hw": hw_vals[i],
                    "h": h_vals[i],
                    "k": k_vals[i],
                    "l": l_vals[i]
                }

                # ★重要変更：RMだけ受け取る（figは捨てる）
                RM, _ = calcresolution_scan3(
                    lc_param, rl, col_param, mos_param,
                    config, approximation, focusing, geom,
                    calc_params_i
                )

                results.append(RM)

            st.session_state.scan_results = results
            st.session_state.scan_slider = 1  # 初期位置リセット

        # ------------------------------------------------------------
        # 表示部分
        # ------------------------------------------------------------
        if st.session_state.scan_results is not None:

            results = st.session_state.scan_results

            col1, col2, col3 = st.columns([4, 1, 1])

            # Prev
            with col2:
                if st.button("◀ Prev"):
                    st.session_state.scan_slider = max(
                        1, st.session_state.scan_slider - 1
                    )

            # Next
            with col3:
                if st.button("Next ▶"):
                    st.session_state.scan_slider = min(
                        len(results), st.session_state.scan_slider + 1
                    )

            # slider（state直結）
            i = st.slider(
                "Scan index",
                1,
                len(results),
                st.session_state.scan_slider,
                key="scan_slider"
            )

            RM = results[i - 1]

            # --------------------------------------------------------
            # ★ここが重要：毎回軽いplot関数で描画する
            # --------------------------------------------------------
            fig = make_resolution_fig(RM, i)  # ←後述関数

            st.pyplot(fig)

            st.write("RM matrix:")
            st.text(np.array2string(RM, precision=6, suppress_small=False))

##################################################################################



