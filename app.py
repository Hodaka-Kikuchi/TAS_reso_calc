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
from QEresolution_scan_dev import calcresolution_scan3

# localでのDebugの仕方
# cd C:\Users\h34\Documents\Python\TAS_reso_calc_web
# python -m streamlit run app_dev.py

# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="TAS Resolution Calculator", layout="wide")
st.title("TAS Resolution Calculator")


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Calculation Point sticky area
       -------------------------------------------------------- */

    .calculation-sticky {
        position: sticky;
        top: 0rem;
        z-index: 999;
        background-color: var(--background-color);
        padding-top: 0.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(128,128,128,0.35);
    }


    /* Sidebar heading spacing */
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.8rem;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Default values
# ============================================================

with open("default_instr_val.json", "r") as f:
    INSTRUMENTS = json.load(f)

INSTRUMENT_LABELS = {
    "arbitrary": "ARBITRARY",
    "CTAX": "CTAX@HFIR",
    "HB1": "HB1@HFIR",
    "HB1A": "HB1A@HFIR",
    "HB3": "HB3@HFIR",
    "HER": "HER@JRR3"
}

def instrument_changed():

    instrument_display = st.session_state.instrument_select

    label_to_key = {
        v: k for k, v in INSTRUMENT_LABELS.items()
    }

    instrument = label_to_key[instrument_display]

    if instrument == "arbitrary":
        st.session_state.Method_config = "Cooper-Nathans"
    else:
        st.session_state.Method_config = "Popovici"

# ============================================================
# Instrument selection
# ============================================================

with st.sidebar:

    #st.markdown("## Instrument")

    st.markdown(
            "📒 [Development Notes]"
            "(https://tasresocalc-4mzh7b5efdx5qsyzkztcaf.streamlit.app/)"
        )

    # ----------------------------------------------------
    # Approximation / Instrument
    # ----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        # 初回だけ
        if "Method_config" not in st.session_state:
            st.session_state.Method_config = "Cooper-Nathans"

        method = st.radio(
            "Approximation",
            ["Cooper-Nathans", "Popovici"],
            key="Method_config"
        )

    with col2:

        instrument_display = st.selectbox(
            "Instrument",
            list(INSTRUMENT_LABELS.values()),
            key="instrument_select",
            on_change=instrument_changed
        )

label_to_key = {
    v: k for k, v in INSTRUMENT_LABELS.items()
}

instrument = label_to_key[instrument_display]

config = INSTRUMENTS[instrument]

# ============================================================
# Load instrument configuration
# ============================================================

if (
    "instrument_loaded" not in st.session_state
    or st.session_state.instrument_loaded != instrument
):

    config = INSTRUMENTS[instrument]

    st.session_state.instrument_loaded = instrument
    
    # checkbox
    st.session_state.gm_1st = (
        config.get("supermirror", {}).get("enabled")
    )

    st.session_state.fc_mono_h = (
        config.get("monochromator", {}).get("hfocus")
    )

    st.session_state.fc_mono_v = (
        config.get("monochromator", {}).get("vfocus")
    )

    st.session_state.fc_ana_h = (
        config.get("analyzer", {}).get("hfocus")
    )

    st.session_state.fc_ana_v = (
        config.get("analyzer", {}).get("vfocus")
    )
    
    st.session_state.energy_mode = (
        config.get("configuration", {}).get("energy_mode")
    )

    st.session_state.geometry = (
        config.get("configuration", {}).get("geometry")
    )

    st.session_state.sign_config = (
        config.get("configuration", {}).get("sign")
    )

    # crystal
    st.session_state.mono = (
        config.get("monochromator", {}).get("crystal")
    )

    st.session_state.ana = (
        config.get("analyzer", {}).get("crystal")
    )

    # collimator
    st.session_state.div_2nd_h = (
        config.get("collimator", {}).get("2nd_h")
    )

    st.session_state.div_2nd_v = (
        config.get("collimator", {}).get("2nd_v")
    )

    st.session_state.div_3rd_h = (
        config.get("collimator", {}).get("3rd_h")
    )

    st.session_state.div_3rd_v = (
        config.get("collimator", {}).get("3rd_v")
    )

    st.session_state.div_4th_h = (
        config.get("collimator", {}).get("4th_h")
    )

    st.session_state.div_4th_v = (
        config.get("collimator", {}).get("4th_v")
    )

    if st.session_state.gm_1st:

        st.session_state.div_1st_m = (
            config.get("supermirror", {}).get("m_value")
        )

    else:

        st.session_state.div_1st_h = (
            config.get("collimator", {}).get("1st_h")
        )

        st.session_state.div_1st_v = (
            config.get("collimator", {}).get("1st_v")
        )

    st.session_state.Ef = (
        config.get("configuration", {}).get("Ef")
    )

    # focusing blades

    st.session_state.mono_h_blade = (
        config.get("monochromator", {}).get("blade_h")
    )

    st.session_state.mono_v_blade = (
        config.get("monochromator", {}).get("blade_v")
    )

    st.session_state.ana_h_blade = (
        config.get("analyzer", {}).get("blade_h")
    )

    st.session_state.ana_v_blade = (
        config.get("analyzer", {}).get("blade_v")
    )

    # mosaic

    st.session_state.mos_mono_h = (
        config.get("monochromator", {}).get("mosaic_h")
    )

    st.session_state.mos_mono_v = (
        config.get("monochromator", {}).get("mosaic_v")
    )

    st.session_state.mos_ana_h = (
        config.get("analyzer", {}).get("mosaic_h")
    )

    st.session_state.mos_ana_v = (
        config.get("analyzer", {}).get("mosaic_v")
    )

    # distance

    st.session_state.L0 = (
        config.get("distance", {}).get("L0")
    )

    st.session_state.L1 = (
        config.get("distance", {}).get("L1")
    )

    st.session_state.L2 = (
        config.get("distance", {}).get("L2")
    )

    st.session_state.L3 = (
        config.get("distance", {}).get("L3")
    )

    # beam

    st.session_state.beam_width = (
        config.get("beam", {}).get("width")
    )

    st.session_state.beam_height = (
        config.get("beam", {}).get("height")
    )

    # monochromator size

    st.session_state.mono_width = (
        config.get("monochromator", {}).get("width")
    )

    st.session_state.mono_height = (
        config.get("monochromator", {}).get("height")
    )

    st.session_state.mono_thickness = (
        config.get("monochromator", {}).get("thickness")
    )

    # analyzer size

    st.session_state.ana_width = (
        config.get("analyzer", {}).get("width")
    )

    st.session_state.ana_height = (
        config.get("analyzer", {}).get("height")
    )

    st.session_state.ana_thickness = (
        config.get("analyzer", {}).get("thickness")
    )

    # detector size

    st.session_state.det_width = (
        config.get("detector", {}).get("width")
    )

    st.session_state.det_height = (
        config.get("detector", {}).get("height")
    )

    st.session_state.instrument_loaded = instrument


# ============================================================
# LEFT SIDEBAR
# ============================================================

with st.sidebar:

    # ========================================================
    # Lattice information
    # ========================================================

    with st.container(border=True):

        st.markdown("#### Lattice parameter")

        # ------------------------------------------------
        # Row 1 : a, b, c
        # ------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            a = st.number_input(
                "a (Å)",
                value=5.0,
                format="%.4f"
            )

        with col2:
            b = st.number_input(
                "b (Å)",
                value=5.0,
                format="%.4f"
            )

        with col3:
            c = st.number_input(
                "c (Å)",
                value=5.0,
                format="%.4f"
            )

        # ------------------------------------------------
        # Row 2 : alpha, beta, gamma
        # ------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            alpha = st.number_input(
                "alpha (deg)",
                value=90.0
            )

        with col2:
            beta = st.number_input(
                "beta (deg)",
                value=90.0
            )

        with col3:
            gamma = st.number_input(
                "gamma (deg)",
                value=90.0
            )

        # ----------------------------------------------------
        # Scattering plane
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Scattering plane")

            # axis 1
            st.markdown("**axis 1 (in plane)**")

            c1, c2, c3 = st.columns(3)

            with c1:
                h1 = st.number_input(
                    "h1",
                    value=1.0
                )

            with c2:
                k1 = st.number_input(
                    "k1",
                    value=0.0
                )

            with c3:
                l1 = st.number_input(
                    "l1",
                    value=0.0
                )

            # axis 2
            st.markdown("**axis 2 (in plane)**")

            c1, c2, c3 = st.columns(3)

            with c1:
                h2 = st.number_input(
                    "h2",
                    value=0.0
                )

            with c2:
                k2 = st.number_input(
                    "k2",
                    value=1.0
                )

            with c3:
                l2 = st.number_input(
                    "l2",
                    value=0.0
                )

            # axis 3
            st.markdown("**axis 3 (out of plane)**")

            c1, c2, c3 = st.columns(3)

            with c1:
                h3 = st.number_input(
                    "h3",
                    value=0.0
                )

            with c2:
                k3 = st.number_input(
                    "k3",
                    value=0.0
                )

            with c3:
                l3 = st.number_input(
                    "l3",
                    value=1.0
                )

    # ========================================================
    # Lattice parameters
    # ========================================================

    lc_param = {
        "a": a,
        "b": b,
        "c": c,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "sv1": np.array([h1, k1, l1]),
        "sv2": np.array([h2, k2, l2]),
        "sv3": np.array([h3, k3, l3]),
    }

    # ========================================================
    # UB matrix calculation
    # ========================================================

    if st.button("Calc UB matrix"):

        rl = RL_calc(lc_param)
        UB = UB_calc(lc_param,rl)

        with st.expander("Calculation result", expanded=True):

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

            st.markdown("U, B, UB matrix", unsafe_allow_html=True)

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

    # ========================================================
    # Instrument setting
    # ========================================================

    with st.container(border=True):

        st.markdown("### Instrument setting")

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Configuration")

            # =================================================
            # Row 1 : Mode / Energy
            # =================================================

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # Mode
            # ------------------------------------------------

            with col1:

                energy_mode = st.radio(
                    "Mode",
                    ["Ei fixed", "Ef fixed"],
                    key="energy_mode"
                )

            # ------------------------------------------------
            # Energy
            # ------------------------------------------------

            with col2:

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


            # =================================================
            # Row 2 : Geometry / Sign
            # =================================================

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            with col1:

                geometry = st.radio(
                    "Geometry",
                    ["W", "anti-W"],
                    key="geometry"
                )

            # ------------------------------------------------
            # Sign
            # ------------------------------------------------

            with col2:

                sign_config = st.radio(
                    "Sign",
                    ["+-+", "-+-"],
                    key="sign_config"
                )
        
        # ----------------------------------------------------
        # Focusing Conditions
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Focusing Conditions")

            col1, col2 = st.columns(2)

            # ------------------------------------------------
            # Monochromator
            # ------------------------------------------------
            with col1:

                st.markdown("**Monochromator**")

                fc_mono_h = st.checkbox(
                    "Horizontal",
                    key="fc_mono_h"
                )

                mono_h_blade = st.number_input(
                    "Horizontal blade number",
                    min_value=1,
                    key="mono_h_blade"
                )

                fc_mono_v = st.checkbox(
                    "Vertical",
                    value=True,
                    key="fc_mono_v"
                )

                mono_v_blade = st.number_input(
                    "Vertical blade number",
                    min_value=1,
                    key="mono_v_blade"
                )

            # ------------------------------------------------
            # Analyzer
            # ------------------------------------------------
            with col2:

                st.markdown("**Analyzer**")

                fc_ana_h = st.checkbox(
                    "Horizontal",
                    key="fc_ana_h"
                )

                ana_h_blade = st.number_input(
                    "Horizontal blade number",
                    min_value=1,
                    key="ana_h_blade"
                )

                fc_ana_v = st.checkbox(
                    "Vertical",
                    value=True,
                    key="fc_ana_v"
                )

                ana_v_blade = st.number_input(
                    "Vertical blade number",
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

    # ========================================================
    # Collimator conditions
    # ========================================================

    with st.container(border=True):

        st.markdown("### Collimator conditions")
        st.caption("unit: min")

        # ----------------------------------------------------
        # 1st
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("### Collimator conditions")
            st.caption("unit: min")

            # ----------------------------------------------------
            # 1st
            # ----------------------------------------------------

            with st.container(border=True):

                st.markdown("**1st**")

                # =================================================
                # Row 1 : Horizontal / Vertical
                # =================================================

                col1, col2 = st.columns(2)

                # Horizontal
                with col1:

                    if st.session_state.get("gm_1st", True):

                        st.number_input(
                            "Horizontal (disabled)",
                            disabled=True,
                            key="div_1st_h_disabled"
                        )

                        div_1st_h = None

                    else:

                        div_1st_h = st.number_input(
                            "Horizontal",
                            key="div_1st_h"
                        )

                # Vertical
                with col2:

                    if st.session_state.get("gm_1st", True):

                        st.number_input(
                            "Vertical (disabled)",
                            disabled=True,
                            key="div_1st_v_disabled"
                        )

                        div_1st_v = None

                    else:

                        div_1st_v = st.number_input(
                            "Vertical",
                            key="div_1st_v"
                        )


                # =================================================
                # Row 2 : Supermirror / m-value
                # =================================================

                col1, col2 = st.columns(2)

                # Supermirror
                with col1:

                    gm_1st = st.checkbox(
                        "Supermirror",
                        value=True,
                        key="gm_1st"
                    )

                # m-value
                with col2:

                    if gm_1st:

                        div_1st_m = st.number_input(
                            "m-value",
                            key="div_1st_m"
                        )

                    else:

                        st.number_input(
                            "m-value (disabled)",
                            disabled=True,
                            key="div_1st_m_disabled"
                        )

                        div_1st_m = None


            # ----------------------------------------------------
            # 2nd
            # ----------------------------------------------------

            with st.container(border=True):

                st.markdown("**2nd**")

                col1, col2 = st.columns(2)

                with col1:
                    div_2nd_h = st.number_input(
                        "Horizontal",
                        key="div_2nd_h"
                    )

                with col2:
                    div_2nd_v = st.number_input(
                        "Vertical",
                        key="div_2nd_v"
                    )


            # ----------------------------------------------------
            # 3rd
            # ----------------------------------------------------

            with st.container(border=True):

                st.markdown("**3rd**")

                col1, col2 = st.columns(2)

                with col1:
                    div_3rd_h = st.number_input(
                        "Horizontal",
                        key="div_3rd_h"
                    )

                with col2:
                    div_3rd_v = st.number_input(
                        "Vertical",
                        key="div_3rd_v"
                    )


            # ----------------------------------------------------
            # 4th
            # ----------------------------------------------------

            with st.container(border=True):

                st.markdown("**4th**")

                col1, col2 = st.columns(2)

                with col1:
                    div_4th_h = st.number_input(
                        "Horizontal",
                        key="div_4th_h"
                    )

                with col2:
                    div_4th_v = st.number_input(
                        "Vertical",
                        key="div_4th_v"
                    )

    col_param = {
        "gm_1st": gm_1st,
        "div_1st_m": div_1st_m,
        "div_1st_h": div_1st_h,
        "div_1st_v": div_1st_v,
        "div_2nd_h": div_2nd_h,
        "div_2nd_v": div_2nd_v,
        "div_3rd_h": div_3rd_h,
        "div_3rd_v": div_3rd_v,
        "div_4th_h": div_4th_h,
        "div_4th_v": div_4th_v,
    }

    # ========================================================
    # Crystal & Mosaic
    # ========================================================

    d_options = {
        "PG(002)": 3.355,
        "PG(004)": 1.677,
        "Heusler": 3.437,
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

        st.markdown("### Crystal & Mosaic")
        st.caption("unit: min")

        # ----------------------------------------------------
        # Monochromator
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Monochromator")

            # Row 1
            col1, col2 = st.columns(2)

            with col1:

                mono_choice = st.selectbox(
                    "crystal",
                    list(d_options.keys()) + ["Other"],
                    key="mono"
                )

            with col2:

                if mono_choice == "Other":

                    d_mono = st.number_input(
                        "d (Å)",
                        value=3.0,
                        format="%.3f",
                        key="d_mono_manual"
                    )

                else:

                    d_mono = d_options[mono_choice]

                    st.number_input(
                        "d (Å)",
                        value=d_mono,
                        format="%.3f",
                        disabled=True,
                        key="d_mono_auto"
                    )

            # 現在の d を表示
            st.write(f"d = {d_mono:.3f} Å")

            # Row 2
            col1, col2 = st.columns(2)

            with col1:

                mos_mono_h = st.number_input(
                    "horizontal",
                    key="mos_mono_h"
                )

            with col2:

                mos_mono_v = st.number_input(
                    "vertical",
                    key="mos_mono_v"
                )

        # ----------------------------------------------------
        # Sample
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Sample")

            col1, col2 = st.columns(2)

            with col1:

                mos_sam_h = st.number_input(
                    "horizontal",
                    value=60,
                    key="mos_sam_h"
                )

            with col2:

                mos_sam_v = st.number_input(
                    "vertical",
                    value=60,
                    key="mos_sam_v"
                )


        # ----------------------------------------------------
        # Analyzer
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("#### Analyzer")

            # Row 1
            col1, col2 = st.columns(2)

            with col1:

                ana_choice = st.selectbox(
                    "crystal",
                    list(d_options.keys()) + ["Other"],
                    key="ana"
                )

            with col2:

                if ana_choice == "Other":

                    d_ana = st.number_input(
                        "d (Å)",
                        value=3.0,
                        format="%.3f",
                        key="d_ana_manual"
                    )

                else:

                    d_ana = d_options[ana_choice]

                    st.number_input(
                        "d (Å)",
                        value=d_ana,
                        format="%.3f",
                        disabled=True,
                        key="d_ana_auto"
                    )

            # 現在の d を表示
            st.write(f"d = {d_ana:.3f} Å")

            # Row 2
            col1, col2 = st.columns(2)

            with col1:

                mos_ana_h = st.number_input(
                    "horizontal",
                    key="mos_ana_h"
                )

            with col2:

                mos_ana_v = st.number_input(
                    "vertical",
                    key="mos_ana_v"
                )

    mos_param = {
        "d_mono": d_mono,
        "mos_mono_h": mos_mono_h,
        "mos_mono_v": mos_mono_v,
        "mos_sam_h": mos_sam_h,
        "mos_sam_v": mos_sam_v,
        "d_ana": d_ana,
        "mos_ana_h": mos_ana_h,
        "mos_ana_v": mos_ana_v,
    }

    # ========================================================
    # Components size
    # ========================================================

    with st.container(border=True):

        st.markdown("### Components size")
        st.caption("unit: m")

        # ----------------------------------------------------
        # Distance
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("**Distance**")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                L0 = st.number_input(
                    "L0 (source→mono)",
                    step=0.1,
                    format="%.3f",
                    key="L0"
                )

            with c2:
                L1 = st.number_input(
                    "L1 (mono→sample)",
                    step=0.01,
                    format="%.3f",
                    key="L1"
                )

            with c3:
                L2 = st.number_input(
                    "L2 (sample→ana)",
                    step=0.01,
                    format="%.3f",
                    key="L2"
                )

            with c4:
                L3 = st.number_input(
                    "L3 (ana→det)",
                    step=0.01,
                    format="%.3f",
                    key="L3"
                )

        # ----------------------------------------------------
        # Beam
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("**Beam**")

            c1, c2 = st.columns(2)

            with c1:
                beam_width = st.number_input(
                    "B_Width",
                    step=0.001,
                    format="%.3f",
                    key="beam_width"
                )

            with c2:
                beam_height = st.number_input(
                    "B_Height",
                    step=0.001,
                    format="%.3f",
                    key="beam_height"
                )

        # ----------------------------------------------------
        # Monochromator
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("**Monochromator (per piece)**")

            c1, c2, c3 = st.columns(3)

            with c1:
                mono_width = st.number_input(
                    "M_Width",
                    step=0.001,
                    format="%.3f",
                    key="mono_width"
                )

            with c2:
                mono_height = st.number_input(
                    "M_Height",
                    step=0.001,
                    format="%.3f",
                    key="mono_height"
                )

            with c3:
                mono_thickness = st.number_input(
                    "M_Thickness",
                    step=0.001,
                    format="%.3f",
                    key="mono_thickness"
                )

        # ----------------------------------------------------
        # Analyzer
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("**Analyzer (per piece)**")

            c1, c2, c3 = st.columns(3)

            with c1:
                ana_width = st.number_input(
                    "A_Width",
                    step=0.001,
                    format="%.3f",
                    key="ana_width"
                )

            with c2:
                ana_height = st.number_input(
                    "A_Height",
                    step=0.001,
                    format="%.3f",
                    key="ana_height"
                )

            with c3:
                ana_thickness = st.number_input(
                    "A_Thickness",
                    step=0.001,
                    format="%.3f",
                    key="ana_thickness"
                )

        # ----------------------------------------------------
        # Detector
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown("**Detector**")

            c1, c2 = st.columns(2)

            with c1:
                det_width = st.number_input(
                    "D_Width",
                    step=0.001,
                    format="%.3f",
                    key="det_width"
                )

            with c2:
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


# ============================================================
# MAIN AREA
# Calculation Point
# ============================================================

st.markdown(
    '<div class="calculation-sticky">',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.subheader("Calculation Point")

    mode = st.radio(
        "Mode",
        ["single", "scan"],
        horizontal=True,
        key="calc_mode"
    )

    # ========================================================
    # SINGLE MODE
    # ========================================================

    if mode == "single":

        c1, c2, c3, c4, c5 = st.columns(
            [1.2, 1, 1, 1, 1]
        )

        with c1:

            hw = st.number_input(
                "ℏω (meV)",
                value=0.0,
                step=0.001,
                key="single_hw",
                format="%.3f"
            )

        with c2:

            h = st.number_input(
                "H",
                value=0.0,
                step=0.001,
                key="single_h",
                format="%.3f"
            )

        with c3:

            k = st.number_input(
                "K",
                value=0.0,
                step=0.001,
                key="single_k",
                format="%.3f"
            )

        with c4:

            l = st.number_input(
                "L",
                value=0.0,
                step=0.001,
                key="single_l",
                format="%.3f"
            )

        with c5:

            st.write("")
            st.write("")

            calc_single_button = st.button(
                "Calculate",
                key="calc_single_button",
                use_container_width=True
            )

        calc_params = {
            "hw": hw,
            "h": h,
            "k": k,
            "l": l
        }

    # ========================================================
    # SCAN MODE
    # ========================================================

    elif mode == "scan":

        c1, c2, c3, c4, c5 = st.columns(5)

        # ----------------------------------------------------
        # ℏω
        # ----------------------------------------------------

        with c1:

            hw_i = st.number_input(
                "ℏω initial (meV)",
                value=0.00,
                step=0.001,
                key="hw_i",
                format="%.3f"
            )

            hw_f = st.number_input(
                "ℏω final (meV)",
                value=0.00,
                step=0.001,
                key="hw_f",
                format="%.3f"
            )

        # ----------------------------------------------------
        # H
        # ----------------------------------------------------

        with c2:

            h_i = st.number_input(
                "H initial",
                value=0.0,
                step=0.001,
                key="h_i",
                format="%.3f"
            )

            h_f = st.number_input(
                "H final",
                value=0.00,
                step=0.001,
                key="h_f",
                format="%.3f"
            )

        # ----------------------------------------------------
        # K
        # ----------------------------------------------------

        with c3:

            k_i = st.number_input(
                "K initial",
                value=0.0,
                step=0.001,
                key="k_i",
                format="%.3f"
            )

            k_f = st.number_input(
                "K final",
                value=0.0,
                step=0.001,
                key="k_f",
                format="%.3f"
            )

        # ----------------------------------------------------
        # L
        # ----------------------------------------------------

        with c4:

            l_i = st.number_input(
                "L initial",
                value=0.0,
                step=0.001,
                key="l_i",
                format="%.3f"
            )

            l_f = st.number_input(
                "L final",
                value=0.0,
                step=0.001,
                key="l_f",
                format="%.3f"
            )

        # ----------------------------------------------------
        # Scan points / Calculate
        # ----------------------------------------------------

        with c5:

            npts = st.number_input(
                "Scan points",
                min_value=2,
                value=2,
                step=1,
                key="scan_npts"
            )

            st.write("")

            calc_scan_button = st.button(
                "Calculate",
                key="calc_scan_button",
                use_container_width=True
            )


        # ====================================================
        # Calculation parameters
        # ====================================================

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


        # ====================================================
        # Scan arrays
        # ====================================================

        hw_vals = np.linspace(
            hw_i,
            hw_f,
            int(npts)
        )

        h_vals = np.linspace(
            h_i,
            h_f,
            int(npts)
        )

        k_vals = np.linspace(
            k_i,
            k_f,
            int(npts)
        )

        l_vals = np.linspace(
            l_i,
            l_f,
            int(npts)
        )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CALCULATION
# ============================================================

# ============================================================
# Single calculation
# ============================================================

if mode == "single":

    if calc_single_button:

        rl = RL_calc(lc_param)

        RM, fig = calcresolution_scan3(
            lc_param,
            rl,
            col_param,
            mos_param,
            config,
            approximation,
            focusing,
            geom,
            calc_params
        )

        st.session_state.single_result = (
            RM,
            fig
        )


# ============================================================
# Scan calculation
# ============================================================

elif mode == "scan":

    if calc_scan_button:

        rl = RL_calc(lc_param)

        results = []

        for i in range(int(npts)):

            calc_params_i = {
                "hw": hw_vals[i],
                "h": h_vals[i],
                "k": k_vals[i],
                "l": l_vals[i]
            }

            RM, fig = calcresolution_scan3(
                lc_param,
                rl,
                col_param,
                mos_param,
                config,
                approximation,
                focusing,
                geom,
                calc_params_i
            )

            results.append(
                (RM, fig)
            )

        st.session_state.scan_results = results

        # 初回は1番目
        st.session_state.scan_slider = 1


# ============================================================
# RESULT AREA
# ============================================================

# ============================================================
# Single result
# ============================================================

if mode == "single":

    if "single_result" in st.session_state:

        RM, fig = st.session_state.single_result

        st.markdown("## Resolution")

        st.pyplot(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Resolution matrix
        # ----------------------------------------------------

        with st.expander(
            "Resolution Matrix",
            expanded=False
        ):

            st.text(
                np.array2string(
                    RM,
                    precision=6,
                    suppress_small=False
                )
            )


# ============================================================
# Scan result
# ============================================================

elif mode == "scan":

    if "scan_results" in st.session_state:

        results = st.session_state.scan_results

        st.markdown("## Resolution")

        # ----------------------------------------------------
        # Scan navigation
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(
            [2, 1, 1]
        )

        with col2:

            if st.button(
                "◀ Prev",
                use_container_width=True
            ):

                if "scan_slider" not in st.session_state:
                    st.session_state.scan_slider = 1

                st.session_state.scan_slider = max(
                    1,
                    st.session_state.scan_slider - 1
                )

        with col3:

            if st.button(
                "Next ▶",
                use_container_width=True
            ):

                if "scan_slider" not in st.session_state:
                    st.session_state.scan_slider = 1

                st.session_state.scan_slider = min(
                    len(results),
                    st.session_state.scan_slider + 1
                )

        # ----------------------------------------------------
        # Scan slider
        # ----------------------------------------------------

        if "scan_slider" not in st.session_state:
            st.session_state.scan_slider = 1

        i = st.slider(
            "Scan index",
            1,
            len(results),
            st.session_state.scan_slider,
            key="scan_slider"
        )

        RM, fig = results[i - 1]

        # ----------------------------------------------------
        # Current scan point
        # ----------------------------------------------------

        st.caption(
            f"Scan point {i} / {len(results)}"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Resolution matrix
        # ----------------------------------------------------

        with st.expander(
            "Resolution Matrix",
            expanded=False
        ):

            st.text(
                np.array2string(
                    RM,
                    precision=6,
                    suppress_small=False
                )
            )