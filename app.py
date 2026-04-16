import streamlit as st
import math

st.title("Neutron Converter")

mode = st.selectbox(
"Select input",
["Energy (meV)", "Wavelength (Å)", "k (Å⁻¹)", "Velocity (m/s)"]
)

val = st.number_input("Input value", value=5.0)

if st.button("Convert"):


    if mode == "Energy (meV)":
        E = val
        k = math.sqrt(E / 2.072)
        lam = 2 * math.pi / k
        v = 437 * math.sqrt(E)

    elif mode == "Wavelength (Å)":
        lam = val
        k = 2 * math.pi / lam
        E = 2.072 * k * k
        v = 437 * math.sqrt(E)

    elif mode == "k (Å⁻¹)":
        k = val
        lam = 2 * math.pi / k
        E = 2.072 * k * k
        v = 437 * math.sqrt(E)

    elif mode == "Velocity (m/s)":
        v = val
        E = (v / 437) ** 2
        k = math.sqrt(E / 2.072)
        lam = 2 * math.pi / k

    st.write(f"Energy: {E:.3f} meV")
    st.write(f"k: {k:.3f} Å⁻¹")
    st.write(f"Wavelength: {lam:.3f} Å")
    st.write(f"Velocity: {v:.1f} m/s")

# python -m streamlit run app.py # 実行コマンド