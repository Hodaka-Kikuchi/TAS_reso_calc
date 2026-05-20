import streamlit as st
st.set_page_config(layout="wide")
st.title("Development Notes")

lang = st.radio("Language", ["JP", "EN"], horizontal=True)

with open("dev_notes.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

lines = reversed(lines)

st.text("\n".join(lines))