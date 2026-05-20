import streamlit as st

st.title("Development Notes")

with open("dev_notes.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

lines = reversed(lines)

st.text("\n".join(lines))