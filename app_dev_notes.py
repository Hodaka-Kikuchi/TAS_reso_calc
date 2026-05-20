import streamlit as st

st.title("Development Notes")

with open("dev_notes.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 逆順（最新が上）
lines = lines[::-1]

st.text(lines)