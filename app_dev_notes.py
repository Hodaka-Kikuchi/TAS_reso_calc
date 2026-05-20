import streamlit as st

st.title("Development Notes")

with open("dev_notes.txt", "r", encoding="utf-8") as f:
    notes = f.read()

st.text(notes)