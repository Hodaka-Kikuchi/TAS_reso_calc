import streamlit as st

st.title("Development Notes")

# =========================
# Language switch
# =========================
lang = st.radio("Language", ["JP", "EN"], horizontal=True, index=0)

def T(ja, en):
    return ja if lang == "JP" else en


# =========================
# Load notes
# =========================
with open("dev_notes.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# latest first (reverse order)
lines = list(reversed(lines))


# =========================
# Display
# =========================
st.subheader(T("開発ログ", "Development Log"))

for line in lines:
    st.write(line)


# =========================
# optional info
# =========================
st.markdown("---")

st.caption(
    T(
        "最新の更新が上に表示されます",
        "Latest updates are shown first"
    )
)