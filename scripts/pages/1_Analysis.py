"""Analysis board: the page is served by scripts/analysis_server.py (CPU search), embedded here."""

import os

import streamlit as st
import streamlit.components.v1 as components

URL = os.environ.get("CHESSRL_ANALYSIS_URL", "http://localhost:8510")

st.set_page_config(page_title="ChessRL analysis", page_icon="♟", layout="wide")
st.caption(
    f"Board served by `scripts/analysis_server.py` at {URL} (systemd user unit chessrl-analysis)."
)
st.link_button("Open in its own tab", URL)
components.iframe(URL, height=1000, scrolling=True)
