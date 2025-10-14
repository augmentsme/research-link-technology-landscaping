import streamlit as st
import sys
from pathlib import Path

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data, render_page_links  # noqa: E402

st.set_page_config(
    page_title="Grant",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_page_links()

keywords, grants, categories = load_data()

st.title("Raw Data")
st.markdown("### Keywords")
st.dataframe(keywords)
st.markdown("### Grants")
st.dataframe(grants)
st.markdown("### Categories")
st.dataframe(categories)