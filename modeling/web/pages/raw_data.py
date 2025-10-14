import streamlit as st
import sys
from pathlib import Path

from shared_utils import (  # noqa: E402
    load_data
)
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)


st.set_page_config(
    page_title="Grant",
    layout="wide",
    initial_sidebar_state="expanded"
)

keywords, grants, categories = load_data()

st.title("Raw Data")
st.markdown("### Keywords")
st.dataframe(keywords)
st.markdown("### Grants")
st.dataframe(grants)
st.markdown("### Categories")
st.dataframe(categories)