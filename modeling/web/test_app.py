import streamlit as st
import sys
import os
from pathlib import Path

st.title("Test App")

# Add the parent directory to the Python path to import from the modeling package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Change working directory to parent to ensure relative paths work
os.chdir(parent_dir)

st.write("Python path updated")

try:
    import config
    st.write("Config imported successfully")
    
    st.write("Loading data...")
    keywords = config.Keywords.load()
    st.write(f"Keywords loaded: {keywords.shape}")
    
    grants = config.Grants.load()
    st.write(f"Grants loaded: {grants.shape}")
    
    # Test FOR codes loading
    import json
    with open('data/for_codes_cleaned.json', 'r') as f:
        for_codes_data = json.load(f)
    st.write(f"FOR codes loaded: {len(for_codes_data)} divisions")
    
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.text(traceback.format_exc())
