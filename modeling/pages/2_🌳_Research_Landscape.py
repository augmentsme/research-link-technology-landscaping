import streamlit as st
from webutils import load_data
from visualisation import create_research_landscape_treemap

st.set_page_config(page_title="Research Landscape", page_icon="🌳")

st.markdown("# 🌳 Research Landscape")
st.sidebar.header("Research Landscape")
st.markdown("Explore the hierarchical structure of research categories and keywords.")

# Load data
keywords, grants, categories = load_data()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar controls for treemap
with st.sidebar:
    st.subheader("Treemap Settings")
    
    max_for_classes = st.selectbox(
        "Maximum FOR classes",
        options=[None, 5, 10, 15, 20],
        index=2,
        format_func=lambda x: "All" if x is None else str(x),
        help="Maximum number of FOR (Field of Research) classes to include"
    )
    
    max_categories_per_for = st.selectbox(
        "Maximum categories per FOR class",
        options=[None, 3, 5, 10, 15],
        index=2,
        format_func=lambda x: "All" if x is None else str(x),
        help="Maximum number of categories to show per FOR class"
    )
    
    max_keywords_per_category = st.selectbox(
        "Maximum keywords per category",
        options=[None, 5, 10, 20, 30],
        index=2,
        format_func=lambda x: "All" if x is None else str(x),
        help="Maximum number of keywords to show per category"
    )
    
    treemap_height = st.slider(
        "Visualization height",
        min_value=400,
        max_value=1200,
        value=800,
        step=50
    )
    
    font_size = st.slider(
        "Font size",
        min_value=6,
        max_value=16,
        value=9
    )

# Generate treemap visualization
if st.button("Generate Research Landscape", type="primary"):
    with st.spinner("Creating research landscape treemap..."):
        categories_list = categories.to_dict('records')
        
        fig_treemap = create_research_landscape_treemap(
            categories=categories_list,
            classification_results=[],
            title="Research Landscape: FOR Classes → Categories → Keywords",
            height=treemap_height,
            font_size=font_size,
            max_for_classes=max_for_classes,
            max_categories_per_for=max_categories_per_for,
            max_keywords_per_category=max_keywords_per_category
        )
        
        if fig_treemap is not None:
            st.plotly_chart(fig_treemap, use_container_width=True)
        else:
            st.error("Failed to create research landscape treemap. Please check your data and try again.")
