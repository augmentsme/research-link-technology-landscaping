"""
Research Landscape Page
Explore the hierarchical structure of research categories and keywords through interactive treemap visualization.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data

st.set_page_config(
    page_title="Research Landscape",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_treemap_data(categories: List[Dict[str, Any]], 
                        max_research_fields: Optional[int] = None,
                        max_categories_per_field: Optional[int] = None,
                        max_keywords_per_category: Optional[int] = None) -> pd.DataFrame:
    """Create hierarchical data structure for treemap visualization"""
    treemap_data = []
    for_groups = {}
    
    for category in categories:
        field_of_research = category.get('field_of_research')
        
        if field_of_research:
            for_code_value = field_of_research
            for_division_name = field_of_research.replace('_', ' ').title()
        else:
            for_code = category.get('for_code', 'Unknown')
            if isinstance(for_code, str):
                for_code_value = for_code
                for_division_name = f'FOR {for_code}'
            elif isinstance(for_code, dict):
                for_code_value = for_code.get('code', 'Unknown')
                for_division_name = for_code.get('name', f'FOR {for_code_value}')
            else:
                for_code_value = 'Unknown'
                for_division_name = 'Unknown FOR Division'
        
        if for_code_value not in for_groups:
            for_groups[for_code_value] = {
                'name': for_division_name,
                'categories': []
            }
        for_groups[for_code_value]['categories'].append(category)
    
    if max_research_fields is not None:
        for_items = []
        for for_code_value, for_data in for_groups.items():
            total_keywords = sum(len(cat.get('keywords', [])) for cat in for_data['categories'])
            for_items.append((for_code_value, for_data, total_keywords))
        for_items.sort(key=lambda x: x[2], reverse=True)
        for_groups = {item[0]: item[1] for item in for_items[:max_research_fields]}
    
    for for_code_value, for_data in for_groups.items():
        for_name = for_data['name']
        categories_for_this_for = for_data['categories']
        
        if max_categories_per_field is not None:
            category_items = [(cat, len(cat.get('keywords', []))) for cat in categories_for_this_for]
            category_items.sort(key=lambda x: x[1], reverse=True)
            categories_for_this_for = [item[0] for item in category_items[:max_categories_per_field]]
        
        for_keyword_count = sum(
            len(cat.get('keywords', [])[:max_keywords_per_category] if max_keywords_per_category else cat.get('keywords', []))
            for cat in categories_for_this_for
        )
        for_size = max(1, for_keyword_count)
        
        treemap_data.append({
            'id': f"FIELD_{for_code_value}",
            'parent': '',
            'name': for_name,
            'value': for_size,
            'level': 'Research Field',
            'item_type': 'research_field'
        })
        
        for category in categories_for_this_for:
            category_name = category['name']
            keywords = category.get('keywords', [])
            
            if max_keywords_per_category is not None:
                keywords = keywords[:max_keywords_per_category]
            
            category_size = max(1, len(keywords))
            
            treemap_data.append({
                'id': f"FIELD_{for_code_value} >> {category_name}",
                'parent': f"FIELD_{for_code_value}",
                'name': category_name,
                'value': category_size,
                'level': 'Category',
                'item_type': 'category'
            })
            
            for keyword in keywords:
                treemap_data.append({
                    'id': f"FIELD_{for_code_value} >> {category_name} >> KW: {keyword}",
                    'parent': f"FIELD_{for_code_value} >> {category_name}",
                    'name': keyword,
                    'value': 1,
                    'level': 'Keyword',
                    'item_type': 'keyword'
                })
    
    return pd.DataFrame(treemap_data)


def create_research_landscape_treemap(categories: List[Dict[str, Any]],
                                      title: str = "Research Landscape",
                                      height: int = 800,
                                      font_size: int = 18,
                                      max_research_fields: Optional[int] = None,
                                      max_categories_per_field: Optional[int] = None,
                                      max_keywords_per_category: Optional[int] = None) -> Optional[go.Figure]:
    """Create a treemap visualization of the research landscape"""
    if not categories:
        return None
    
    treemap_df = create_treemap_data(
        categories,
        max_research_fields=max_research_fields,
        max_categories_per_field=max_categories_per_field,
        max_keywords_per_category=max_keywords_per_category
    )
    
    if treemap_df.empty:
        return None
    
    color_map = {
        'Research Field': '#1f77b4',
        'Category': '#ff7f0e',
        'Keyword': '#2ca02c',
    }
    
    fig = px.treemap(
        treemap_df,
        ids='id',
        names='name',
        parents='parent',
        values='value',
        title=title,
        color='level',
        color_discrete_map=color_map,
        hover_data=['level', 'value', 'item_type']
    )
    
    fig.update_layout(
        font_size=font_size,
        title_font_size=font_size + 7,
        height=height,
        margin=dict(t=60, l=25, r=25, b=25)
    )
    
    fig.update_traces(
        textinfo="label+value",
        textfont_size=font_size,
        textposition="middle center",
        marker=dict(
            line=dict(width=2, color='white'),
            pad=dict(t=10, l=10, r=10, b=10)
        )
    )
    
    return fig


def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("Treemap Settings")
    
    with st.sidebar.expander("Filtering Options", expanded=True):
        max_research_fields = st.selectbox(
            "Maximum Research Fields",
            options=[None, 5, 10, 15, 20],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of research fields to include",
            key="landscape_max_research_fields"
        )
        
        max_categories_per_field = st.selectbox(
            "Maximum categories per research field",
            options=[None, 3, 5, 10, 15],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of categories to show per research field",
            key="landscape_max_categories_per_field"
        )
        
        max_keywords_per_category = st.selectbox(
            "Maximum keywords per category",
            options=[None, 5, 10, 20, 30],
            index=1,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of keywords to show per category",
            key="landscape_max_keywords_per_category"
        )
    
    with st.sidebar.expander("⚙️ Display Settings", expanded=False):
        treemap_height = st.slider(
            "Visualization height",
            min_value=400,
            max_value=1200,
            value=800,
            step=50,
            key="landscape_treemap_height"
        )
        
        font_size = st.slider(
            "Font size",
            min_value=8,
            max_value=24,
            value=18,
            help="Adjust text size in the treemap boxes",
            key="landscape_font_size"
        )
    
    st.sidebar.markdown("---")
    update_settings = st.sidebar.button("Update Settings", type="primary", use_container_width=True, key="landscape_update_button")
    
    return {
        'max_research_fields': max_research_fields,
        'max_categories_per_field': max_categories_per_field,
        'max_keywords_per_category': max_keywords_per_category,
        'treemap_height': treemap_height,
        'font_size': font_size,
        'update_settings': update_settings
    }


def show_statistics(categories_df: pd.DataFrame):
    """Display statistics about the research landscape"""
    categories_list = categories_df.to_dict('records')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Categories", len(categories_df))
    with col2:
        total_keywords = sum(len(cat.get('keywords', [])) for cat in categories_list)
        st.metric("Total Keywords", total_keywords)
    with col3:
        unique_fields = len(
            set(
                cat.get('field_of_research', cat.get('for_code', 'Unknown'))
                for cat in categories_list
            )
        )
        st.metric("Unique Research Fields", unique_fields)


def create_treemap(categories_df: pd.DataFrame, config: dict):
    """Create and display the treemap visualization"""
    cleaned = categories_df.fillna(0).reset_index(drop=True)
    categories_list = cleaned.to_dict('records')
    
    fig_treemap = create_research_landscape_treemap(
        categories=categories_list,
        title="Research Landscape: Research Fields → Categories → Keywords",
        height=config['treemap_height'],
        font_size=config['font_size'],
        max_research_fields=config['max_research_fields'],
        max_categories_per_field=config['max_categories_per_field'],
        max_keywords_per_category=config['max_keywords_per_category']
    )
    
    if fig_treemap is not None:
        st.plotly_chart(fig_treemap, use_container_width=True)
        show_statistics(categories_df)
        
        with st.expander("Debug: View Underlying Data", expanded=False):
            st.subheader("Categories DataFrame")
            st.write(f"**Shape:** {categories_df.shape}")
            st.dataframe(categories_df, use_container_width=True)
            
            st.subheader("Categories List (Dict Format)")
            st.write(f"**Length:** {len(categories_list)} categories")
    else:
        st.warning("No data available for the selected parameters.")


def main():
    st.header("Research Landscape Treemap")
    st.markdown("Explore the hierarchical structure of research categories and keywords.")
    
    _, _, categories_df = load_data()
    
    if categories_df is None or categories_df.empty:
        st.error("Unable to load categories data. Please check your data files.")
        return
    
    config = render_sidebar()
    
    if "landscape_initialized" not in st.session_state:
        st.session_state.landscape_initialized = True
        should_generate = True
    else:
        should_generate = config['update_settings']
    
    if should_generate:
        with st.spinner("Creating research landscape treemap..."):
            create_treemap(categories_df, config)
    else:
        st.info("Adjust settings in the sidebar and click 'Update Settings' to generate a new visualization.")


if __name__ == "__main__":
    main()
