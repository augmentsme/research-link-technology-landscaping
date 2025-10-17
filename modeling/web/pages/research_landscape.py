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

from shared_utils import load_data, load_css, render_page_links

st.set_page_config(
    page_title="Research Landscape",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

render_page_links()


def create_treemap_data(categories: List[Dict[str, Any]], 
                        max_research_fields: Optional[int] = None,
                        max_categories_per_field: Optional[int] = None,
                        max_keywords_per_category: Optional[int] = None) -> pd.DataFrame:
    """Create hierarchical data structure for treemap visualization"""
    treemap_data = []
    
    field_groups = group_categories_by_field(categories)
    
    if max_research_fields is not None:
        field_groups = limit_research_fields(field_groups, max_research_fields)
    
    for field_name, field_data in field_groups.items():
        add_field_level_data(
            treemap_data, field_name, field_data, 
            max_categories_per_field, max_keywords_per_category
        )
    
    return pd.DataFrame(treemap_data)


def group_categories_by_field(categories: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group categories by research field"""
    field_groups = {}
    
    for category in categories:
        field_of_research = category.get('field_of_research', 'Unknown')
        
        if pd.isna(field_of_research) or not field_of_research:
            field_of_research = 'Unknown'
        
        field_display_name = field_of_research.replace('_', ' ').title()
        
        if field_of_research not in field_groups:
            field_groups[field_of_research] = {
                'name': field_display_name,
                'categories': []
            }
        field_groups[field_of_research]['categories'].append(category)
    
    return field_groups


def limit_research_fields(field_groups: Dict[str, Dict[str, Any]], 
                         max_research_fields: int) -> Dict[str, Dict[str, Any]]:
    """Limit research fields by sorting by total keyword count"""
    field_items = []
    for field_name, field_data in field_groups.items():
        total_keywords = sum(len(cat.get('keywords', [])) for cat in field_data['categories'])
        field_items.append((field_name, field_data, total_keywords))
    
    field_items.sort(key=lambda x: x[2], reverse=True)
    return {item[0]: item[1] for item in field_items[:max_research_fields]}


def limit_categories(categories: List[Dict[str, Any]], max_categories: int) -> List[Dict[str, Any]]:
    """Limit categories by keyword count"""
    category_items = []
    for category in categories:
        keyword_count = len(category.get('keywords', []))
        category_items.append((category, keyword_count))
    
    category_items.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in category_items[:max_categories]]


def add_field_level_data(treemap_data: List[Dict[str, Any]], field_name: str, 
                        field_data: Dict[str, Any], max_categories_per_field: Optional[int],
                        max_keywords_per_category: Optional[int]):
    """Add field-level data to treemap"""
    field_display_name = field_data['name']
    
    categories_for_this_field = field_data['categories']
    if max_categories_per_field is not None:
        categories_for_this_field = limit_categories(categories_for_this_field, max_categories_per_field)
    
    field_keyword_count = sum(
        len(cat.get('keywords', [])[:max_keywords_per_category] if max_keywords_per_category else cat.get('keywords', []))
        for cat in categories_for_this_field
    )
    field_size = max(1, field_keyword_count)
    
    treemap_data.append({
        'id': f"FIELD_{field_name}",
        'parent': '',
        'name': field_display_name,
        'value': field_size,
        'level': 'Research Field',
        'item_type': 'research_field'
    })
    
    add_category_and_keyword_data(
        treemap_data, field_name, categories_for_this_field, max_keywords_per_category
    )


def add_category_and_keyword_data(treemap_data: List[Dict[str, Any]], field_name: str,
                                 categories: List[Dict[str, Any]], max_keywords_per_category: Optional[int]):
    """Add category and keyword level data to treemap"""
    for category in categories:
        category_name = category['name']
        keywords = category.get('keywords', [])
        
        if max_keywords_per_category is not None:
            keywords = keywords[:max_keywords_per_category]
        
        category_size = max(1, len(keywords))
        
        treemap_data.append({
            'id': f"FIELD_{field_name} >> {category_name}",
            'parent': f"FIELD_{field_name}",
            'name': category_name,
            'value': category_size,
            'level': 'Category',
            'item_type': 'category'
        })
        
        for keyword in keywords:
            treemap_data.append({
                'id': f"FIELD_{field_name} >> {category_name} >> KW: {keyword}",
                'parent': f"FIELD_{field_name} >> {category_name}",
                'name': keyword,
                'value': 1,
                'level': 'Keyword',
                'item_type': 'keyword'
            })


def create_treemap(categories: List[Dict[str, Any]],
                  title: str = 'Research Landscape: Research Fields → Categories → Keywords',
                  height: int = 800,
                  font_size: int = 18,
                  max_research_fields: Optional[int] = None,
                  max_categories_per_field: Optional[int] = None,
                  max_keywords_per_category: Optional[int] = None) -> Optional[go.Figure]:
    """Create a comprehensive treemap visualization of the research landscape"""
    
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
            line=dict(width=2, color='white')
        )
    )
    
    return fig


def render_sidebar(categories_df: pd.DataFrame):
    """Render sidebar controls"""
    st.sidebar.header("Treemap Settings")
    
    with st.sidebar.expander("Filtering Options", expanded=True):
        max_research_fields = st.selectbox(
            "Maximum Research Fields",
            options=[None, 5, 10, 15, 20],
            index=1,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of research fields to include"
        )
        
        max_categories_per_field = st.selectbox(
            "Maximum categories per research field",
            options=[None, 3, 5, 10, 15],
            index=2,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of categories to show per research field"
        )
        
        max_keywords_per_category = st.selectbox(
            "Maximum keywords per category",
            options=[None, 5, 10, 20, 30],
            index=1,
            format_func=lambda x: "All" if x is None else str(x),
            help="Maximum number of keywords to show per category"
        )
    
    with st.sidebar.expander("⚙️ Display Settings", expanded=False):
        treemap_height = st.slider(
            "Visualization height",
            min_value=400,
            max_value=1200,
            value=800,
            step=50
        )
        
        font_size = st.slider(
            "Font size",
            min_value=8,
            max_value=24,
            value=24,
            help="Adjust text size in the treemap boxes"
        )
    
    return {
        'max_research_fields': max_research_fields,
        'max_categories_per_field': max_categories_per_field,
        'max_keywords_per_category': max_keywords_per_category,
        'treemap_height': treemap_height,
        'font_size': font_size
    }


def show_statistics(categories_list: List[Dict[str, Any]]):
    """Display statistics about the research landscape"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Categories", len(categories_list))
    with col2:
        total_keywords = sum(len(cat.get('keywords', [])) for cat in categories_list)
        st.metric("Total Keywords", total_keywords)
    with col3:
        unique_fields = len(
            set(
                cat.get('field_of_research', 'Unknown')
                for cat in categories_list
            )
        )
        st.metric("Unique Research Fields", unique_fields)


def main():
    st.header("Research Landscape Treemap")
    st.markdown("Explore the hierarchical structure of research categories and keywords.")

    with st.spinner("Loading data..."):
    
        _, _, categories_df = load_data()

        if categories_df is None or categories_df.empty:
            st.error("Unable to load categories data.")
            return

        # Reset index to include 'name' as a column instead of dropping it
        categories_df = categories_df.fillna(0).reset_index()
        categories_list = categories_df.to_dict('records')

        config = render_sidebar(categories_df)
    
    with st.spinner("Creating research landscape treemap..."):
        fig = create_treemap(
            categories=categories_list,
            title="Research Landscape: Research Fields → Categories → Keywords",
            height=config['treemap_height'],
            font_size=config['font_size'],
            max_research_fields=config['max_research_fields'],
            max_categories_per_field=config['max_categories_per_field'],
            max_keywords_per_category=config['max_keywords_per_category']
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            show_statistics(categories_list)
        else:
            st.warning("No data available for the selected parameters.")


if __name__ == "__main__":
    main()
