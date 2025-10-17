"""
Keyword Trends Analysis Page
Analyze how research keywords evolve over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    load_data, load_css, render_page_links,
    get_keyword_grant_links, expand_links_to_years
)

st.set_page_config(
    page_title="Keywords",
    layout="wide",
    initial_sidebar_state="expanded"
)


load_css()

render_page_links()


def create_keyword_years_table(keywords_df: pd.DataFrame, grants_df: pd.DataFrame) -> pd.DataFrame:
    """Create a table with one row per keyword per year based on active grants"""
    keyword_grant_links = get_keyword_grant_links(keywords_df)
    return expand_links_to_years(keyword_grant_links, grants_df, 'keyword', use_active_period=True)


def apply_filters(keywords_df: pd.DataFrame, grants_df: pd.DataFrame, sources, 
                  keyword_types, min_count, max_count, year_min, year_max):
    """Apply filters to keywords and grants"""
    filtered_grants = grants_df.copy()
    
    if sources:
        filtered_grants = filtered_grants[filtered_grants['source'].isin(sources)]
    
    if 'start_year' in filtered_grants.columns:
        start = filtered_grants['start_year']
        end = filtered_grants['end_year']
        start = start.fillna(end)
        end = end.fillna(start)
        mask = pd.Series(True, index=filtered_grants.index)
        if year_min is not None:
            mask &= end >= year_min
        if year_max is not None:
            mask &= start <= year_max
        filtered_grants = filtered_grants[mask]
    
    filtered_keywords = keywords_df.copy()
    
    if keyword_types:
        filtered_keywords = filtered_keywords[filtered_keywords['type'].isin(keyword_types)]
    
    keyword_grant_links = get_keyword_grant_links(filtered_keywords)
    valid_links = keyword_grant_links[keyword_grant_links['grant_id'].isin(filtered_grants.index)]
    
    grant_counts = valid_links.groupby('keyword').size()
    valid_keywords = grant_counts[(grant_counts >= min_count) & (grant_counts <= max_count)].index
    

    filtered_keywords = filtered_keywords[filtered_keywords.index.isin(valid_keywords)]
    
    return filtered_keywords, filtered_grants


def prepare_keyword_trends_data(filtered_keywords: pd.DataFrame, filtered_grants: pd.DataFrame, 
                                year_min, year_max, metric: str) -> pd.DataFrame:
    """Prepare data for keyword trends visualization"""
    keyword_years = create_keyword_years_table(filtered_keywords, filtered_grants)
    
    if keyword_years.empty:
        return pd.DataFrame()
    
    if year_min is not None:
        keyword_years = keyword_years[keyword_years['year'] >= year_min]
    if year_max is not None:
        keyword_years = keyword_years[keyword_years['year'] <= year_max]
    
    if metric == "funding":
        aggregated = keyword_years.groupby(['year', 'keyword'], as_index=False).agg({
            'funding_amount': 'sum'
        }).rename(columns={'funding_amount': 'value'})
    else:
        aggregated = keyword_years.groupby(['year', 'keyword'], as_index=False).agg({
            'grant_id': 'count'
        }).rename(columns={'grant_id': 'value'})
    
    # Fill in missing years with 0 for each keyword to show accurate drops
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        all_keywords = aggregated['keyword'].unique()
        
        # Create a complete grid of year x keyword combinations
        full_index = pd.MultiIndex.from_product(
            [all_years, all_keywords],
            names=['year', 'keyword']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        # Merge with actual data and fill missing values with 0
        aggregated = full_df.merge(aggregated, on=['year', 'keyword'], how='left')
        aggregated['value'] = aggregated['value'].fillna(0)
    
    return aggregated


def get_top_keywords(trends_data: pd.DataFrame, num_keywords: int, metric: str) -> list:
    """Get top N keywords by total value"""
    total_by_keyword = trends_data.groupby('keyword')['value'].sum().sort_values(ascending=False)
    return total_by_keyword.head(num_keywords).index.tolist()


def create_keyword_trends_chart(data: pd.DataFrame, num_keywords: int, metric: str, title: str) -> go.Figure:
    """Create a line chart for keyword trends"""
    if data.empty:
        return None
    
    top_keywords = get_top_keywords(data, num_keywords, metric)
    data_plot = data[data['keyword'].isin(top_keywords)].copy()
    
    fig = go.Figure()
    
    for keyword in top_keywords:
        keyword_data = data_plot[data_plot['keyword'] == keyword].sort_values('year')
        fig.add_trace(go.Scatter(
            x=keyword_data['year'],
            y=keyword_data['value'],
            name=keyword,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    y_label = 'Funding Amount' if metric == "funding" else 'Grant Count'
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_label,
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def render_sidebar(keywords_df: pd.DataFrame, grants_df: pd.DataFrame):
    """Render sidebar controls"""
    st.sidebar.header("Filters")
    
    sources = grants_df['source'].dropna().unique().tolist() if 'source' in grants_df.columns else []
    source_filter = st.sidebar.multiselect("Source", sorted(sources), default=["arc.gov.au"] if "arc.gov.au" in sources else [])
    
    keyword_types = keywords_df['type'].dropna().unique().tolist() if 'type' in keywords_df.columns else []
    keyword_type_filter = st.sidebar.multiselect("Keyword Type", sorted(keyword_types))
    
    st.sidebar.subheader("Grant Count Range")
    min_count = st.sidebar.number_input("Minimum Grants", min_value=1, value=5, step=1)
    max_count = st.sidebar.number_input("Maximum Grants", min_value=1, value=1000, step=10)
    
    min_year = int(grants_df['start_year'].min()) if 'start_year' in grants_df.columns else 2000
    max_year = int(grants_df['start_year'].max()) if 'start_year' in grants_df.columns else 2024
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    st.sidebar.header("Display Options")
    
    metric = st.sidebar.radio("Metric", ["count", "funding"], index=0)
    num_keywords = st.sidebar.slider("Number of Top Keywords", 5, 30, 10)
    
    return {
        'source_filter': source_filter,
        'keyword_type_filter': keyword_type_filter,
        'min_count': min_count,
        'max_count': max_count,
        'year_min': year_range[0],
        'year_max': year_range[1],
        'metric': metric,
        'num_keywords': num_keywords
    }


def show_statistics(total_keywords: int, total_grants: int, filtered_keywords: pd.DataFrame, 
                   filtered_grants: pd.DataFrame, displayed_keywords: int):
    """Display statistics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Keywords", total_keywords)
    with col2:
        st.metric("Filtered Keywords", len(filtered_keywords))
    with col3:
        st.metric("Total Grants", total_grants)
    with col4:
        st.metric("Filtered Grants", len(filtered_grants))
    with col5:
        st.metric("Displayed Keywords", displayed_keywords)


def main():
    st.header("Keywords Analysis")

    with st.spinner("Loading data..."):
    
        keywords_df, grants_df, _ = load_data()

        if keywords_df is None or keywords_df.empty or grants_df is None or grants_df.empty:
            st.error("Unable to load data.")
            return

        config = render_sidebar(keywords_df, grants_df)
    
    with st.spinner("Creating visualization..."):
        filtered_keywords, filtered_grants = apply_filters(
            keywords_df,
            grants_df,
            config['source_filter'],
            config['keyword_type_filter'],
            config['min_count'],
            config['max_count'],
            config['year_min'],
            config['year_max']
        )
        
        if filtered_keywords.empty:
            st.error("No keywords found with current filters.")
            return
        
        trends_data = prepare_keyword_trends_data(
            filtered_keywords,
            filtered_grants,
            config['year_min'],
            config['year_max'],
            config['metric']
        )
        
        if trends_data.empty:
            st.warning("No keyword activity in selected period.")
            return
        
        title = f"Keyword Trends Over Time (Active Grants Only)"
        
        fig = create_keyword_trends_chart(
            trends_data,
            config['num_keywords'],
            config['metric'],
            title
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            show_statistics(
                len(keywords_df), 
                len(grants_df), 
                filtered_keywords, 
                filtered_grants,
                min(config['num_keywords'], len(filtered_keywords))
            )
        else:
            st.warning("No data available for visualization.")


if __name__ == "__main__":
    main()
