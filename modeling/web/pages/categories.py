"""
Category Analysis Page
Visualize and analyze research categories generated from keyword clustering and categorization.
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import (
    load_data, load_css, render_page_links,
    get_category_grant_links, expand_links_to_years
)

st.set_page_config(
    page_title="Category Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_css()

render_page_links()


def create_category_years_table(categories_df, keywords_df, grants_df):
    """Create a table with one row per category per year based on active grants"""
    category_grant_links = get_category_grant_links(categories_df, keywords_df)
    return expand_links_to_years(category_grant_links, grants_df, 'category', 
                                 use_active_period=True, include_source=True)


def apply_filters(categories_df, keywords_df, grants_df, sources, fields, min_keywords, 
                  max_keywords, year_min, year_max, search_term):
    """Apply filters to categories"""
    filtered_categories = categories_df.copy()
    filtered_grants = grants_df.copy()
    
    # Add keyword_count if not present
    if 'keyword_count' not in filtered_categories.columns and 'keywords' in filtered_categories.columns:
        filtered_categories['keyword_count'] = filtered_categories['keywords'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    
    # Filter categories by field of research
    if fields:
        filtered_categories = filtered_categories[
            filtered_categories['field_of_research'].isin(fields)
        ]
    
    # Filter categories by keyword count
    if min_keywords is not None or max_keywords is not None:
        min_val = min_keywords if min_keywords is not None else 0
        max_val = max_keywords if max_keywords is not None else float('inf')
        filtered_categories = filtered_categories[
            (filtered_categories['keyword_count'] >= min_val) &
            (filtered_categories['keyword_count'] <= max_val)
        ]
    
    # Filter categories by search term
    if search_term:
        search_mask = (
            filtered_categories.index.str.contains(search_term, case=False, na=False) |
            filtered_categories['description'].str.contains(search_term, case=False, na=False)
        )
        filtered_categories = filtered_categories[search_mask]
    
    # Filter grants by source
    if sources:
        filtered_grants = filtered_grants[filtered_grants['source'].isin(sources)]
    
    # Filter grants by year range
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
    
    # Filter grants to only those linked to filtered categories
    if not filtered_categories.empty:
        category_grant_links = get_category_grant_links(filtered_categories, keywords_df)
        if not category_grant_links.empty:
            category_grant_ids = set(category_grant_links['grant_id'].unique())
            filtered_grants = filtered_grants[filtered_grants.index.isin(category_grant_ids)]
        else:
            # No grants linked to filtered categories
            filtered_grants = filtered_grants.iloc[0:0]
    else:
        # No categories matched the filters
        filtered_grants = filtered_grants.iloc[0:0]
    
    # Filter categories that have at least one grant after all filters
    if not filtered_grants.empty:
        category_grant_links = get_category_grant_links(filtered_categories, keywords_df)
        if not category_grant_links.empty:
            # Only keep categories that have grants in the filtered grants set
            valid_grant_ids = set(filtered_grants.index)
            categories_with_grants = category_grant_links[
                category_grant_links['grant_id'].isin(valid_grant_ids)
            ]['category'].unique()
            filtered_categories = filtered_categories[
                filtered_categories.index.isin(categories_with_grants)
            ]
        else:
            filtered_categories = filtered_categories.iloc[0:0]
    else:
        filtered_categories = filtered_categories.iloc[0:0]
    
    return filtered_categories, filtered_grants


def prepare_category_trends_data(filtered_categories, keywords_df, filtered_grants, 
                                 year_min, year_max, metric):
    """Prepare data for category trends visualization"""
    category_years = create_category_years_table(filtered_categories, keywords_df, filtered_grants)
    
    if category_years.empty:
        return pd.DataFrame()
    
    if year_min is not None:
        category_years = category_years[category_years['year'] >= year_min]
    if year_max is not None:
        category_years = category_years[category_years['year'] <= year_max]
    
    if metric == "funding":
        aggregated = category_years.groupby(['year', 'category'], as_index=False).agg({
            'funding_amount': 'sum'
        }).rename(columns={'funding_amount': 'value'})
    else:
        aggregated = category_years.groupby(['year', 'category'], as_index=False).agg({
            'grant_id': 'count'
        }).rename(columns={'grant_id': 'value'})
    
    # Fill in missing years with 0 for each category to show accurate drops
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        all_categories = aggregated['category'].unique()
        
        full_index = pd.MultiIndex.from_product(
            [all_years, all_categories],
            names=['year', 'category']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        aggregated = full_df.merge(aggregated, on=['year', 'category'], how='left')
        aggregated['value'] = aggregated['value'].fillna(0)
    
    return aggregated


def get_top_categories(trends_data, num_categories, metric):
    """Get top N categories by total value"""
    total_by_category = trends_data.groupby('category')['value'].sum().sort_values(ascending=False)
    return total_by_category.head(num_categories).index.tolist()


def create_category_trends_chart(data, num_categories, metric, title):
    """Create a line chart for category trends"""
    if data.empty:
        return None
    
    top_categories = get_top_categories(data, num_categories, metric)
    data_plot = data[data['category'].isin(top_categories)].copy()
    
    fig = go.Figure()
    
    for category in top_categories:
        category_data = data_plot[data_plot['category'] == category].sort_values('year')
        fig.add_trace(go.Scatter(
            x=category_data['year'],
            y=category_data['value'],
            name=category,
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


def render_sidebar(categories_df, grants_df):
    """Render sidebar controls"""
    st.sidebar.header("Filters")
    
    sources = grants_df['source'].dropna().unique().tolist() if 'source' in grants_df.columns else []
    source_filter = st.sidebar.multiselect(
        "Source", 
        sorted(sources), 
        default=["arc.gov.au"] if "arc.gov.au" in sources else []
    )
    
    fields = categories_df['field_of_research'].dropna().unique().tolist() if 'field_of_research' in categories_df.columns else []
    field_filter = st.sidebar.multiselect("Field of Research", sorted(fields))
    
    st.sidebar.subheader("Keyword Count Range")
    min_keywords = st.sidebar.number_input("Minimum Keywords", min_value=1, value=3, step=1)
    max_keywords = st.sidebar.number_input("Maximum Keywords", min_value=1, value=100, step=5)
    
    search_term = st.sidebar.text_input("Search Categories", "")
    
    min_year = int(grants_df['start_year'].min()) if 'start_year' in grants_df.columns else 2000
    max_year = int(grants_df['start_year'].max()) if 'start_year' in grants_df.columns else 2024
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    st.sidebar.header("Display Options")
    
    metric = st.sidebar.radio("Metric", ["count", "funding"], index=0)
    num_categories = st.sidebar.slider("Number of Top Categories", 5, 20, 10)
    
    return {
        'source_filter': source_filter,
        'field_filter': field_filter,
        'min_keywords': min_keywords,
        'max_keywords': max_keywords,
        'search_term': search_term,
        'year_min': year_range[0],
        'year_max': year_range[1],
        'metric': metric,
        'num_categories': num_categories
    }


def show_statistics(total_categories, filtered_categories, total_grants, filtered_grants, 
                   displayed_categories):
    """Display statistics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Categories", total_categories)
    with col2:
        st.metric("Filtered Categories", len(filtered_categories))
    with col3:
        st.metric("Total Grants", total_grants)
    with col4:
        st.metric("Filtered Grants", len(filtered_grants))
    with col5:
        st.metric("Displayed Categories", displayed_categories)


def main():
    st.header("Category Analysis")
    
    keywords_df, grants_df, categories_df = load_data()
    
    if categories_df is None or categories_df.empty:
        st.error("No category data available. Run categorization process first.")
        st.info("Run: `make categorise` to generate categories from keywords.")
        return
    
    if keywords_df is None or keywords_df.empty or grants_df is None or grants_df.empty:
        st.error("Unable to load keywords or grants data.")
        return
    
    # Add keyword_count column if not present
    if 'keyword_count' not in categories_df.columns and 'keywords' in categories_df.columns:
        categories_df['keyword_count'] = categories_df['keywords'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    
    config = render_sidebar(categories_df, grants_df)
    
    with st.spinner("Creating visualization..."):
        filtered_categories, filtered_grants = apply_filters(
            categories_df,
            keywords_df,
            grants_df,
            config['source_filter'],
            config['field_filter'],
            config['min_keywords'],
            config['max_keywords'],
            config['year_min'],
            config['year_max'],
            config['search_term']
        )
        
        if filtered_categories.empty:
            st.error("No categories found with current filters.")
            return
        
        trends_data = prepare_category_trends_data(
            filtered_categories,
            keywords_df,
            filtered_grants,
            config['year_min'],
            config['year_max'],
            config['metric']
        )
        
        if trends_data.empty:
            st.warning("No category activity in selected period.")
            return
        
        title = "Category Trends Over Time (Active Grants Only)"
        
        fig = create_category_trends_chart(
            trends_data,
            config['num_categories'],
            config['metric'],
            title
        )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            show_statistics(
                len(categories_df),
                filtered_categories,
                len(grants_df),
                filtered_grants,
                min(config['num_categories'], len(filtered_categories))
            )
        else:
            st.warning("No data available for visualization.")


if __name__ == "__main__":
    main()

