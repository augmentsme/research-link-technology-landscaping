"""
Grants Analysis Page
Analyze grant distributions over time and view grants with their extracted keywords.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data, load_css, render_page_links, expand_grants_to_years

st.set_page_config(
    page_title="Grant Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

render_page_links()


def create_grant_years_table(grants_df: pd.DataFrame, active: bool = False) -> pd.DataFrame:
    """Create a table with one row per grant per year per organization"""
    return expand_grants_to_years(grants_df, use_active_period=active, include_org_ids=True)


def apply_filters(grants_df: pd.DataFrame, sources, year_min, year_max) -> pd.DataFrame:
    """Apply filters to grants dataframe"""
    filtered = grants_df.copy()
    
    if sources:
        filtered = filtered[filtered['source'].isin(sources)]
    
    if 'start_year' in filtered.columns:
        start = filtered['start_year']
        end = filtered['end_year']
        start = start.fillna(end)
        end = end.fillna(start)
        mask = pd.Series(True, index=filtered.index)
        if year_min is not None:
            mask &= end >= year_min
        if year_max is not None:
            mask &= start <= year_max
        filtered = filtered[mask]
    
    return filtered


def prepare_aggregated_data(filtered_grants: pd.DataFrame, year_min, year_max) -> pd.DataFrame:
    """Aggregate grants by year and organization"""
    grant_records = create_grant_years_table(filtered_grants, active=True)
    filtered_records = grant_records.dropna(subset=['organisation_id'])
    
    if year_min is not None:
        filtered_records = filtered_records[filtered_records['year'] >= year_min]
    if year_max is not None:
        filtered_records = filtered_records[filtered_records['year'] <= year_max]
    
    aggregated = filtered_records.groupby(['year', 'organisation_id'], as_index=False).agg({
        'grant_id': 'count',
        'funding_amount': 'sum'
    })
    
    aggregated = aggregated.rename(columns={'grant_id': 'grant_count', 'funding_amount': 'total_funding'})
    
    # Fill in missing years with 0 for each organization to show accurate drops
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        all_orgs = aggregated['organisation_id'].unique()
        
        # Create a complete grid of year x organization combinations
        full_index = pd.MultiIndex.from_product(
            [all_years, all_orgs],
            names=['year', 'organisation_id']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        # Merge with actual data and fill missing values with 0
        aggregated = full_df.merge(aggregated, on=['year', 'organisation_id'], how='left')
        aggregated['grant_count'] = aggregated['grant_count'].fillna(0)
        aggregated['total_funding'] = aggregated['total_funding'].fillna(0)
    
    return aggregated


def create_stacked_area_chart(data: pd.DataFrame, metric: str, num_entities: int, title: str) -> go.Figure:
    """Create a stacked area chart"""
    value_col = 'total_funding' if metric == "funding" else 'grant_count'
    
    total_by_entity = data.groupby('organisation_id')[value_col].sum().sort_values(ascending=False)
    top_entities = total_by_entity.head(num_entities).index.tolist()
    
    data_plot = data[data['organisation_id'].isin(top_entities)].copy()
    
    fig = go.Figure()
    
    for entity in top_entities:
        entity_data = data_plot[data_plot['organisation_id'] == entity].sort_values('year')
        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data[value_col],
            name=entity,
            mode='lines',
            stackgroup='one'
        ))
    
    y_label = 'Funding Amount' if metric == "funding" else 'Yearly Grant Count'
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_label,
        height=600,
        hovermode='x unified'
    )
    
    return fig


def render_sidebar(grants_df: pd.DataFrame):
    """Render sidebar controls"""
    st.sidebar.header("Filters")
    
    sources = grants_df['source'].unique().tolist() if 'source' in grants_df.columns else []
    source_filter = st.sidebar.multiselect("Source", sources, default=["arc.gov.au"])
    
    min_year = int(grants_df['start_year'].min()) if 'start_year' in grants_df.columns else 2000
    max_year = int(grants_df['start_year'].max()) if 'start_year' in grants_df.columns else 2024
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    st.sidebar.header("Display Options")
    
    metric = st.sidebar.radio("Metric", ["count", "funding"], index=1)
    num_entities = st.sidebar.slider("Number of Top Organizations", 5, 20, 10)
    
    return {
        'source_filter': source_filter,
        'year_min': year_range[0],
        'year_max': year_range[1],
        'metric': metric,
        'num_entities': num_entities
    }


def show_statistics(total_grants: int, filtered_grants: pd.DataFrame, aggregated: pd.DataFrame):
    """Display statistics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    grants_per_year = aggregated.groupby('year')['grant_count'].sum()
    total_funding = filtered_grants.get('funding_amount', pd.Series(dtype=float)).fillna(0).sum()
    
    with col1:
        st.metric("Total Grants", total_grants)
    with col2:
        st.metric("Filtered Grants", len(filtered_grants))
    with col3:
        st.metric("Filtered Funding", f"${total_funding:,.0f}")
    with col4:
        st.metric("Unique Organizations", aggregated['organisation_id'].nunique())
    with col5:
        if not grants_per_year.empty:
            peak_year = grants_per_year.idxmax()
            peak_count = grants_per_year.max()
            st.metric("Peak Year", f"{peak_year} ({int(peak_count)} grants)")


def main():
    st.header("Grants Analysis")
    
    _, grants_df, _ = load_data()
    
    if grants_df is None or grants_df.empty:
        st.error("Unable to load grants data.")
        return
    
    config = render_sidebar(grants_df)
    
    with st.spinner("Creating visualization..."):
        filtered_grants = apply_filters(
            grants_df,
            config['source_filter'],
            config['year_min'],
            config['year_max']
        )
        
        if filtered_grants.empty:
            st.error("No grants found with current filters.")
            return
        
        aggregated = prepare_aggregated_data(
            filtered_grants,
            config['year_min'],
            config['year_max']
        )
        
        if aggregated.empty:
            st.warning("No grant activity in selected period.")
            return
        
        title = "Yearly Grant Distribution by Year and Organization"
        
        fig = create_stacked_area_chart(
            aggregated,
            config['metric'],
            config['num_entities'],
            title
        )
        
        st.plotly_chart(fig, use_container_width=True)
        show_statistics(len(grants_df), filtered_grants, aggregated)


if __name__ == "__main__":
    main()

