"""
Organisation Insights Page
Focus on grant, keyword, and category trends for a single organisation.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import sys
from pathlib import Path

web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data

st.set_page_config(
    page_title="Organisation Insights",
    layout="wide",
    initial_sidebar_state="expanded"
)


def create_grant_years_table(grants_df, use_active_period=True):
    """Create a table with one row per grant per year"""
    records = []
    for _, grant in grants_df.iterrows():
        start = grant.get('start_year')
        end = grant.get('end_year', start)
        if pd.isna(start):
            continue
        start = int(start)
        end = int(end) if not pd.isna(end) else start
        
        years = range(start, end + 1) if use_active_period else [start]
        funding = grant.get('funding_amount', 0)
        
        for year in years:
            records.append({
                'grant_id': grant.name,
                'year': year,
                'funding_amount': funding
            })
    
    return pd.DataFrame(records)


def get_keyword_grant_links(keywords_df):
    """Extract keyword-grant relationships"""
    records = []
    for idx, row in keywords_df.iterrows():
        keyword_name = row.get('name', idx)
        grants = row.get('grants', [])
        if isinstance(grants, list):
            for grant_id in grants:
                records.append({'keyword': keyword_name, 'grant_id': grant_id})
    return pd.DataFrame(records)


def create_keyword_years_table(keywords_df, grants_df, use_active_period=True):
    """Create a table with one row per keyword per year based on active grants"""
    keyword_grant_links = get_keyword_grant_links(keywords_df)
    
    if keyword_grant_links.empty:
        return pd.DataFrame()
    
    merged = keyword_grant_links.merge(
        grants_df[['start_year', 'end_year', 'funding_amount']], 
        left_on='grant_id', 
        right_index=True, 
        how='inner'
    )
    
    records = []
    for _, row in merged.iterrows():
        start = row.get('start_year')
        end = row.get('end_year', start)
        if pd.isna(start):
            continue
        start = int(start)
        end = int(end) if not pd.isna(end) else start
        
        years = range(start, end + 1) if use_active_period else [start]
        
        for year in years:
            records.append({
                'keyword': row['keyword'],
                'year': year,
                'grant_id': row['grant_id'],
                'funding_amount': row.get('funding_amount', 0)
            })
    
    return pd.DataFrame(records)


def get_category_grant_links(categories_df, keywords_df):
    """Extract category-grant relationships"""
    if categories_df is None or keywords_df is None:
        return pd.DataFrame()
    
    records = []
    for cat_name, cat in categories_df.iterrows():
        cat_keywords = cat.get('keywords', [])
        if not isinstance(cat_keywords, list):
            continue
        
        for kw in cat_keywords:
            if kw in keywords_df.index:
                kw_row = keywords_df.loc[[kw]]
            else:
                if 'name' in keywords_df.columns:
                    kw_row = keywords_df[keywords_df['name'] == kw]
                else:
                    continue
            
            if not kw_row.empty:
                grants = kw_row.iloc[0].get('grants', [])
                if isinstance(grants, list):
                    for grant_id in grants:
                        records.append({'category': cat_name, 'grant_id': grant_id})
    
    return pd.DataFrame(records)


def create_category_years_table(categories_df, keywords_df, grants_df, use_active_period=True):
    """Create a table with one row per category per year based on active grants"""
    category_grant_links = get_category_grant_links(categories_df, keywords_df)
    
    if category_grant_links.empty:
        return pd.DataFrame()
    
    merged = category_grant_links.merge(
        grants_df[['start_year', 'end_year', 'funding_amount']], 
        left_on='grant_id', 
        right_index=True, 
        how='inner'
    )
    
    records = []
    for _, row in merged.iterrows():
        start = row.get('start_year')
        end = row.get('end_year', start)
        if pd.isna(start):
            continue
        start = int(start)
        end = int(end) if not pd.isna(end) else start
        
        years = range(start, end + 1) if use_active_period else [start]
        
        for year in years:
            records.append({
                'category': row['category'],
                'year': year,
                'grant_id': row['grant_id'],
                'funding_amount': row.get('funding_amount', 0)
            })
    
    return pd.DataFrame(records)


def filter_grants_by_organisation(grants_df, organisation_ids, year_min, year_max, use_active_period):
    """Filter grants for specific organisations and year range"""
    if not organisation_ids or 'organisation_ids' not in grants_df.columns:
        return pd.DataFrame()
    
    exploded = grants_df.explode('organisation_ids')
    filtered = exploded[exploded['organisation_ids'].isin(organisation_ids)].copy()
    
    if filtered.empty:
        return pd.DataFrame()
    
    if year_min is not None or year_max is not None:
        start = filtered['start_year'].fillna(filtered['end_year'])
        end = filtered['end_year'].fillna(filtered['start_year'])
        
        mask = pd.Series(True, index=filtered.index)
        if year_min is not None:
            if use_active_period:
                mask &= end >= year_min
            else:
                mask &= start >= year_min
        if year_max is not None:
            if use_active_period:
                mask &= start <= year_max
            else:
                mask &= start <= year_max
        
        filtered = filtered[mask]
    
    return filtered[~filtered.index.duplicated(keep='first')]


def prepare_grant_trends(filtered_grants, year_min, year_max, use_active_period):
    """Prepare grant trends data"""
    grant_years = create_grant_years_table(filtered_grants, use_active_period)
    
    if grant_years.empty:
        return pd.DataFrame()
    
    if year_min is not None:
        grant_years = grant_years[grant_years['year'] >= year_min]
    if year_max is not None:
        grant_years = grant_years[grant_years['year'] <= year_max]
    
    aggregated = grant_years.groupby('year', as_index=False).agg({
        'grant_id': 'count',
        'funding_amount': 'sum'
    }).rename(columns={'grant_id': 'grant_count', 'funding_amount': 'total_funding'})
    
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        full_df = pd.DataFrame({'year': list(all_years)})
        aggregated = full_df.merge(aggregated, on='year', how='left')
        aggregated['grant_count'] = aggregated['grant_count'].fillna(0)
        aggregated['total_funding'] = aggregated['total_funding'].fillna(0)
    
    return aggregated


def prepare_keyword_trends(keywords_df, filtered_grants, year_min, year_max, use_active_period, top_n):
    """Prepare keyword trends data"""
    if keywords_df is None or keywords_df.empty:
        return pd.DataFrame()
    
    grant_ids = set(filtered_grants.index)
    org_keywords = keywords_df.copy()
    org_keywords['grants'] = org_keywords['grants'].apply(
        lambda ids: [g for g in ids if g in grant_ids] if isinstance(ids, list) else []
    )
    org_keywords = org_keywords[org_keywords['grants'].apply(len) > 0]
    
    if org_keywords.empty:
        return pd.DataFrame()
    
    keyword_years = create_keyword_years_table(org_keywords, filtered_grants, use_active_period)
    
    if keyword_years.empty:
        return pd.DataFrame()
    
    if year_min is not None:
        keyword_years = keyword_years[keyword_years['year'] >= year_min]
    if year_max is not None:
        keyword_years = keyword_years[keyword_years['year'] <= year_max]
    
    aggregated = keyword_years.groupby(['year', 'keyword'], as_index=False).agg({
        'grant_id': 'count',
        'funding_amount': 'sum'
    }).rename(columns={'grant_id': 'grant_count', 'funding_amount': 'total_funding'})
    
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        all_keywords = aggregated['keyword'].unique()
        
        full_index = pd.MultiIndex.from_product(
            [all_years, all_keywords],
            names=['year', 'keyword']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        aggregated = full_df.merge(aggregated, on=['year', 'keyword'], how='left')
        aggregated['grant_count'] = aggregated['grant_count'].fillna(0)
        aggregated['total_funding'] = aggregated['total_funding'].fillna(0)
    
    return aggregated


def prepare_category_trends(categories_df, keywords_df, filtered_grants, year_min, year_max, use_active_period, top_n):
    """Prepare category trends data"""
    if categories_df is None or categories_df.empty:
        return pd.DataFrame()
    
    grant_ids = set(filtered_grants.index)
    org_keywords = keywords_df.copy()
    org_keywords['grants'] = org_keywords['grants'].apply(
        lambda ids: [g for g in ids if g in grant_ids] if isinstance(ids, list) else []
    )
    org_keywords = org_keywords[org_keywords['grants'].apply(len) > 0]
    keyword_names = set(org_keywords['name']) if 'name' in org_keywords.columns else set(org_keywords.index)
    
    org_categories = categories_df.copy()
    org_categories = org_categories[org_categories['keywords'].apply(
        lambda items: any(keyword in keyword_names for keyword in items) if isinstance(items, list) else False
    )]
    
    if org_categories.empty:
        return pd.DataFrame()
    
    category_years = create_category_years_table(org_categories, org_keywords, filtered_grants, use_active_period)
    
    if category_years.empty:
        return pd.DataFrame()
    
    if year_min is not None:
        category_years = category_years[category_years['year'] >= year_min]
    if year_max is not None:
        category_years = category_years[category_years['year'] <= year_max]
    
    aggregated = category_years.groupby(['year', 'category'], as_index=False).agg({
        'grant_id': 'count',
        'funding_amount': 'sum'
    }).rename(columns={'grant_id': 'grant_count', 'funding_amount': 'total_funding'})
    
    if not aggregated.empty:
        all_years = range(int(aggregated['year'].min()), int(aggregated['year'].max()) + 1)
        all_categories = aggregated['category'].unique()
        
        full_index = pd.MultiIndex.from_product(
            [all_years, all_categories],
            names=['year', 'category']
        )
        full_df = pd.DataFrame(index=full_index).reset_index()
        
        aggregated = full_df.merge(aggregated, on=['year', 'category'], how='left')
        aggregated['grant_count'] = aggregated['grant_count'].fillna(0)
        aggregated['total_funding'] = aggregated['total_funding'].fillna(0)
    
    return aggregated


def apply_smoothing(data, entity_col, value_col, window):
    """Apply rolling average smoothing to time series data"""
    if window <= 1 or data.empty:
        return data
    
    smoothed = data.copy()
    for entity in data[entity_col].unique():
        mask = smoothed[entity_col] == entity
        smoothed.loc[mask, value_col] = (
            smoothed.loc[mask, value_col]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
    
    return smoothed


def get_top_entities(data, entity_col, value_col, top_n):
    """Get top N entities by total value"""
    total_by_entity = data.groupby(entity_col)[value_col].sum().sort_values(ascending=False)
    return total_by_entity.head(top_n).index.tolist()


def create_line_chart(data, entity_col, value_col, top_n, title, y_label):
    """Create a line chart for trends"""
    if data.empty:
        return None
    
    top_entities = get_top_entities(data, entity_col, value_col, top_n)
    data_plot = data[data[entity_col].isin(top_entities)].copy()
    
    fig = go.Figure()
    
    for entity in top_entities:
        entity_data = data_plot[data_plot[entity_col] == entity].sort_values('year')
        fig.add_trace(go.Scatter(
            x=entity_data['year'],
            y=entity_data[value_col],
            name=entity,
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_label,
        height=500,
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


def create_area_chart(data, value_col, title, y_label):
    """Create an area chart for single entity trends"""
    if data.empty:
        return None
    
    fig = go.Figure()
    
    data_sorted = data.sort_values('year')
    fig.add_trace(go.Scatter(
        x=data_sorted['year'],
        y=data_sorted[value_col],
        mode='lines',
        fill='tozeroy',
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Year',
        yaxis_title=y_label,
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    
    return fig


def render_sidebar(grants_df):
    """Render sidebar controls"""
    st.sidebar.header("Organisation Selection")
    
    if 'organisation_ids' not in grants_df.columns:
        st.sidebar.error("No organisation data available")
        return None
    
    exploded = grants_df.explode('organisation_ids').dropna(subset=['organisation_ids'])
    organisation_counts = (
        exploded.groupby('organisation_ids').size()
        .sort_values(ascending=False)
    )
    
    if organisation_counts.empty:
        st.sidebar.error("No organisations found")
        return None
    
    organisation_options = organisation_counts.index.tolist()
    
    search_phrase = st.sidebar.text_input(
        "Filter organisations (regex)",
        value="",
        help="Type a regex pattern to select organisations, or leave empty to use multiselect"
    ).strip()
    
    if search_phrase:
        try:
            mask = pd.Series(organisation_options).str.contains(search_phrase, case=False, regex=True, na=False)
            selected_orgs = pd.Series(organisation_options)[mask].tolist()
            if selected_orgs:
                st.sidebar.success(f"Selected {len(selected_orgs)} organisations matching pattern")
                with st.sidebar.expander("Show selected organisations"):
                    for org in selected_orgs[:20]:
                        st.write(f"• {org}")
                    if len(selected_orgs) > 20:
                        st.write(f"... and {len(selected_orgs) - 20} more")
            else:
                st.sidebar.warning("No organisations match the regex pattern")
                return None
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {str(e)}")
            return None
    else:
        selected_orgs = st.sidebar.multiselect(
            "Select organisations",
            options=organisation_options,
            default=organisation_options[:3] if organisation_options else [],
            help="Select one or more organisations to analyze"
        )
        
        if not selected_orgs:
            st.sidebar.warning("Please select at least one organisation")
            return None
    
    if len(selected_orgs) == 1:
        selection_label = selected_orgs[0]
    else:
        selection_label = f"{len(selected_orgs)} organisations"
    
    st.sidebar.header("Filters")
    
    min_year = int(grants_df['start_year'].min()) if 'start_year' in grants_df.columns else 2000
    max_year = int(grants_df['start_year'].max()) if 'start_year' in grants_df.columns else 2024
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    use_active_period = st.sidebar.checkbox(
        "Count grants across active years",
        value=True,
        help="Include grants in all years between start and end"
    )
    
    st.sidebar.header("Display Options")
    
    metric = st.sidebar.radio("Metric", ["count", "funding"], index=0)
    top_keywords = st.sidebar.slider("Number of Top Keywords", 5, 20, 10)
    top_categories = st.sidebar.slider("Number of Top Categories", 5, 15, 8)
    
    smooth_trends = st.sidebar.checkbox("Smooth trends", value=False)
    smoothing_window = 1
    if smooth_trends:
        smoothing_window = st.sidebar.slider("Smoothing window (years)", 2, 10, 3)
    
    return {
        'selected_orgs': selected_orgs,
        'selection_label': selection_label,
        'year_min': year_range[0],
        'year_max': year_range[1],
        'use_active_period': use_active_period,
        'metric': metric,
        'top_keywords': top_keywords,
        'top_categories': top_categories,
        'smooth_trends': smooth_trends,
        'smoothing_window': smoothing_window
    }


def main():
    st.title("Organisation Insights")
    st.caption("Explore grants, keywords, and categories for a specific organisation")
    
    keywords_df, grants_df, categories_df = load_data()
    
    if grants_df is None or grants_df.empty:
        st.error("Unable to load grants data.")
        return
    
    config = render_sidebar(grants_df)
    if config is None:
        return
    
    with st.spinner("Loading organisation data..."):
        filtered_grants = filter_grants_by_organisation(
            grants_df,
            config['selected_orgs'],
            config['year_min'],
            config['year_max'],
            config['use_active_period']
        )
        
        if filtered_grants.empty:
            st.warning("No grants found for selected organisation(s) in the specified time range.")
            return
        
        value_col = 'total_funding' if config['metric'] == 'funding' else 'grant_count'
        y_label = 'Funding Amount' if config['metric'] == 'funding' else 'Grant Count'
        
        total_grants = len(filtered_grants)
        total_funding = filtered_grants['funding_amount'].fillna(0).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Organisation", config['selection_label'])
        with col2:
            st.metric("Total Grants", f"{total_grants:,}")
        with col3:
            st.metric("Total Funding", f"${total_funding:,.0f}")
        
        st.markdown("---")
        st.subheader("Grant Trends")
        
        grant_trends = prepare_grant_trends(
            filtered_grants,
            config['year_min'],
            config['year_max'],
            config['use_active_period']
        )
        
        if not grant_trends.empty:
            if config['smooth_trends']:
                grant_trends = grant_trends.copy()
                grant_trends[value_col] = (
                    grant_trends[value_col]
                    .rolling(window=config['smoothing_window'], center=True, min_periods=1)
                    .mean()
                )
            
            fig = create_area_chart(
                grant_trends,
                value_col,
                f"Grant Trends for {config['selection_label']}",
                y_label
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No grant trend data available.")
        
        st.markdown("---")
        st.subheader("Keyword Trends")
        
        keyword_trends = prepare_keyword_trends(
            keywords_df,
            filtered_grants,
            config['year_min'],
            config['year_max'],
            config['use_active_period'],
            config['top_keywords']
        )
        
        if not keyword_trends.empty:
            if config['smooth_trends']:
                keyword_trends = apply_smoothing(
                    keyword_trends,
                    'keyword',
                    value_col,
                    config['smoothing_window']
                )
            
            fig = create_line_chart(
                keyword_trends,
                'keyword',
                value_col,
                config['top_keywords'],
                f"Top Keywords for {config['selection_label']}",
                y_label
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No keyword data available for this organisation.")
        
        st.markdown("---")
        st.subheader("Category Trends")
        
        category_trends = prepare_category_trends(
            categories_df,
            keywords_df,
            filtered_grants,
            config['year_min'],
            config['year_max'],
            config['use_active_period'],
            config['top_categories']
        )
        
        if not category_trends.empty:
            if config['smooth_trends']:
                category_trends = apply_smoothing(
                    category_trends,
                    'category',
                    value_col,
                    config['smoothing_window']
                )
            
            fig = create_line_chart(
                category_trends,
                'category',
                value_col,
                config['top_categories'],
                f"Top Categories for {config['selection_label']}",
                y_label
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available for this organisation.")


if __name__ == "__main__":
    main()
