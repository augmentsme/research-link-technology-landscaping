"""
Comparing Organisations Page
Create connected scatterplots comparing research activity between two sets of organisations over time.
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
    page_title="Comparing Organisations",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


def filter_grants_by_organisations(grants_df, organisation_ids):
    """Filter grants for specific organisations"""
    if not organisation_ids or 'organisation_ids' not in grants_df.columns:
        return pd.DataFrame()
    
    exploded = grants_df.explode('organisation_ids')
    filtered = exploded[exploded['organisation_ids'].isin(organisation_ids)].copy()
    
    return filtered[~filtered.index.duplicated(keep='first')]


def calculate_category_metrics_by_year(grants_df, categories_df, keywords_df, selected_categories, metric='count'):
    """Calculate category metrics by year and organisation"""
    if grants_df.empty or categories_df is None or keywords_df is None:
        return pd.DataFrame()
    
    category_grant_links = get_category_grant_links(categories_df, keywords_df)
    
    if category_grant_links.empty:
        return pd.DataFrame()
    
    if selected_categories:
        category_grant_links = category_grant_links[category_grant_links['category'].isin(selected_categories)]
    
    if category_grant_links.empty:
        return pd.DataFrame()
    
    merged = category_grant_links.merge(
        grants_df[['start_year', 'funding_amount', 'organisation_ids']], 
        left_on='grant_id', 
        right_index=True, 
        how='inner'
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    records = []
    for _, row in merged.iterrows():
        year = row.get('start_year')
        if pd.isna(year):
            continue
        year = int(year)
        
        org_ids = row.get('organisation_ids', [])
        if not isinstance(org_ids, list):
            org_ids = [org_ids] if pd.notna(org_ids) else []
        
        for org_id in org_ids:
            records.append({
                'category': row['category'],
                'year': year,
                'organisation_id': org_id,
                'grant_id': row['grant_id'],
                'funding_amount': row.get('funding_amount', 0)
            })
    
    df = pd.DataFrame(records)
    
    if df.empty:
        return df
    
    if metric == 'funding':
        aggregated = df.groupby(['year', 'organisation_id', 'category'], as_index=False).agg({
            'funding_amount': 'sum'
        }).rename(columns={'funding_amount': 'value'})
    else:
        aggregated = df.groupby(['year', 'organisation_id', 'category'], as_index=False).agg({
            'grant_id': 'count'
        }).rename(columns={'grant_id': 'value'})
    
    return aggregated


def calculate_cumulative_metrics(data, extend_to_year=None):
    """Calculate cumulative metrics over time for each category and org_group
    
    Args:
        data: DataFrame with columns category, org_group, year, value
        extend_to_year: Optional year to extend all cumulative series to (uses last value)
    """
    if data.empty:
        return data
    
    # First, aggregate across all organisations within each category, org_group, and year
    aggregated = data.groupby(['category', 'org_group', 'year'], as_index=False).agg({
        'value': 'sum'
    })
    
    # Determine global year range
    global_min_year = int(aggregated['year'].min())
    global_max_year = int(aggregated['year'].max())
    if extend_to_year:
        global_max_year = max(global_max_year, extend_to_year)
    
    cumulative_data = []
    
    for (category, org_group), group in aggregated.groupby(['category', 'org_group']):
        group_sorted = group.sort_values('year')
        
        # Get ALL years that exist in the data for this category/org_group
        years = sorted(group_sorted['year'].unique())
        if not years:
            continue
            
        # Extend to global range to ensure all series have same year coverage
        min_year = min(years)
        max_year = global_max_year
        all_years = range(min_year, max_year + 1)
        
        cumulative_value = 0
        for year in all_years:
            year_data = group_sorted[group_sorted['year'] == year]
            
            # Add this year's value to cumulative (0 if year not in data)
            if not year_data.empty:
                year_value = year_data['value'].iloc[0]
                cumulative_value += year_value
            
            # Always append a record for every year in the range
            cumulative_data.append({
                'category': category,
                'org_group': org_group,
                'year': year,
                'value': cumulative_value
            })
    
    return pd.DataFrame(cumulative_data)


def calculate_category_slopes(data):
    """Calculate the slope (trajectory direction) for each category"""
    if data.empty:
        return []
    
    slopes = []
    
    for category in data['category'].unique():
        cat_data = data[data['category'] == category]
        
        years = sorted(cat_data['year'].unique())
        if len(years) < 2:
            continue
        
        # Get values for first and last year
        first_year = years[0]
        last_year = years[-1]
        
        first_org1 = cat_data[(cat_data['year'] == first_year) & (cat_data['org_group'] == 'org1')]['value'].sum()
        first_org2 = cat_data[(cat_data['year'] == first_year) & (cat_data['org_group'] == 'org2')]['value'].sum()
        
        last_org1 = cat_data[(cat_data['year'] == last_year) & (cat_data['org_group'] == 'org1')]['value'].sum()
        last_org2 = cat_data[(cat_data['year'] == last_year) & (cat_data['org_group'] == 'org2')]['value'].sum()
        
        # Calculate slope: (y2 - y1) / (x2 - x1)
        dx = last_org1 - first_org1
        dy = last_org2 - first_org2
        
        # Handle division by zero
        if dx == 0:
            if dy == 0:
                slope = 0
            else:
                slope = float('inf') if dy > 0 else float('-inf')
        else:
            slope = dy / dx
        
        slopes.append({
            'category': category,
            'slope': slope,
            'dx': dx,
            'dy': dy
        })
    
    return slopes


def get_top_and_bottom_slope_categories(slopes, n=3):
    """Get categories with the n highest and n lowest slopes"""
    if not slopes:
        return []
    
    # Filter out infinite slopes for ranking
    finite_slopes = [s for s in slopes if not (s['slope'] == float('inf') or s['slope'] == float('-inf'))]
    
    if not finite_slopes:
        return [s['category'] for s in slopes[:min(10, len(slopes))]]
    
    # Sort by slope
    sorted_slopes = sorted(finite_slopes, key=lambda x: x['slope'])
    
    # Get bottom n and top n
    bottom_n = sorted_slopes[:min(n, len(sorted_slopes))]
    top_n = sorted_slopes[-min(n, len(sorted_slopes)):]
    
    # Combine and get unique categories
    selected = bottom_n + top_n
    categories = [s['category'] for s in selected]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_categories = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            unique_categories.append(cat)
    
    return unique_categories



def select_organisations_widget(label, key_prefix, organisation_options, default_regex=""):
    """Widget for selecting organisations with regex or multiselect"""
    search_phrase = st.text_input(
        f"{label} (regex)",
        value=default_regex,
        key=f"{key_prefix}_regex",
        help="Type a regex pattern to select organisations, or leave empty to use multiselect"
    ).strip()
    
    if search_phrase:
        try:
            normalized_pattern = search_phrase.replace(' ', '[_ ]')
            mask = pd.Series(organisation_options).str.contains(normalized_pattern, case=False, regex=True, na=False)
            selected_orgs = pd.Series(organisation_options)[mask].tolist()
            if selected_orgs:
                st.success(f"Selected {len(selected_orgs)} organisations matching pattern")
                with st.expander("Show selected organisations"):
                    for org in selected_orgs[:20]:
                        st.write(f"• {org}")
                    if len(selected_orgs) > 20:
                        st.write(f"... and {len(selected_orgs) - 20} more")
                return selected_orgs
            else:
                st.warning("No organisations match the regex pattern")
                return []
        except re.error as e:
            st.error(f"Invalid regex pattern: {str(e)}")
            return []
    else:
        selected_orgs = st.multiselect(
            f"Select {label.lower()}",
            options=organisation_options,
            default=[],
            key=f"{key_prefix}_multiselect",
            help="Select one or more organisations"
        )
        return selected_orgs


def create_connected_scatterplot(data, org1_label, org2_label, category_label, metric_label):
    """Create a connected scatterplot comparing two organisation groups over time"""
    if data.empty:
        return None
    
    fig = go.Figure()
    
    years = sorted(data['year'].unique())
    
    import plotly.express as px
    
    # Distinct colors for each category
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10
    
    categories = list(data['category'].unique())
    
    for cat_idx, category in enumerate(categories):
        cat_data = data[data['category'] == category].copy()
        
        # Get single color for this category
        color = colors[cat_idx % len(colors)]
        
        x_vals = []
        y_vals = []
        year_labels = []
        
        # Track last known values for each org_group
        last_org1_val = 0
        last_org2_val = 0
        
        for year in years:
            year_data = cat_data[cat_data['year'] == year]
            
            org1_data = year_data[year_data['org_group'] == 'org1']
            org2_data = year_data[year_data['org_group'] == 'org2']
            
            # Use last known value if no data for this year
            if not org1_data.empty:
                last_org1_val = org1_data['value'].iloc[0]
            if not org2_data.empty:
                last_org2_val = org2_data['value'].iloc[0]
            
            org1_val = last_org1_val
            org2_val = last_org2_val
            
            x_vals.append(org1_val)
            y_vals.append(org2_val)
            year_labels.append(year)
        
        if len(x_vals) > 0:
            # Draw line with single color
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                line=dict(color=color, width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add markers with same color
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=category,
                text=[f"{category}<br>Year: {year}<br>{org1_label}: {x}<br>{org2_label}: {y}" 
                      for year, x, y in zip(year_labels, x_vals, y_vals)],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1, color='white')
                )
            ))
            
            # Add year labels at start and end
            if len(year_labels) > 0:
                fig.add_trace(go.Scatter(
                    x=[x_vals[0]],
                    y=[y_vals[0]],
                    mode='text',
                    text=[str(year_labels[0])],
                    textposition='top center',
                    textfont=dict(size=10, color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[x_vals[-1]],
                    y=[y_vals[-1]],
                    mode='text',
                    text=[str(year_labels[-1])],
                    textposition='top center',
                    textfont=dict(size=10, color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title=f"Connected Scatterplot: {category_label}",
        xaxis_title=f"{org1_label} ({metric_label})",
        yaxis_title=f"{org2_label} ({metric_label})",
        height=700,
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def render_sidebar(grants_df, categories_df, auto_selected_categories=None):
    """Render sidebar controls"""
    st.sidebar.header("Organisation Groups")
    
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
    
    st.sidebar.subheader("Group 1 (X-axis)")
    org_group1 = select_organisations_widget(
        "Organisation Group 1", 
        "org1", 
        organisation_options,
        default_regex="au:melbourne_university"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Group 2 (Y-axis)")
    org_group2 = select_organisations_widget(
        "Organisation Group 2", 
        "org2", 
        organisation_options,
        default_regex="au:monash_university"
    )
    
    if not org_group1:
        st.sidebar.warning("Please select organisations for Group 1")
        return None
    
    if not org_group2:
        st.sidebar.warning("Please select organisations for Group 2")
        return None
    
    st.sidebar.header("Categories")
    
    if categories_df is None or categories_df.empty:
        st.sidebar.error("No categories available")
        return None
    
    category_options = categories_df.index.tolist()
    
    # Use auto-selected categories if available, otherwise use first category
    default_categories = auto_selected_categories if auto_selected_categories else [category_options[0]]
    
    selected_categories = st.sidebar.multiselect(
        "Select categories to compare",
        options=category_options,
        default=default_categories,
        help="Auto-selected: 3 largest and 3 smallest slope categories"
    )
    
    if not selected_categories:
        st.sidebar.warning("Please select at least one category")
        return None
    
    st.sidebar.header("Filters")
    
    min_year = int(grants_df['start_year'].min()) if 'start_year' in grants_df.columns else 2000
    max_year = int(grants_df['start_year'].max()) if 'start_year' in grants_df.columns else 2024
    year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
    
    st.sidebar.header("Display Options")
    
    metric = st.sidebar.radio("Metric", ["count", "funding"], index=0)
    
    org1_label = st.sidebar.text_input("Group 1 Label", value="Group 1")
    org2_label = st.sidebar.text_input("Group 2 Label", value="Group 2")
    
    return {
        'org_group1': org_group1,
        'org_group2': org_group2,
        'org1_label': org1_label or "Group 1",
        'org2_label': org2_label or "Group 2",
        'selected_categories': selected_categories,
        'year_min': year_range[0],
        'year_max': year_range[1],
        'metric': metric
    }


def main():
    st.title("Comparing Organisations")
    st.caption("Compare research activity between two groups of organisations using connected scatterplots")
    
    keywords_df, grants_df, categories_df = load_data()
    
    if grants_df is None or grants_df.empty:
        st.error("Unable to load grants data.")
        return
    
    # Initialize session state for auto-selected categories
    if 'auto_categories' not in st.session_state:
        st.session_state.auto_categories = None
    
    # First render to get org groups
    config = render_sidebar(grants_df, categories_df, st.session_state.auto_categories)
    if config is None:
        return
    
    with st.spinner("Preparing comparison data..."):
        org1_grants = filter_grants_by_organisations(grants_df, config['org_group1'])
        org2_grants = filter_grants_by_organisations(grants_df, config['org_group2'])
        
        if org1_grants.empty and org2_grants.empty:
            st.warning("No grants found for selected organisations.")
            return
        
        # Calculate data for ALL categories to find slopes
        org1_data_all = calculate_category_metrics_by_year(
            org1_grants, 
            categories_df, 
            keywords_df, 
            None,  # Get all categories
            config['metric']
        )
        org1_data_all['org_group'] = 'org1'
        
        org2_data_all = calculate_category_metrics_by_year(
            org2_grants, 
            categories_df, 
            keywords_df, 
            None,  # Get all categories
            config['metric']
        )
        org2_data_all['org_group'] = 'org2'
        
        combined_data_all = pd.concat([org1_data_all, org2_data_all], ignore_index=True)
        combined_data_all = calculate_cumulative_metrics(combined_data_all)
        
        # Calculate slopes for all categories
        slopes = calculate_category_slopes(combined_data_all)
        auto_categories = get_top_and_bottom_slope_categories(slopes, n=3)
        
        # Update session state if categories changed
        if st.session_state.auto_categories != auto_categories:
            st.session_state.auto_categories = auto_categories
            st.rerun()
        
        # Filter to selected categories
        combined_data = combined_data_all[combined_data_all['category'].isin(config['selected_categories'])].copy()
        
        combined_data = combined_data[
            (combined_data['year'] >= config['year_min']) & 
            (combined_data['year'] <= config['year_max'])
        ]
        
        if combined_data.empty:
            st.warning("No data available for the selected parameters.")
            return
        
        metric_label = 'Funding Amount' if config['metric'] == 'funding' else 'Grant Count'
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(config['org1_label'], f"{len(config['org_group1'])} orgs")
        with col2:
            st.metric("Grants", f"{len(org1_grants):,}")
        with col3:
            st.metric(config['org2_label'], f"{len(config['org_group2'])} orgs")
        with col4:
            st.metric("Grants", f"{len(org2_grants):,}")
        
        st.markdown("---")
        
        category_label = f"Selected Categories ({len(config['selected_categories'])})"
        
        fig = create_connected_scatterplot(
            combined_data,
            config['org1_label'],
            config['org2_label'],
            category_label,
            metric_label
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **How to read this chart:**\n"
                   "- Each line represents a category's **cumulative** trajectory over time\n"
                   "- Values accumulate from start year onwards (grants persist after they end)\n"
                   "- Marker color indicates time: darker (earlier years) → lighter (later years)\n"
                   "- Points closer to the diagonal indicate similar cumulative activity between groups\n"
                   "- Trajectories typically move away from origin showing accumulated growth\n"
                   "- Year labels are shown at the start and end of each trajectory")
        else:
            st.warning("Unable to create chart with the current data.")


if __name__ == "__main__":
    main()
