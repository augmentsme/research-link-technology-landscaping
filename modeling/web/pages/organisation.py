"""
Organisation Insights Page
Focus on grant, keyword, and category trends for a single organisation.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure shared utilities are importable
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from shared_utils import load_data  # noqa: E402
from visualisation import (  # noqa: E402
    TrendsConfig,
    TrendsDataPreparation,
    TrendsVisualizer,
    compute_active_years,
)


st.set_page_config(
    page_title="Organisation Insights",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _select_year_range(grants_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    if 'start_year' not in grants_df.columns:
        return None, None
    valid_years = grants_df['start_year'].dropna().astype(int)
    if valid_years.empty:
        return None, None
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())
    if min_year == max_year:
        st.sidebar.caption(f"Year range fixed at {min_year}")
        return min_year, max_year
    selected_min, selected_max = st.sidebar.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help="Limit grants to those active within this period.",
    )
    return selected_min, selected_max


def _format_org_label(org_id: str) -> str:
    return org_id.replace("_", " ")


def _filter_grants(
    grants_df: pd.DataFrame,
    organisation_id: str,
    year_min: Optional[int],
    year_max: Optional[int],
    use_active_period: bool,
) -> pd.DataFrame:
    if 'organisation_ids' not in grants_df.columns:
        return pd.DataFrame()

    exploded = grants_df.explode('organisation_ids')
    filtered = exploded[exploded['organisation_ids'] == organisation_id].copy()
    if filtered.empty:
        return pd.DataFrame()

    if year_min is None and year_max is None:
        return filtered.drop_duplicates(subset='id').assign(organisation=organisation_id)

    if use_active_period:
        mask = filtered.apply(
            lambda row: bool(
                compute_active_years(
                    row.get('start_year'),
                    row.get('end_year'),
                    use_active_period=True,
                    min_year=year_min,
                    max_year=year_max,
                )
            ),
            axis=1,
        )
        filtered = filtered[mask]
        return filtered.drop_duplicates(subset='id').assign(organisation=organisation_id)

    if year_min is not None:
        filtered = filtered[filtered['start_year'] >= year_min]
    if year_max is not None:
        filtered = filtered[filtered['start_year'] <= year_max]
    return filtered.drop_duplicates(subset='id').assign(organisation=organisation_id)


def _prepare_keyword_data(keywords_df: pd.DataFrame, grant_ids: set[str]) -> pd.DataFrame:
    if keywords_df is None or keywords_df.empty:
        return pd.DataFrame()
    keywords = keywords_df.copy()
    keywords['grants'] = keywords['grants'].apply(
        lambda ids: [g for g in ids if g in grant_ids] if isinstance(ids, list) else []
    )
    keywords = keywords[keywords['grants'].apply(len) > 0]
    return keywords


def _prepare_category_data(categories_df: pd.DataFrame, keyword_names: set[str]) -> pd.DataFrame:
    if categories_df is None or categories_df.empty:
        return pd.DataFrame()
    categories = categories_df.copy()
    if 'keywords' not in categories.columns:
        return pd.DataFrame()
    categories = categories[categories['keywords'].apply(
        lambda items: any(keyword in keyword_names for keyword in items) if isinstance(items, list) else False
    )]
    return categories


def _top_entities(df: pd.DataFrame, entity_col: str, limit: int) -> List[str]:
    if df.empty or entity_col not in df.columns:
        return []
    if entity_col == 'name' and 'grants' in df.columns:
        counts = df['grants'].apply(len)
        ranked = df.assign(_count=counts).sort_values('_count', ascending=False)
        return ranked.head(limit)['name'].tolist()
    if 'keyword_count' in df.columns:
        ranked = df.sort_values('keyword_count', ascending=False)
        return ranked.head(limit)[entity_col].tolist()
    if 'grants' in df.columns:
        ranked = df.assign(_count=df['grants'].apply(len)).sort_values('_count', ascending=False)
        return ranked.head(limit)[entity_col].tolist()
    return df.head(limit)[entity_col].tolist()


def _summarise_grants(grants_df: pd.DataFrame) -> Tuple[int, float, Tuple[int, int]]:
    if grants_df.empty:
        return 0, 0.0, (0, 0)
    total = len(grants_df)
    funding = float(grants_df['funding_amount'].fillna(0).sum()) if 'funding_amount' in grants_df.columns else 0.0
    if 'start_year' in grants_df.columns and not grants_df['start_year'].dropna().empty:
        years = grants_df['start_year'].dropna().astype(int)
        return total, funding, (int(years.min()), int(years.max()))
    return total, funding, (0, 0)


def _format_currency(value: float) -> str:
    if value <= 0:
        return "N/A"
    return f"${value:,.0f}"


def main() -> None:
    keywords_df, grants_df, categories_df = load_data()
    if grants_df is None or grants_df.empty:
        st.error("Grants data unavailable.")
        return

    if 'organisation_ids' not in grants_df.columns:
        st.error("Grants data missing organisation identifiers.")
        return

    exploded = grants_df.explode('organisation_ids').dropna(subset=['organisation_ids'])
    organisation_counts = (
        exploded.groupby('organisation_ids')['id']
        .nunique()
        .sort_values(ascending=False)
    )
    if organisation_counts.empty:
        st.error("No funded organisations found in the dataset.")
        return
    organisation_options = organisation_counts.index.tolist()

    st.title("Organisation Insights")
    st.caption("Explore grants, keywords, and categories associated with a funded organisation.")

    with st.sidebar:
        st.header("Organisation Focus")
        selected_org = st.selectbox(
            "Funded organisation",
            organisation_options,
            format_func=_format_org_label,
        )
        use_active_period = st.checkbox(
            "Treat grants as active through their end year",
            value=True,
            help="Counts each grant in every year between start and end dates.",
        )
        year_min, year_max = _select_year_range(grants_df)
        top_keywords = st.slider("Top keywords", min_value=5, max_value=20, value=10)
        top_categories = st.slider("Top categories", min_value=5, max_value=15, value=8)
        smooth_trends = st.checkbox(
            "Smooth trend lines",
            value=True,
            help="Apply a rolling average to yearly counts."
        )
        if smooth_trends:
            smoothing_window = st.slider(
                "Rolling window (years)",
                min_value=2,
                max_value=10,
                value=5,
                step=1,
                help="Number of years included in the rolling average."
            )
        else:
            smoothing_window = 1
    window_size = smoothing_window if smooth_trends else 1

    filtered_grants = _filter_grants(grants_df, selected_org, year_min, year_max, use_active_period)
    if filtered_grants.empty:
        st.warning("No grants match the current filters. Try adjusting the year range or active-period setting.")
        return

    grant_ids = set(filtered_grants['id'])
    keyword_df = _prepare_keyword_data(keywords_df, grant_ids)
    keyword_names = set(keyword_df['name']) if 'name' in keyword_df.columns else set()
    category_df = _prepare_category_data(categories_df, keyword_names)

    total_grants, total_funding, (range_start, range_end) = _summarise_grants(filtered_grants)
    unique_keywords = len(keyword_df) if not keyword_df.empty else 0
    unique_categories = len(category_df) if not category_df.empty else 0

    stat_cols = st.columns(4)
    stat_cols[0].metric("Organisation", _format_org_label(selected_org))
    stat_cols[1].metric("Grants", f"{total_grants:,}")
    stat_cols[2].metric("Funding", _format_currency(total_funding))
    if range_start and range_end:
        stat_cols[3].metric("Active Years", f"{range_start}-{range_end}")
    else:
        stat_cols[3].metric("Active Years", "N/A")

    secondary_cols = st.columns(2)
    secondary_cols[0].metric("Keywords", f"{unique_keywords:,}")
    secondary_cols[1].metric("Categories", f"{unique_categories:,}")

    st.markdown("---")
    st.subheader("Grant Trends")
    grants_trend = TrendsDataPreparation.from_grants_by_attribute(
        filtered_grants,
        'organisation',
        use_active_period=use_active_period,
        year_min=year_min,
        year_max=year_max,
    )
    if grants_trend.empty:
        st.info("No grant trend data available.")
    else:
        grant_config = TrendsConfig(
            entity_col='organisation',
            time_col='year',
            value_col='grant_count',
            aggregation='sum',
            use_cumulative=False,
            chart_type='area',
            max_entities=1,
            title=f"Grant volume for {_format_org_label(selected_org)}",
            x_label="Year",
            y_label="Grant count",
            height=420,
            smooth_trends=smooth_trends,
            smoothing_window=window_size,
        )
        grant_fig = TrendsVisualizer().create_plot(grants_trend, grant_config)
        st.plotly_chart(grant_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Keyword Trends")
    top_keyword_names = _top_entities(keyword_df, 'name', top_keywords)
    if not top_keyword_names:
        st.info("No keywords linked to this organisation.")
    else:
        keyword_trend = TrendsDataPreparation.from_keyword_grants(
            keyword_df,
            filtered_grants,
            selected_keywords=top_keyword_names,
            use_active_period=use_active_period,
            year_min=year_min,
            year_max=year_max,
        )
        if keyword_trend.empty:
            st.info("No keyword trend data available for this selection.")
        else:
            keyword_config = TrendsConfig(
                entity_col='keyword',
                time_col='year',
                value_col='grant_count',
                aggregation='sum',
                use_cumulative=False,
                chart_type='line',
                max_entities=len(top_keyword_names),
                title=f"Top keywords in {_format_org_label(selected_org)} grants",
                x_label="Year",
                y_label="Grant count",
                height=420,
                smooth_trends=smooth_trends,
                smoothing_window=window_size,
            )
            keyword_fig = TrendsVisualizer().create_plot(keyword_trend, keyword_config)
            st.plotly_chart(keyword_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Category Trends")
    if category_df.empty:
        st.info("No categories linked to this organisation.")
        return
    category_df = category_df.assign(
        keyword_count=category_df['keywords'].apply(
            lambda items: sum(1 for item in items if item in keyword_names) if isinstance(items, list) else 0
        )
    )
    top_category_names = _top_entities(category_df, 'name', top_categories)
    if not top_category_names:
        st.info("No category trend data available for this selection.")
        return
    category_trend = TrendsDataPreparation.from_category_grants(
        category_df,
        filtered_grants,
        selected_categories=top_category_names,
        use_active_period=use_active_period,
        year_min=year_min,
        year_max=year_max,
    )
    if category_trend.empty:
        st.info("No category trend data available for this selection.")
        return
    category_config = TrendsConfig(
        entity_col='category',
        time_col='year',
        value_col='grant_count',
        ranking_col='total_funding',
        aggregation='sum',
        use_cumulative=False,
        chart_type='line',
        max_entities=len(top_category_names),
        title=f"Category trends for {_format_org_label(selected_org)}",
        x_label="Year",
        y_label="Grant count",
        height=420,
        smooth_trends=smooth_trends,
        smoothing_window=window_size,
    )
    category_fig = TrendsVisualizer().create_plot(category_trend, category_config)
    st.plotly_chart(category_fig, use_container_width=True)


if __name__ == "__main__":
    main()
