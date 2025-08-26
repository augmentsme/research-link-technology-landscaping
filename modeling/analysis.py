from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import pandas as pd

from config import KEYWORDS_PATH, GRANTS_FILE
from utils import load_keywords, load_grants
import numpy as np
from scipy.stats import binned_statistic


def extract_grant_id_from_keyword(keyword_id: str) -> str:
    """Extract grant ID from keyword ID.

    Example: 'arc/DP0343410_0' -> 'arc/DP0343410'
    """
    if '_' in keyword_id:
        return keyword_id.rsplit('_', 1)[0]
    return keyword_id


def create_time_bins(start_year: int, end_year: int, bin_size: int) -> List[Tuple[int, int]]:
    """Create time bins for grouping data.

    Returns list of (start, end) inclusive tuples.
    """
    bins: List[Tuple[int, int]] = []
    current_year = start_year
    while current_year <= end_year:
        bin_end = min(current_year + bin_size - 1, end_year)
        bins.append((current_year, bin_end))
        current_year += bin_size
    return bins


def assign_to_time_bin(year: float, time_bins: List[Tuple[int, int]]) -> Optional[str]:
    """Assign a year to the appropriate time bin label."""
    if pd.isna(year):
        return None
    year = int(year)
    for bin_start, bin_end in time_bins:
        if bin_start <= year <= bin_end:
            if bin_start == bin_end:
                return str(bin_start)
            return f"{bin_start}-{bin_end}"
    return None


def analyze_keyword_trends(
    time_range: Optional[Tuple[int, int]] = None,
    bin_size: int = 5,
    keyword_types: Optional[List[str]] = None,
    top_n_terms: int = 20,
    min_frequency: int = 2,
    keywords_path: Optional[Path] = None,
    grants_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Analyze keyword trends over time using grant end dates.

    Returns a dict with 'trends_data' (DataFrame), 'summary_stats', 'time_bins'
    and 'parameters'.
    """
    if keywords_path is None:
        keywords_path = KEYWORDS_PATH
    if grants_path is None:
        grants_path = GRANTS_FILE

    keywords_data = load_keywords(keywords_path)
    grants_data = load_grants(grants_path)

    # Create grant lookup dictionary for quick access
    grants_lookup = {grant['id']: grant for grant in grants_data}

    # Link keywords to grants and extract temporal information
    linked_data = []

    for keyword in keywords_data:
        # Handle the new keyword structure where keywords have a 'grants' list
        # instead of a single 'id' field that needs parsing
        if 'grants' in keyword:
            grant_ids = keyword['grants']
        elif 'id' in keyword:
            # Fallback for old structure
            grant_ids = [extract_grant_id_from_keyword(keyword['id'])]
        else:
            continue  # Skip keywords without grant information

        # Process each grant ID associated with this keyword
        for grant_id in grant_ids:
            if grant_id in grants_lookup:
                grant = grants_lookup[grant_id]

                # Skip if keyword type filter is specified and doesn't match
                if keyword_types and keyword['type'] not in keyword_types:
                    continue

                linked_data.append({
                    'keyword_id': f"{grant_id}_{keyword['term']}",  # Create a unique ID
                    'term': keyword['term'],
                    'type': keyword['type'],
                    'description': keyword['description'],
                    'grant_id': grant_id,
                    'grant_title': grant['title'],
                    'end_year': grant['end_year'],
                    'start_year': grant['start_year'],
                    'funding_amount': grant['funding_amount'],
                    'funder': grant['funder']
                })

    if not linked_data:
        raise ValueError("No keywords could be linked to grants")

    # Convert to DataFrame
    df = pd.DataFrame(linked_data)

    # Filter out records with missing end_year
    df = df[df['end_year'].notna()]

    # Determine time range
    if time_range is None:
        start_year = int(df['end_year'].min())
        end_year = int(df['end_year'].max())
    else:
        start_year, end_year = time_range
        df = df[(df['end_year'] >= start_year) & (df['end_year'] <= end_year)]

    # Create bin edges and assign records to bins using scipy.stats.binned_statistic
    # Build bin edges so each bin spans `bin_size` years. Use edges that mark
    # the start of each bin; we make bin end inclusive by subtracting 1 from
    # the next edge when creating (start, end) tuples.
    edges = np.arange(start_year, end_year + bin_size, bin_size)
    if edges[-1] <= end_year:
        edges = np.append(edges, end_year + 1)

    # Use binned_statistic to compute the binnumber for each record
    _, bin_edges, binnumber = binned_statistic(
        df['end_year'].values,
        df['end_year'].values,
        statistic='count',
        bins=edges
    )

    # Convert bin_edges into inclusive (start, end) tuples
    nbins = len(bin_edges) - 1
    time_bins = [
        (int(bin_edges[i]), int(bin_edges[i+1]) - 1 if i < nbins - 1 else int(bin_edges[i+1]) - 1)
        for i in range(nbins)
    ]

    # Human-readable labels
    time_bin_labels = [
        f"{start}-{end}" if start != end else str(start)
        for start, end in time_bins
    ]

    # binnumber returned by binned_statistic is 1-based; 0 means out of range
    df['time_bin_idx'] = binnumber
    df = df[df['time_bin_idx'] > 0]
    df['time_bin'] = df['time_bin_idx'].apply(lambda i: time_bin_labels[int(i) - 1])

    # Count keyword frequencies to filter by minimum frequency
    term_counts = Counter(df['term'])
    frequent_terms = {term for term, count in term_counts.items() if count >= min_frequency}
    df = df[df['term'].isin(frequent_terms)]

    # Get top N terms by overall frequency
    top_terms = [term for term, _ in term_counts.most_common(top_n_terms) if term in frequent_terms]
    df_top = df[df['term'].isin(top_terms)]

    # Create trends data
    trends_data = []

    # Ensure we have data for all time bins and all top terms
    time_bin_labels = [f"{start}-{end}" if start != end else str(start) for start, end in time_bins]

    for time_bin_label in time_bin_labels:
        bin_data = df_top[df_top['time_bin'] == time_bin_label]

        for term in top_terms:
            count = 0
            most_common_type = 'unknown'
            funding_stats = {'total_funding': 0, 'mean_funding': 0, 'median_funding': 0}

            if not bin_data.empty:
                term_data = bin_data[bin_data['term'] == term]
                count = len(term_data)

                if not term_data.empty:
                    most_common_type = term_data['type'].mode().iloc[0] if len(term_data) > 0 else 'unknown'
                    funding_stats = {
                        'total_funding': term_data['funding_amount'].sum(),
                        'mean_funding': term_data['funding_amount'].mean(),
                        'median_funding': term_data['funding_amount'].median()
                    }

            trends_data.append({
                'time_bin': time_bin_label,
                'term': term,
                'count': count,
                'frequency': count / len(bin_data) if len(bin_data) > 0 else 0,
                'type': most_common_type,
                'total_funding': funding_stats['total_funding'],
                'mean_funding': funding_stats['mean_funding'],
                'median_funding': funding_stats['median_funding'],
                'total_grants_in_bin': len(bin_data)
            })

    trends_df = pd.DataFrame(trends_data)

    # Calculate summary statistics
    summary_stats = {
        'total_keywords': len(df),
        'unique_terms': len(df['term'].unique()),
        'unique_grants': len(df['grant_id'].unique()),
        'time_span': f"{start_year}-{end_year}",
        'total_time_bins': len(time_bins),
        'keyword_types': df['type'].value_counts().to_dict(),
        'top_terms': top_terms,
        'avg_keywords_per_bin': trends_df.groupby('time_bin')['count'].sum().mean() if not trends_df.empty else 0
    }

    return {
        'trends_data': trends_df,
        'summary_stats': summary_stats,
        'time_bins': time_bins,
        'parameters': {
            'time_range': time_range,
            'bin_size': bin_size,
            'keyword_types': keyword_types,
            'top_n_terms': top_n_terms,
            'min_frequency': min_frequency
        }
    }
