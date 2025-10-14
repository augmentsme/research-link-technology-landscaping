"""Precomputed datasets for the Streamlit app.

This module materialises frequently used joins so interactive pages only
need lightweight lookups when rendering visualisations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

import config


@dataclass(frozen=True)
class _EmptyFrames:
    grant_years: pd.DataFrame
    keyword_links: pd.DataFrame
    category_links: pd.DataFrame


def _empty_frames() -> _EmptyFrames:
    grant_years = pd.DataFrame(
        columns=["grant_id", "year", "funding_credit", "funder", "source", "primary_subject"]
    )
    keyword_links = pd.DataFrame(columns=["keyword", "grant_id"])
    category_links = pd.DataFrame(columns=["category", "keyword", "grant_id", "source"])
    return _EmptyFrames(grant_years, keyword_links, category_links)


@st.cache_resource(show_spinner=False)
def load_base_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw keywords, grants, and categories tables."""
    keywords = config.Keywords.load()
    grants = config.Grants.load()
    categories = config.Categories.load()

    if keywords is None:
        keywords = pd.DataFrame()
    if grants is None:
        grants = pd.DataFrame()
    if categories is None:
        categories = pd.DataFrame()
    return keywords.copy(), grants.copy(), categories.copy()


@st.cache_data(show_spinner=False)
def _build_grant_years_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand grants into per-year records for active and start-year views."""
    _, grants_df, _ = load_base_tables()
    if grants_df.empty or "start_year" not in grants_df.columns or "id" not in grants_df.columns:
        empty = _empty_frames().grant_years
        return empty.copy(), empty.copy()

    working = grants_df.dropna(subset=["start_year"]).copy()
    if working.empty:
        empty = _empty_frames().grant_years
        return empty.copy(), empty.copy()

    start_year = working["start_year"].fillna(0).astype(int)
    if "end_year" in working.columns:
        end_year = working["end_year"].fillna(start_year).astype(int)
    else:
        end_year = start_year.copy()
    end_year = end_year.where(end_year >= start_year, start_year)

    funding = working["funding_amount"].fillna(0) if "funding_amount" in working.columns else pd.Series(0, index=working.index, dtype=float)

    counts = (end_year - start_year + 1).astype(int)
    counts[counts < 1] = 1

    repeated = working.loc[working.index.repeat(counts)].reset_index(drop=True)
    if repeated.empty:
        active = _empty_frames().grant_years.copy()
    else:
        year_sequence: list[int] = []
        first_flags: list[bool] = []
        start_vals = start_year.to_numpy()
        end_vals = end_year.to_numpy()
        count_vals = counts.to_numpy()
        for s, e, length in zip(start_vals, end_vals, count_vals):
            years = list(range(s, e + 1))
            year_sequence.extend(years)
            first_flags.extend([True] + [False] * (length - 1))
        funding_repeated = np.repeat(funding.to_numpy(dtype=float), count_vals)
        first_mask = np.array(first_flags, dtype=bool)
        funding_credit = funding_repeated * first_mask
        repeated["year"] = pd.Series(year_sequence, index=repeated.index)
        repeated["funding_credit"] = funding_credit
        active = _project_grant_years(repeated)

    start_only = working.copy()
    start_only["year"] = start_year
    start_only["funding_credit"] = funding.to_numpy(dtype=float)
    start_frame = _project_grant_years(start_only)

    return start_frame, active


def _project_grant_years(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed downstream and standardise names."""
    keep_cols = [col for col in ["id", "year", "funding_credit", "funder", "source", "primary_subject"] if col in frame.columns]
    projected = frame[keep_cols].copy()
    projected = projected.rename(columns={"id": "grant_id"})
    for column in ["funder", "source", "primary_subject"]:
        if column not in projected.columns:
            projected[column] = None
    if "funding_credit" not in projected.columns:
        projected["funding_credit"] = 0.0
    return projected.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_grant_years_table(active: bool) -> pd.DataFrame:
    """Return pre-expanded grant records by year."""
    start_frame, active_frame = _build_grant_years_tables()
    return active_frame.copy() if active else start_frame.copy()


@st.cache_data(show_spinner=False)
def get_keyword_grant_links() -> pd.DataFrame:
    """Return keyword to grant mappings exploded to one row per pair."""
    keywords_df, _, _ = load_base_tables()
    if keywords_df.empty or "grants" not in keywords_df.columns:
        return _empty_frames().keyword_links.copy()
    column = "name" if "name" in keywords_df.columns else keywords_df.index.name or "name"
    base = keywords_df[[column, "grants"]].reset_index(drop=True)
    exploded = base.explode("grants").dropna(subset=["grants"])
    if exploded.empty:
        return _empty_frames().keyword_links.copy()
    exploded = exploded.rename(columns={column: "keyword", "grants": "grant_id"})
    return exploded.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_category_grant_links() -> pd.DataFrame:
    """Return category to grant mappings using keyword membership."""
    _, _, categories_df = load_base_tables()
    if categories_df.empty or "keywords" not in categories_df.columns:
        return _empty_frames().category_links.copy()
    keyword_links = get_keyword_grant_links()
    if keyword_links.empty:
        return _empty_frames().category_links.copy()
    categories = categories_df[["name", "keywords"]].reset_index(drop=True)
    exploded = categories.explode("keywords").dropna(subset=["keywords"])
    if exploded.empty:
        return _empty_frames().category_links.copy()
    exploded = exploded.rename(columns={"name": "category", "keywords": "keyword"})
    merged = exploded.merge(keyword_links, on="keyword", how="inner")
    _, grants_df, _ = load_base_tables()
    if grants_df.empty or "id" not in grants_df.columns:
        merged["source"] = None
        return merged.reset_index(drop=True)
    sources = grants_df[["id", "source"]].rename(columns={"id": "grant_id"})
    merged = merged.merge(sources, on="grant_id", how="left")
    return merged.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_keyword_years_table(active: bool) -> pd.DataFrame:
    """Return keyword-year records derived from grant expansions."""
    keyword_links = get_keyword_grant_links()
    if keyword_links.empty:
        return pd.DataFrame(columns=["keyword", "grant_id", "year", "funding_credit"])
    grant_years = get_grant_years_table(active)
    if grant_years.empty:
        return pd.DataFrame(columns=["keyword", "grant_id", "year", "funding_credit"])
    merged = keyword_links.merge(grant_years, on="grant_id", how="inner")
    columns = ["keyword", "grant_id", "year", "funding_credit"]
    return merged[columns].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_category_years_table(active: bool) -> pd.DataFrame:
    """Return category-year records derived from grant expansions."""
    category_links = get_category_grant_links()
    if category_links.empty:
        return pd.DataFrame(columns=["category", "grant_id", "year", "funding_credit"])
    grant_years = get_grant_years_table(active)
    if grant_years.empty:
        return pd.DataFrame(columns=["category", "grant_id", "year", "funding_credit"])
    merged = category_links.merge(grant_years, on="grant_id", how="inner")
    columns = ["category", "grant_id", "year", "funding_credit"]
    return merged[columns].reset_index(drop=True)
