import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


import utils
import config

# def preprocess_grants(grants_df):
#     df = grants_df[~grants_df.title.isna()] 
#     df = df[~df.title.str.contains("equipment grant", case=False) & ~df.title.str.contains("travel grant", case=False)]
#     return df

def deduplicate_keywords(input_path):
    """
    Deduplicate keywords by normalizing terms and selecting best variants.
    
    Returns:
        list: Deduplicated keywords with grants lists
    """
    keywords_dict = {}  # normalized_term -> {keyword_data, variants}
    all_keywords = utils.load_jsonl_file(input_path)  # Ensure file is loaded
    
    for keyword in all_keywords:
        normalized_term = utils.normalize(keyword["name"])

        if normalized_term not in keywords_dict:
            keywords_dict[normalized_term] = {
                "variants": [],
                "grants": set()
            }
        
        keywords_dict[normalized_term]["variants"].append(keyword)
        keywords_dict[normalized_term]["grants"].add(keyword["grant_id"])
    
    final_keywords = []
    
    # For each normalized term, choose the variant with the longest description
    for normalized_term, data in keywords_dict.items():
        # Find variant with longest description
        best_variant = max(data["variants"], key=lambda k: len(k["description"]))
        
        # Create final keyword with all grants
        grants_list = list(data["grants"])
        final_keyword = {
            "name": best_variant["name"],
            "type": best_variant["type"],
            "description": best_variant["description"],
            "grants": grants_list
        }
        
        final_keywords.append(final_keyword)
    
    return final_keywords

def postprocess_keywords(input_path=config.Keywords.extracted_keywords_path, output_path=config.Keywords.keywords_path):
    """
    Postprocess keywords by deduplicating and adding organization information.
    """

    deduplicated = pd.DataFrame(deduplicate_keywords(input_path))
    grants_df = config.Grants.load(as_dataframe=True)

    if not deduplicated.empty:
        if "organisation_ids" not in grants_df.columns:
            grants_df = grants_df.assign(organisation_ids=[[]] * len(grants_df))
        if "countries" not in grants_df.columns:
            grants_df = grants_df.assign(countries=[[]] * len(grants_df))
        if "country_codes" not in grants_df.columns:
            grants_df = grants_df.assign(country_codes=[[]] * len(grants_df))
        if "researcher_ids" not in grants_df.columns:
            grants_df = grants_df.assign(researcher_ids=[[]] * len(grants_df))

        grant_lookup = grants_df.set_index("id")[
            ["organisation_ids", "countries", "country_codes", "researcher_ids"]
        ]

        def _collect_lists(series: pd.Series) -> List[str]:
            values: List[str] = []
            for item in series.dropna():
                if isinstance(item, list):
                    values.extend([v for v in item if pd.notna(v)])
                elif pd.notna(item):
                    values.append(item)
            return sorted(set(values))

        def enrich_keyword(grant_ids: List[str]) -> Dict[str, List[str]]:
            subset = grant_lookup.reindex(grant_ids).dropna(how="all")
            if subset.empty:
                return {
                    "organisation_ids": [],
                    "countries": [],
                    "country_codes": [],
                    "researcher_ids": [],
                }
            return {
                "organisation_ids": _collect_lists(subset["organisation_ids"]),
                "countries": _collect_lists(subset["countries"]),
                "country_codes": _collect_lists(subset["country_codes"]),
                "researcher_ids": _collect_lists(subset["researcher_ids"]),
            }

        enrichment = deduplicated["grants"].apply(enrich_keyword).apply(pd.Series)
        final = pd.concat([deduplicated, enrichment], axis=1)
    else:
        final = deduplicated.assign(
            organisation_ids=[], countries=[], country_codes=[], researcher_ids=[]
        )

    final = final[final["country_codes"].map(lambda x: "AU" in x if isinstance(x, list) else True)]
    utils.save_jsonl_file(final.to_dict(orient="records"), output_path)


def postprocess_category(input_path=config.Categories.last_merged_path(), output_path=config.Categories.CATEGORIY_PATH):
    cats = utils.load_jsonl_file(input_path, as_dataframe=True)
    cats = cats.drop(cats[cats.keywords.map(len) == 0].index) # drop categories with no keywords
    utils.save_jsonl_file(cats.to_dict(orient="records"), output_path)


    
