import re
import pandas as pd
from pathlib import Path


import utils
import config

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

def postprocess(input_path=config.Keywords.extracted_keywords_path, output_path=config.Keywords.keywords_path):
    """
    Postprocess keywords by deduplicating and adding organization information.
    """

    deduplicated = pd.DataFrame(deduplicate_keywords(input_path))
    links = config.Grants.load_links()
    
    joined = pd.DataFrame(deduplicated).explode("grants").merge(links, left_on="grants", right_on="grant_key", how="left")
    joined = joined[~(joined['organisation_country'].isna())]
    # joined = joined[~(joined['organisation_country'].isna() & joined['organisation_name'].isna() & joined['researcher_orcid'].isna())]
    linked = joined.groupby("name").agg({"organisation_name": list, "organisation_country": list, "researcher_orcid": list})
    

    linked = linked.reset_index().astype("str")
    final = deduplicated.merge(linked, left_on="name", right_on="name", how="left")
    # final = final[~final['organisation_country'].isna()]
    # df_with_valid_country_records = final[~final["organisation_country"].isna()]
    final = final[final['organisation_country'].map(lambda x: "AU" in x if pd.notna(x) else True)]
    utils.save_jsonl_file(final.to_dict(orient="records"), output_path)

if __name__ == "__main__":
    postprocess()