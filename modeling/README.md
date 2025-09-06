1. Run the following commands to prepare the python virtual environment 
```
cd modeling/
uv sync
```

2. Point the root directory to the location of the modeling directory. For example, in my `research-link-technology-landscaping/modeling/.env` path i define the following

```
ROOT_DIR="/Users/luhancheng/Desktop/research-link-technology-landscaping/modeling"
```

3. Now copy both `data/` and `results/` directory to the `modeling` directory. I have uploaded both `data/` and `results/` to the shared onedrive folder. 

```
(base) luhancheng at macmini in ~/Library/CloudStorage/OneDrive-SwinburneUniversity/Amir Aryani's files - 2025-eResearch-Luhan   22:00
> t
[ 160]  .
├── [ 416]  data
│   ├── [836K]  anzsrc_codes_merged.json
│   ├── [503K]  anzsrc2020_for.xlsx
│   ├── [322K]  anzsrc2020_seo.xlsx
│   ├── [495K]  for_codes_cleaned.json
│   ├── [527K]  for_codes_structured.json
│   ├── [ 50M]  grants_cleaned.jsonl
│   ├── [ 95M]  grants_enriched.json
│   ├── [ 58M]  grants_raw.json
│   ├── [ 51M]  grants.jsonl
│   └── [297K]  seo_codes_cleaned.json
└── [ 160]  results
    ├── [ 224]  category
    │   ├── [ 97M]  0.jsonl
    │   ├── [977K]  1.jsonl
    │   ├── [ 85K]  2.jsonl
    │   ├── [5.7K]  3.jsonl
    └── [ 128]  keywords
        ├── [ 97M]  extracted_keywords.jsonl
        └── [ 97M]  keywords.jsonl
```

4. run `streamlit run app.py`, this should launch the web application