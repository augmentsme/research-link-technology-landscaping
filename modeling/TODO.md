Complete the following tasks for main logic files `extract.py`, `categorise_keywords.py`, `merge_categories.py`, `semantic_clustering.py`
- [x] use tenacity to implement timeout for sample requests and retry logic, set default retry attempts and timeout value in config.py 
- [x] `extract.py`: reuse the existing `finished_grants` cache when launching new runs by skipping record iteration entirely when the dataset has no unseen IDs to avoid opening the JSONL writer unnecessarily.
- [x] `categorise_keywords.py`: stream batches directly from disk with a generator so large keyword sets do not load fully in memory before dispatching requests.
- [x] `merge_categories.py`: deduplicate batch loading/formatting helpers with the categorisation script to avoid diverging schemas; both could share a small utility module for JSONL batch orchestration.
