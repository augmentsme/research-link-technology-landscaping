extract: 
	uv run inspect eval extract.py@extract
clean-extract: 
	rm -f logs/*extract*.eval
	rm -f results/keywords/extracted_keywords.jsonl results/keywords/keywords.jsonl

categorise:
	uv run inspect eval categorise.py@categorise
clean-categorise:
	rm -f logs/*categorise*.eval
	rm -f results/category/*.jsonl

merge:
	uv run inspect eval categorise.py@merge
clean-merge:
	rm -f logs/*merge*.eval
	rm -f results/merge/*.jsonl


.PHONY: extract clean-extract categorise clean-categorise