extract: 
	uv run inspect eval extract.py@extract

clustering:
	uv run clustering.py

review:
	uv run inspect eval review.py@review

categorise: 
	uv run inspect eval categorise.py@categorise

merge:
	uv run inspect eval merge.py@merge --limit 1

classify: 
	uv run inspect eval classify.py@classify --limit 100

clean: 
	rm -f logs/*.eval
	
.PHONY: extract harmonise categorise merge classify clean