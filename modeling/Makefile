extract: 
	uv run inspect eval extract.py@extract

clustering:
	uv run clustering.py

review:
	uv run inspect eval review.py@review

categorise: 
	uv run inspect eval categorise.py@categorise --limit 1

merge:
	uv run inspect eval merge.py@merge --retry-on-error 3

classify: 
	uv run inspect eval classify.py@classify --retry-on-error 3

.PHONY: extract harmonise categorise merge classify