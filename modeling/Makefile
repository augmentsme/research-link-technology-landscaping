extract: 
	uv run inspect eval extract.py@extract

abstract: 
	uv run inspect eval abstract.py@abstract --limit 1

classify:
	uv run inspect eval abstract.py@classify --limit 100 --retry-on-error

clean: 
	rm -f logs/*.eval
	
.PHONY: extract harmonise categorise merge classify clean