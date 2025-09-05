extract: 
	uv run inspect eval extract.py@extract

abstract: 
	uv run inspect eval abstract.py@abstract

classify:
	uv run inspect eval abstract.py@classify --limit 10

clean: 
	rm -f logs/*.eval
	
.PHONY: extract harmonise categorise merge classify clean