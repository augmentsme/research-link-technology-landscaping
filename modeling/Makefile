extract: 
	uv run inspect eval extract.py@extract

categorise: 
	uv run inspect eval categorise.py@categorise

refine:
	uv run inspect eval refine.py@refine

classify: 
	uv run inspect eval classify.py@classify -T category_path=/Users/luhancheng/Desktop/research-link-technology-landscaping/modeling/logs/results/refined_categories.json --limit 50


clean-categorise: clean-refine
	rm -f logs/*categorise*.eval 
	rm -f logs/results/categorise.json

clean-refine: clean-classify
	rm -f logs/*refine*.eval
	rm -f logs/results/refined_categories.json

clean-classify:
	rm -f logs/*classify*.eval

.PHONY: extract categorise refine classify clean-categorise clean-refine clean-classify
