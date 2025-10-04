extract: 
	uv run inspect eval extract.py@extract
clean-extract: 
	rm -f logs/*extract*.eval
	rm -f results/keywords/extracted_keywords.jsonl results/keywords/keywords.jsonl



# clean-merge:
# 	rm -f logs/*merge*.eval
# 	rm -f results/merge/*.jsonl

# clean-categorise:
# 	rm -f logs/*categorise*.eval
# 	rm -f results/category/*.jsonl




# generate-keywords-dataset: 
# 	uv run semantic_clustering.py pipeline results/keywords/keywords.jsonl results/keywords_batches

# generate-category-dataset:
# 	uv run semantic_clustering.py pipeline results/category/category/category_extracted.jsonl results/categories_batches

generate-keywords:
	uv run semantic_clustering.py pipeline results/keywords/keywords.jsonl results/keywords-batches

generate-%:
	uv run semantic_clustering.py pipeline results/$*/output.jsonl results/$*-batches

categorise:
	uv run inspect eval categorise_keywords.py@categorise -T input_dir=results/keywords-batches/batches -T output_dir=results/1

merge-%:
	$(eval NEXT := $(shell echo $$(($(subst merge-,,$@) + 1))))
	uv run inspect eval merge_categories.py@merge -T input_dir=results/$*-batches/batches -T output_dir=results/$(NEXT)



.PHONY: extract clean-extract categorise merge clean-categorise clean-merge