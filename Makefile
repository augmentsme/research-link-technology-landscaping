clean: clean-classify clean-refine
clean-classify:
	rm -f logs/*classify*.eval
	rm -rf logs/results/classify
	rm -f logs/results/all_grants_classified.json
clean-refine:
	rm -f logs/*refine*.eval
	rm -f logs/results/*refine*.json
	rm -rf logs/results/refine


.PHONY: clean-refine clean-classify clean
