
data/active_grants.json:
	pyrla grants --status Active --json data/active_grants.json

extract-keywords: data/active_grants.json
	inspect eval modeling/keywords_extraction.py

harmonise-keywords: extract-keywords
	inspect eval modeling/keywords_harmonisation.py

assign-keywords: harmonise-keywords
	python modeling/keyword_assignment.py


.PHONY: extract-keywords harmonise-keywords assign-keywords