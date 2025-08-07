# Makefile for Research Link Technology Landscaping

# Main workflow targets
workflow: extract-keywords cluster-keywords assign-topics

# Alternative workflow using new CLI
workflow-cli:
	python modeling/cli.py run-all

# Individual steps using CLI
extract-keywords-cli:
	python modeling/cli.py extract

cluster-keywords-cli:
	python modeling/cli.py cluster

assign-topics-cli:
	python modeling/cli.py classify

# Check workflow status
status:
	python modeling/cli.py status

# Data collection
data/active_grants.json:
	pyrla grants --status Active --json data/active_grants.json

# Legacy individual steps (deprecated - use CLI instead)
extract-keywords: data/active_grants.json
	inspect eval modeling/keywords_extraction.py --model openai/gpt-4o-mini

cluster-keywords: extract-keywords
	inspect eval modeling/keywords_clustering.py --model openai/gpt-4o-mini

assign-topics: cluster-keywords
	python modeling/topic_classification.py

# Web dashboard
web-app:
	streamlit run web/app.py

# Utilities
clean: 
	rm -rf logs/*
	rm -rf data/keyword_clusters.json
	rm -rf data/grant_topic_assignments.json

clean-all: clean
	rm -rf data/active_grants.json

# Help
help:
	@echo "Available targets:"
	@echo "  workflow-cli     - Run complete workflow using CLI"
	@echo "  extract-keywords-cli - Extract keywords using CLI"
	@echo "  cluster-keywords-cli - Cluster keywords using CLI"
	@echo "  assign-topics-cli    - Classify topics using CLI"
	@echo "  status           - Check workflow status"
	@echo "  web-app          - Launch Streamlit dashboard"
	@echo "  clean            - Clean generated files"
	@echo "  clean-all        - Clean all data files"

.PHONY: workflow workflow-cli extract-keywords-cli cluster-keywords-cli assign-topics-cli status web-app clean clean-all help
	rm -rf data/keywords.json
	rm -rf data/clusters.json
	rm -rf data/topics.json

.PHONY: clean workflow web-app