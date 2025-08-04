data/active_grants.json:
	pyrla grants --status Active --all --json $@

data/arc.json:
	pyrla grants --funder "Australian Research Council" --all --json $@

data/fields.json:
	pyalex fields --all --json-file $@

data/subfields.json:
	pyalex subfields --all --json-file $@

data/topics.json:
	pyalex topics --all --json-file $@

data/domains.json:
	pyalex domains --all --json-file $@


inference:
	inspect eval modeling/topic_classification.py --limit 1000 --model openai/o4-mini -T classification_type=fields -T data_path=/fred/oz318/luhanc/research-link-technology-landscaping/data/active_grants.json


.PHONY: inference