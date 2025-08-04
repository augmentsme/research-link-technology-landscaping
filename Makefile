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


