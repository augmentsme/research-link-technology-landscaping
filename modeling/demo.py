%load_ext autoreload
%autoreload 2
import json
import pandas as pd
from inspect_ai.analysis import messages_df, evals_df, samples_df
from tasks import load_extracted_keywords, load_harmonised_keywords, load_grants_data, LOGS_DIR, DATA_DIR, GRANTS_FILE

def get_eval_ids(evals):
	"""Return harmonise and extract eval IDs from evals dataframe."""
	harmonise_id = evals[evals.task_name == "harmonise"].eval_id.item()
	extract_id = evals[evals.task_name == "extract"].eval_id.item()
	return harmonise_id, extract_id

def get_messages_and_samples(logs_dir):
	"""Load and index messages and samples dataframes."""
	messages = messages_df(logs_dir).set_index('message_id')
	samples = samples_df(logs_dir).set_index('sample_id')
	return messages, samples

def process_extracted_keywords(messages, samples, extract_id):
	"""Return extracted keywords dataframe joined with sample metadata."""
	extracted_samples = samples[samples.eval_id == extract_id]
	extracted_messages = messages[(messages.eval_id == extract_id) & (messages.role == "assistant")]
	extracted_keywords_df = pd.json_normalize(extracted_messages.content.map(json.loads)).set_index(extracted_messages.sample_id)
	extracted_keywords_df = extracted_keywords_df.join(extracted_samples.metadata_grant_id)
	return extracted_keywords_df

def process_harmonised_keywords(messages, harmonise_id, logs_dir):
	"""Return harmonised keywords dict from messages with string-based mappings."""
	harmonised_messages = messages[(messages.eval_id == harmonise_id) & (messages.role == "assistant")]
	harmonised_keywords_raw = harmonised_messages.content.item()
	print("Raw harmonised keywords:", harmonised_keywords_raw)
	
	harmonised_result = json.loads(harmonised_keywords_raw)
	
	# Convert index-based mappings to string-based mappings
	if "keyword_mappings" in harmonised_result:
		# Get the original keywords in the same order they were indexed
		all_keywords = load_extracted_keywords(logs_dir)
		flat_keywords = [kw for sublist in all_keywords.values() for kw in sublist]
		unique_keywords = sorted(set(flat_keywords))
		
		# Convert index mappings to string mappings
		index_mappings = harmonised_result["keyword_mappings"]
		
		string_mappings = {}
		# Handle new format: [{"original_index": [0, 1, 5], "harmonised": "machine learning"}, ...]
		if isinstance(index_mappings, list) and len(index_mappings) > 0:
			first_mapping = index_mappings[0]
			if isinstance(first_mapping, dict) and "harmonised" in first_mapping:
				# New format with direct harmonised keywords
				for mapping in index_mappings:
					original_indices = mapping["original_index"]
					harmonised_keyword = mapping["harmonised"]
					for original_idx in original_indices:
						if 0 <= original_idx < len(unique_keywords):
							original_keyword = unique_keywords[original_idx]
							string_mappings[original_keyword] = harmonised_keyword
				print(f"Converted {len(index_mappings)} new format mappings to string mappings")
			elif isinstance(first_mapping, dict) and "harmonised_index" in first_mapping:
				# Old format with harmonised_index
				harmonised_keywords_list = harmonised_result.get("harmonised_keywords", [])
				for mapping in index_mappings:
					original_idx = mapping["original_index"]
					harmonised_idx = mapping["harmonised_index"]
					if 0 <= original_idx < len(unique_keywords) and 0 <= harmonised_idx < len(harmonised_keywords_list):
						original_keyword = unique_keywords[original_idx]
						harmonised_keyword = harmonised_keywords_list[harmonised_idx]
						string_mappings[original_keyword] = harmonised_keyword
				print(f"Converted {len(index_mappings)} old list mappings to string mappings")
		elif isinstance(index_mappings, dict) and isinstance(list(index_mappings.keys())[0], str):
			# Even older dict format: {"0": 0, "1": 2, ...}
			harmonised_keywords_list = harmonised_result.get("harmonised_keywords", [])
			for original_idx_str, harmonised_idx in index_mappings.items():
				original_idx = int(original_idx_str)
				if 0 <= original_idx < len(unique_keywords) and 0 <= harmonised_idx < len(harmonised_keywords_list):
					original_keyword = unique_keywords[original_idx]
					harmonised_keyword = harmonised_keywords_list[harmonised_idx]
					string_mappings[original_keyword] = harmonised_keyword
			print(f"Converted {len(index_mappings)} dict mappings to string mappings")
		else:
			# Already string-based mappings
			string_mappings = index_mappings
			print("Using existing string-based mappings")
		
		harmonised_result["keyword_mappings"] = string_mappings
	
	return harmonised_result

def join_with_grants(extracted_keywords_df, grants_file):
	"""Join extracted keywords dataframe with grants dataframe."""
	grants_df = load_grants_data(grants_file, as_dataframe=True)
	results = extracted_keywords_df.join(grants_df, on='metadata_grant_id', how='left')
	return results


evals = evals_df(LOGS_DIR)
harmonise_id, extract_id = get_eval_ids(evals)

messages, samples = get_messages_and_samples(LOGS_DIR)

# harmonised_messages = messages[(messages.eval_id == harmonise_id) & (messages.role == "assistant")]

extracted_keywords_df = process_extracted_keywords(messages, samples, extract_id)
harmonised_keywords = process_harmonised_keywords(messages, harmonise_id, LOGS_DIR)

results = join_with_grants(extracted_keywords_df, GRANTS_FILE)

harmonised_keywords
harmonised_keywords['keyword_mappings'].keys()



results.keywords.map(lambda x: [harmonised_keywords['keyword_mappings'].get(kw, kw) for kw in x])


# print(results.loc[:, ['keywords', 'title', 'grant_summary']].head().iloc[0].tolist())
