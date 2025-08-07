# Todo List

- [x] Get the list of grants from (RLA data)[https://researchlink.ardc.edu.au/swagger-ui/index.html]
- [] Run 4-5 small language models (e.g. Qwen 2.5, phi4, gemma3n) to extract keywords.
- [] Feed all keywords to one large model (e.g. gemini-2.5-pro) with long context to create multiple buckets of topics (without explicitly link topics to keywords)
- [] For each (topic t, keyword k) pair, use a smaller language model to identify if keyword belongs to topics k 
- [] visusalsation on evolution of the topics

note that the logic of #file:keywords_clustering.py does not seem to be correct, i want 1 big query that include all (not for each sample) the keywords we have identified in the previous step (i.e. keywords extraction), so the LLM can have a holistic view of all available options. And in addition to the set of topics (i.e. harmonised keywords) , the output should also provide the info on which keywords correspond to which topic, so it can be used in topic_classification to create linkage between grants and topics