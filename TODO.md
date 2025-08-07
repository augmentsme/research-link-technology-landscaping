# Todo List

- [x] Get the list of grants from (RLA data)[https://researchlink.ardc.edu.au/swagger-ui/index.html]
- [] Run 4-5 small language models (e.g. Qwen 2.5, phi4, gemma3n) to extract keywords.
- [] Feed all keywords to one large model (e.g. gemini-2.5-pro) with long context to create multiple buckets of topics (without explicitly link topics to keywords)
- [] For each (topic t, keyword k) pair, use a smaller language model to identify if keyword belongs to topics k 
- [] visusalsation on evolution of the topics

note that the logic of keywords_harmonisation.py now correctly uses 1 big query that includes all keywords from the previous step (keywords extraction), allowing the LLM to have a holistic view of all available options. The output provides harmonised keywords and complete mappings from original keywords to their harmonised versions for consistent terminology.