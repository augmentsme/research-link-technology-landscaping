# Todo List

1. Get the list of grants from (RLA data)[https://researchlink.ardc.edu.au/swagger-ui/index.html]
2. Run 4-5 small language models (e.g. Qwen 2.5, phi4, gemma3n) to extract keywords.
3. Feed all keywords to one large model (e.g. gemini-2.5-pro) with long context to create multiple buckets of topics (without explicitly link topics to keywords)
4. For each (topic t, keyword k) pair, use a smaller language model to identify if keyword belongs to topics k 
5. visusalsation on evolution of the topics