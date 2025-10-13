
CATEGORY_DIR := /home/lcheng/oz318/research-link-technology-landscaping/modeling/results/categories
uv run cli.py embed $CATEGORY_DIR/1/output.jsonl $CATEGORY_DIR/1/embeddings.npy
uv run cli.py cluster $CATEGORY_DIR/1/output.jsonl $CATEGORY_DIR/1/embeddings.npy $CATEGORY_DIR/1/categories_clusters.json
uv run cli.py merge $CATEGORY_DIR/1/categories_clusters.json $CATEGORY_DIR/2