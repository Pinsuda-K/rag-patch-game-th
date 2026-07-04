# Retrieval defaults
export EMBEDDER_MODEL="intfloat/multilingual-e5-small"
export CHROMA_DIR="$HOME/persist/chroma_db"
export CHROMA_COLLECTION="docs-e5"
export BM25_INDEX_PATH="data/bm25_idx.json"

# Perf knobs
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
