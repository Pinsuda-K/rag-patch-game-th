ENV?=. .venv/bin/activate && . ./.env

.PHONY: venv deps crawl extract normalize facets bm25 dense serve bench test smoke

venv:
	python -m venv .venv

deps: venv
	. .venv/bin/activate && pip install -U pip && \
	pip install -U fastapi uvicorn[standard] requests pydantic \
	    sentence-transformers chromadb rank-bm25 beautifulsoup4 tqdm regex orjson && \
	pip install playwright && python -m playwright install chromium

crawl:
	$(ENV) && python -m src.ingest.discover_patch_urls \
	    --start "https://rov.in.th/patch-notes" --out data/urls_all.txt --max-pages 10 --max-urls 500

extract:
	$(ENV) && python -m src.ingest.extract_patch_playwright \
	    --urls-file data/urls_all.txt --out data/corpus_raw.jsonl --concurrency 3

normalize:
	$(ENV) && python -m src.ingest.normalize_chunk \
	    --in data/corpus_raw.jsonl --out data/corpus.jsonl --aggressive-infer --qa

facets:
	$(ENV) && python -m src.ingest.normalize_chunk \
	    --in data/corpus_raw.jsonl --out data/corpus.jsonl --aggressive-infer \
	    --facet-out data/corpus_facets.jsonl --facet-min-conf 2 --qa

bm25:
	$(ENV) && python -m src.index.build_bm25 --in data/corpus.jsonl --out data/bm25_idx.json

dense:
	$(ENV) && python -m src.index.build_dense --in data/corpus.jsonl --db data/chroma_db --collection docs --model $$EMBEDDER_MODEL

serve:
	$(ENV) && uvicorn src.serve.api:app --host 0.0.0.0 --port 8000 --reload

bench:
	$(ENV) && python -m src.eval.bench_client --url http://localhost:8000/answer --n 30 --k 10

test:
	$(ENV) && pytest -q
