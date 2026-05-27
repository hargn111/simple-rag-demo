# Simple Rag Demo

Demonstrate vectorization and semantic search of vector DB (ChromaDB) to enrich LLM output with data injected into context.

## Installation and Env setup

```bash
# Clone the repo
git clone https://github.com/hargn111/simple-rag-demo.git
cd simple-rag-demo

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```


## Set up secrets

Fill out your URL, model and API key in `.env`.


## Ingest docs

Place your documents in `documents` and run:

```bash
source .venv/bin/activate
python3 ingest_docs.py
``` 


## Query Docs

Run

```bash
source .venv/bin/activate
python3 query_rag.py
``` 

and ask a question.


## Analyze Vector Store / Retrieval Precision

Add your test queries to `evaluate_rag.py` then run:

```bash
source .venv/bin/activate
python3 evaluate_rag.py
```


### For local LLM (optional):

```bash
# Option A: llama.cpp (recommended for CPU/GPU)
brew install llama.cpp  # macOS
# Or download from: https://github.com/ggerganov/llama.cpp/releases

# Option B: vLLM (for GPU-only, OpenAI-compatible API)
pip install vllm
```

---

### Download and Prepare LLM (Local Option)

```bash
# Create models directory
mkdir -p ./models

# Download Llama 3.3 8B (4-bit quantized) - ~5GB
# From Hugging Face (example)
# Visit: https://huggingface.co/models?search=llama-3.3-8b-GGUF
# Download a Q4_K_M quantized version (good quality/size balance)

# Example using wget (adjust URL to actual model):
wget -P ./models https://huggingface.co/.../llama-3.3-8b-instruct-q4_k_m.gguf
```

### Start llama.cpp server (OpenAI-compatible):

```bash
llama-server \
  --model ./models/llama-3.3-8b-instruct-q4_k_m.gguf \
  --ctx-size 4096 \
  --port 8080 \
  --n-gpu-layers 33  # Adjust based on your GPU; 0 for CPU-only
```

This creates an OpenAI-compatible endpoint at http://localhost:8080


### Not fully implemented

- Epub to PDF converter (`epub_to_pdf.py`) - its there but not implemented in a meaningful way. need to install `requirements-epub.txt` in your venv to use.
- Sphinx/web html doc ingestion (its in the ingestion script and functional, just no proper CLI arg. uncomment it and comment out the load_and_process_documents call to use)
