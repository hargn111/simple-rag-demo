import os
import json  # [New] For saving and loading manifest data
import hashlib  # [New] For computing file content hashes
import sys

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredEPubLoader, DirectoryLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pathlib import Path
from bs4 import BeautifulSoup

# [New] Optionally import OpenAIEmbeddings for remote embedding support
try:
    from langchain_openai import OpenAIEmbeddings  # [New] For OpenAI embedding API
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

# Configuration
PDF_DIR = './documents'
PERSIST_DIR = './chroma_db'
MANIFEST_FILE = os.path.join(PERSIST_DIR, 'manifest.json')  # [New] Manifest file path for version tracking
USE_OPENAI_EMBEDDINGS = False  # [New] Toggle: False = use local embeddings, True = use OpenAI /v1/embeddings API
OPENAI_EMBEDDING_MODEL = 'text-embedding-ada-002'  # [New] OpenAI embedding model (e.g. Ada v2)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # [New] API key for OpenAI embeddings (required if using remote)
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE',
                            'https://api.openai.com/v1')  # [New] Base URL for OpenAI-compatible embedding endpoint
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5'  # Default local embedding model (HuggingFace)
ACCEPTED_FILE_EXTS = [
    ".pdf",
    ".md"
]


def load_manifest():
    """Load manifest JSON if it exists, else return empty dict"""
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print('Warning: Manifest unreadable. Starting fresh.')
    return {}  # If no manifest or parse error, start with empty


def save_manifest(manifest):
    """Save manifest data to JSON file"""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)


def compute_file_hash(filepath, chunk_size=8192):
    """Compute SHA-256 hash of a file's content"""
    hash_obj = hashlib.sha256()  # [New] Use SHA-256 for robust content hashing
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def load_and_process_documents(directory, prev_manifest):
    """Load and chunk PDFs, skipping those that haven't changed"""
    new_chunks = []
    updated_manifest = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith('.pdf'):
            continue  # Skip non-PDF files
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue  # Skip if path is not a file (or file was removed)
        current_mtime = os.path.getmtime(filepath)
        current_hash = None
        if filename in prev_manifest:
            prev = prev_manifest[filename]
            if prev.get('mtime') == current_mtime:
                # File unchanged since last ingestion
                updated_manifest[filename] = prev
                print(f"[Skip] {filename} unchanged (already indexed).")
                continue
            # Timestamp differs - check if content changed
            current_hash = compute_file_hash(filepath)  # [New] compute hash for change detection
            if prev.get('hash') == current_hash:
                # Content is identical, only timestamp changed
                updated_manifest[filename] = {'hash': current_hash, 'mtime': current_mtime}
                print(f"[Skip] {filename} content unchanged (timestamp updated).")
                continue
            else:
                # File content has changed - reprocess
                print(f"[Update] {filename} modified; re-indexing.")
        else:
            # New file - process it
            print(f"[New] {filename} not seen before; indexing.")
        # Load PDF and split into text chunks
        loader = PyMuPDFLoader(filepath)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        new_chunks.extend(chunks)
        # Compute hash for new or modified file (if not already computed above)
        if current_hash is None:
            current_hash = compute_file_hash(filepath)
        # Update manifest entry for this file
        updated_manifest[filename] = {'hash': current_hash, 'mtime': current_mtime}
    return new_chunks, updated_manifest


# Mode is single or elements
def load_epub(epub_path, mode="single", lazy=True):
    loader = UnstructuredEPubLoader(epub_path, mode=mode)
    docs = loader.lazy_load() if lazy else loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks


def strip_sphinx_junk(web_path):
    print(f"Cleaning junk from sphinx doc")
    for html_file in web_path.rglob("*.html"):
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "html.parser")

        # Remove unwanted elements
        for selector in [
            "nav",
            "footer",
            ".headerlink",
            ".rst-versions",
            ".sphinxsidebar",
            "#sidebar",
            ".related",  # prev/next navigation bars
            ".navigation",
            "#searchbox",
        ]:
            for el in soup.select(selector):
                el.decompose()

        html_file.write_text(str(soup), encoding="utf-8")

    print("Done cleaning.")


def load_web_offline(web_path, chunk_size=1000, chunk_overlap=150):
    loader = DirectoryLoader(
        web_path,  # extracted zip folder
        glob="**/*.html",
        loader_cls=BSHTMLLoader,
        loader_kwargs={"open_encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
    )

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # was 512
        chunk_overlap=chunk_overlap,  # was 100
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 200]

    return chunks


def init_embeddings():
    """Initialize embedding generation (local model or OpenAI API)"""
    if USE_OPENAI_EMBEDDINGS:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY must be set to use OpenAI embeddings.")
        print(f"Using OpenAI embeddings model: {OPENAI_EMBEDDING_MODEL}")
        # [New] Use OpenAI's /v1/embeddings API via LangChain's OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL,
                                api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
    else:
        print(f"Using local embeddings model: {EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cuda'},
                                     encode_kwargs={'normalize_embeddings': True})


if __name__ == '__main__':
    manifest_o = load_manifest()  # [New] load existing manifest to know already indexed files
    chunks_to_index, new_manifest = load_and_process_documents(PDF_DIR, manifest_o)

    # For loading html / web based(sphinx) docs
    # new_manifest = manifest_o
    #strip_sphinx_junk(Path(r".\html\web\docs"))
    # chunks_to_index = load_web_offline(r".\html\web\docs")

    if not chunks_to_index:
        print("No new or changed documents to embed.")
    else:
        embeddings = init_embeddings()
        batch_size = 5461  # Maximum batch size allowed
        # Add to existing vector store or create a new one
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            vectorstore = Chroma(persist_directory=PERSIST_DIR,
                                 embedding_function=embeddings)
            # vectorstore.add_documents(chunks_to_index)  # [New] add only new/updated chunks
            # Split the records into smaller batches
            batches = [chunks_to_index[i:i + batch_size] for i in range(0, len(chunks_to_index), batch_size)]

            # Process each batch separately
            for batch in batches:
                vectorstore.add_documents(batch)
            # vectorstore.persist() not needed?
            print(f"Vector store updated with {len(chunks_to_index)} new/updated chunks. [BATCH: {len(batches)}]")
        else:
            #vectorstore = Chroma.from_documents(chunks_to_index, embedding=embeddings,
            #                                    persist_directory=PERSIST_DIR)
            batches = [chunks_to_index[i:i + batch_size] for i in range(0, len(chunks_to_index), batch_size)]
            vectorstore = Chroma.from_documents(batches[0], embedding=embeddings,
                                                persist_directory=PERSIST_DIR)
            for batch in batches[1:]:
                vectorstore.add_documents(batch)
            print(f"Created new vector store with {len(chunks_to_index)} chunks. [BATCH: {len(batches)}]")
        save_manifest(new_manifest)  # [New] persist updated manifest
        print("Ingestion complete. Manifest updated.")
