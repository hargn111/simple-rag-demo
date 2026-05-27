#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
from dotenv import load_dotenv
import json

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Test dataset: questions with known relevant document keywords
TEST_QUERIES = [
    {
        "query": "What are the system requirements for z/OS?",
        "expected_keywords": ["memory", "storage", "processor", "cpu"],
        "relevant_docs": ["zos_manual.pdf"]  # Adjust to your actual files
    },
    {
        "query": "How do I configure the backup procedure?",
        "expected_keywords": ["backup", "schedule", "configuration"],
        "relevant_docs": ["backup_procedure.pdf"]
    },
    # Add more test queries based on your documentation
]

def evaluate_retrieval(vectorstore, test_queries: List[Dict], k=4):
    """Evaluate retrieval quality using Precision@K and Recall@K"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    results = []

    for test in test_queries:
        query = test["query"]
        relevant_docs = set(test.get("relevant_docs", []))
        expected_keywords = set(test.get("expected_keywords", []))

        # Retrieve documents
        retrieved = retriever.get_relevant_documents(query)
        retrieved_sources = {doc.metadata.get('source', '').split('/')[-1] for doc in retrieved}

        # Calculate Precision@K: how many retrieved are relevant?
        if len(retrieved_sources) > 0:
            precision = len(relevant_docs.intersection(retrieved_sources)) / len(retrieved_sources)
        else:
            precision = 0.0

        # Calculate Recall@K: how many relevant docs were retrieved?
        if len(relevant_docs) > 0:
            recall = len(relevant_docs.intersection(retrieved_sources)) / len(relevant_docs)
        else:
            recall = 1.0 if len(retrieved_sources) == 0 else 0.0

        # Check keyword coverage in retrieved text
        retrieved_text = " ".join([doc.page_content.lower() for doc in retrieved])
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_text)
        keyword_coverage = keyword_hits / len(expected_keywords) if expected_keywords else 0.0

        result = {
            "query": query,
            "precision@k": precision,
            "recall@k": recall,
            "keyword_coverage": keyword_coverage,
            "retrieved_count": len(retrieved),
            "retrieved_sources": list(retrieved_sources)
        }
        results.append(result)

        # Print result
        print(f"\nQuery: {query}")
        print(f"  Precision@{k}: {precision:.2f}")
        print(f"  Recall@{k}: {recall:.2f}")
        print(f"  Keyword Coverage: {keyword_coverage:.2f}")
        print(f"  Retrieved: {retrieved_sources}")

    # Calculate averages
    avg_precision = sum(r["precision@k"] for r in results) / len(results)
    avg_recall = sum(r["recall@k"] for r in results) / len(results)
    avg_coverage = sum(r["keyword_coverage"] for r in results) / len(results)

    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}")
    print(f"Average Precision@{k}: {avg_precision:.2f}")
    print(f"Average Recall@{k}: {avg_recall:.2f}")
    print(f"Average Keyword Coverage: {avg_coverage:.2f}")

    return results


if __name__ == "__main__":
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    print("\nRunning retrieval evaluation...")
    evaluate_retrieval(vectorstore, TEST_QUERIES, k=4)
