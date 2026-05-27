import os
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuration
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# LLM Configuration (choose one)
USE_LOCAL_LLM = False  # Set to False for OpenAI API
load_dotenv()

if USE_LOCAL_LLM:
    # Local llama.cpp server (OpenAI-compatible)
    LLM_API_KEY = "not-needed"  # llama.cpp doesn't require a key
    LLM_BASE_URL = os.getenv('LOCAL_LLM_URL', 'http://localhost:8080/v1')
    LLM_MODEL = os.getenv('LOCAL_LLM_MODEL', 'llama-3.3-8b-instruct')  # Arbitrary name for local
else:
    # OpenAI API
    LLM_API_KEY = os.getenv('OPENAI_API_KEY', '')  # [New] API key for OpenAI embeddings (required if using remote)
    LLM_BASE_URL = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1')
    LLM_MODEL = os.getenv('OPENAI_API_MODEL', 'gpt-5.4')

# Custom prompt template for RAG
PROMPT_TEMPLATE = """You are an assistant helping with technical documentation. Use the following context to answer the question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer (be specific and cite relevant details from context):"""


def load_vector_store():
    """Load existing vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

    return vectorstore


def create_rag_chain(vectorstore, k=4):
    """Create RAG chain with retriever"""

    # Initialize LLM (works with both local and OpenAI)
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0.1,  # Low temperature for factual answers
    )
    #llm.openai_api_key = LLM_API_KEY
    
    # Create retriever (top-k chunks)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    # Create prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = insert all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


def query_rag(question, qa_chain, show_sources=True):
    """Query the RAG system"""
    result = qa_chain.invoke({"query": question})

    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}")
    print(f"\nANSWER:\n{result['result']}\n")

    if show_sources:
        print(f"\n{'-'*80}")
        print("SOURCE DOCUMENTS:")
        print(f"{'-'*80}")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n--- Source {i} ---")
            print(f"File: {doc.metadata.get('source', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:300]}...")

    return result


if __name__ == "__main__":
    import sys

    # Load vector store
    print("Loading vector store...")
    vectorstore_o = load_vector_store()

    # Create RAG chain
    print("Initializing RAG chain...")
    use_k_override = 4  # 4 is default, =num sources, make configurable
    qa_chain_o = create_rag_chain(vectorstore_o, k=use_k_override)

    # Interactive mode or single query
    if len(sys.argv) > 1:
        # Single query from command line
        question_o = " ".join(sys.argv[1:])
        query_rag(question_o, qa_chain_o)
    else:
        # Interactive CLI
        print("\n" + "="*80)
        print(f"RAG SYSTEM READY [k={use_k_override}] - Enter your questions (type 'exit' to quit)")
        print("="*80)

        while True:
            question_o = input("\nYour question: ").strip()
            if question_o.lower() in ['exit', 'quit', 'q']:
                break

            if question_o:
                query_rag(question_o, qa_chain_o)
