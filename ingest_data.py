import os
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
DATA_PATH = "Data files"
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """
    Reads PDF files from the DATA_PATH, chunks them, 
    creates embeddings, and saves the FAISS index locally.
    """
    print(f"--- Starting Data Ingestion ---")
    
    # 1. Setup Data Directory
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Directory '{DATA_PATH}' created. Please add your .pdf files there and run this script again.")
        return

    # 2. Load PDF Documents
    print(f"Scanning '{DATA_PATH}' for PDF files...")
    documents = []
    if os.path.exists(DATA_PATH):
        for file in os.listdir(DATA_PATH):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(DATA_PATH, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    if not documents:
        print(f"No PDF files found in '{DATA_PATH}'. Exiting.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 3. Split Text into Chunks
    # Using a smaller chunk size for better RAG retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split documents into {len(texts)} chunks.")

    # 4. Create Embeddings
    print(f"Loading Embedding Model ({MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME, 
        model_kwargs={'device': 'cpu'}
    )

    # 5. Create and Save FAISS Index
    print("Building FAISS Vector Store...")
    db = FAISS.from_documents(texts, embeddings)
    
    print(f"Saving FAISS index to '{DB_FAISS_PATH}'...")
    db.save_local(DB_FAISS_PATH)
    
    print("--- Ingestion Complete. You can now run the Streamlit App. ---")

if __name__ == "__main__":
    create_vector_db()