import os
import sys
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# NOTE: This script creates/overwrites the vectorstore from scratch.
# It will DELETE any previously uploaded documents.
# Use the frontend upload feature to add documents incrementally instead.
# Use this script only for initial setup or to reset the knowledge base.

# Get file path from command line argument or user input
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = input("Enter the path to your document (PDF or TXT): ").strip().strip('"').strip("'")

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found!")
    sys.exit(1)

# Check file extension
file_extension = file_path.split(".")[-1].lower()

if file_extension not in ["pdf", "txt"]:
    print(f"Error: Only PDF and TXT files are supported. Got: {file_extension}")
    sys.exit(1)

print(f"Loading document: {file_path}")

# Load document based on file type
if file_extension == "pdf":
    loader = PyPDFLoader(file_path)
else:  # txt
    loader = TextLoader(file_path, encoding="utf-8")

documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

# Split doc into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
print("Creating vectorstore...")

# FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print(f"✅ Vectorstore saved successfully to 'faiss_index' with {len(chunks)} chunks!")

