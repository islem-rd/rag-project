from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

load_dotenv()
llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings,
    allow_dangerous_deserialization=True 
)
#retrieval
retriever = vectorstore.as_retriever()

#prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

# Create RetrievalQA chain
retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

#fastapi setup
app = FastAPI(
    title="RAG Chatbot API",
    description="An API for chatting with a knowledge base."
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    print(f"Received question: {request.question}")
    
    result = retrieval_qa.invoke({retrieval_qa.input_key: request.question})
    
    return {"answer": result[retrieval_qa.output_key]}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF or TXT) and add it to the vectorstore.
    """
    # Check file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_extension = file.filename.split(".")[-1].lower()
    
    if file_extension not in ["pdf", "txt"]:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF and TXT files are supported"
        )
    
    tmp_file_path = None
    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load the document based on file type
        if file_extension == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        else:  # txt
            loader = TextLoader(tmp_file_path, encoding="utf-8")
        
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Add documents to existing vectorstore
        global vectorstore, retriever, retrieval_qa
        vectorstore.add_documents(chunks)
        vectorstore.save_local("faiss_index")
        
        # Update retriever and chain with new vectorstore
        retriever = vectorstore.as_retriever()
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        return {
            "message": f"Successfully uploaded and processed {file.filename}",
            "chunks_added": len(chunks)
        }
    
    except Exception as e:
        # Clean up temporary file on error
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

#run
if __name__ == "__main__":
    print("Starting FastAPI server at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)