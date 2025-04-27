# import os
# import json
# import shutil
# from datetime import datetime
# from pathlib import Path

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# load_dotenv()
# app = FastAPI()

# # Constants
# PDF_DIR = Path("pdfs")
# INDEX_DIR = Path("indexes")
# LOG_DIR = Path("chat_logs")
# PDF_DIR.mkdir(exist_ok=True)
# INDEX_DIR.mkdir(exist_ok=True)
# LOG_DIR.mkdir(exist_ok=True)

# # Init LLM + Embedding
# api_key = os.getenv("GROQ_API_KEY")
# if not api_key:
#     raise ValueError("Set GROQ_API_KEY in .env")

# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key)

# # Helper: Get index path
# def get_index_path(pdf_name):
#     stem = Path(pdf_name).stem.replace(" ", "_")
#     return INDEX_DIR / stem

# # Models
# class QueryRequest(BaseModel):
#     pdf_name: str
#     question: str
#     mode: str = "detailed"

# # === ROUTES ===

# @app.post("/upload_pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     pdf_path = PDF_DIR / file.filename
#     with open(pdf_path, "wb") as f:
#         f.write(await file.read())

#     index_path = get_index_path(file.filename)
#     index_path.mkdir(exist_ok=True)

#     loader = PyPDFLoader(str(pdf_path))
#     pages = loader.load()

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(pages)

#     vectorstore = FAISS.from_documents(chunks, embedding_model)
#     vectorstore.save_local(str(index_path))

#     return {"message": f"{file.filename} uploaded and indexed successfully."}


# @app.get("/list_pdfs")
# def list_pdfs():
#     return {"pdfs": [f.name for f in PDF_DIR.iterdir() if f.suffix == ".pdf"]}


# @app.post("/ask")
# def ask_pdf(request: QueryRequest):
#     pdf_path = PDF_DIR / request.pdf_name
#     index_path = get_index_path(request.pdf_name)

#     if not pdf_path.exists() or not (index_path / "index.faiss").exists():
#         return JSONResponse(status_code=404, content={"error": "PDF or index not found"})

#     vectorstore = FAISS.load_local(
#         str(index_path),
#         embeddings=embedding_model,
#         allow_dangerous_deserialization=True
#     )

#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     # Prompt Template
#     base_template = """You are an AI assistant helping users understand a PDF.

# Use ONLY the context below to answer the user's question.

# - If the user asked for a short or 2-line answer, follow that.
# - If the mode is 'short', give a brief 1–2 line answer.
# - If the mode is 'detailed', give a complete, rich explanation.
# - If the answer isn't in the context, say: "I don't know".

# Mode: {mode}
# Context:
# {context}

# Question:
# {question}

# Answer:"""

#     def build_prompt(mode_value):
#         return PromptTemplate(
#             template=base_template.replace("{mode}", mode_value),
#             input_variables=["context", "question"]
#         )

#     prompt = build_prompt(request.mode)

#     rag_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt},
#         return_source_documents=True
#     )

#     result = rag_chain({"query": request.question})
#     answer = result["result"]
#     sources = result["source_documents"]

#     # Save chat log
#     log_file = LOG_DIR / f"{request.pdf_name.replace('.pdf', '')}.json"
#     chat_record = {
#         "timestamp": datetime.now().isoformat(),
#         "question": request.question,
#         "answer": answer,
#         "mode": request.mode
#     }
#     if log_file.exists():
#         with open(log_file, "r", encoding="utf-8") as f:
#             logs = json.load(f)
#     else:
#         logs = []
#     logs.append(chat_record)
#     with open(log_file, "w", encoding="utf-8") as f:
#         json.dump(logs, f, indent=2)

#     return {
#         "answer": answer,
#         "sources": [doc.page_content[:500] for doc in sources]
#     }


# @app.post("/summarize")
# def summarize_pdf(request: QueryRequest):
#     pdf_path = PDF_DIR / request.pdf_name
#     if not pdf_path.exists():
#         return JSONResponse(status_code=404, content={"error": "PDF not found"})

#     loader = PyPDFLoader(str(pdf_path))
#     pages = loader.load()
#     full_text = "\n".join([p.page_content for p in pages])[:3000]

#     prompt = f"""Summarize the following document content briefly:

# {full_text}

# Summary:"""

#     summary = llm.invoke(prompt)
#     return {"summary": summary}


# @app.delete("/delete_pdf")
# def delete_pdf(pdf_name: str = Form(...)):
#     pdf_path = PDF_DIR / pdf_name
#     index_path = get_index_path(pdf_name)

#     if pdf_path.exists():
#         os.remove(pdf_path)
#     if index_path.exists():
#         shutil.rmtree(index_path)

#     return {"message": f"{pdf_name} and its index deleted."}


# @app.get("/get_chat_logs")
# def get_chat_logs(pdf_name: str):
#     log_file = LOG_DIR / f"{pdf_name.replace('.pdf', '')}.json"
#     if not log_file.exists():
#         return {"logs": []}
#     with open(log_file, "r", encoding="utf-8") as f:
#         logs = json.load(f)
#     return {"logs": logs}
import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from pymongo import MongoClient

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()

app = FastAPI()

# Constants
PDF_DIR = Path("pdfs")
INDEX_DIR = Path("indexes")
PDF_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["chatbot_db"]
chats_collection = db["chat_logs"]

# Init LLM + Embedding
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Set GROQ_API_KEY in .env")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=api_key)

# Helper
def get_index_path(pdf_name):
    stem = Path(pdf_name).stem.replace(" ", "_")
    return INDEX_DIR / stem

class QueryRequest(BaseModel):
    pdf_name: str
    question: str
    mode: str = "detailed"

# === ROUTES ===

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = PDF_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    index_path = get_index_path(file.filename)
    index_path.mkdir(exist_ok=True)

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(str(index_path))

    return {"message": f"{file.filename} uploaded and indexed successfully."}

@app.get("/list_pdfs")
def list_pdfs():
    return {"pdfs": [f.name for f in PDF_DIR.iterdir() if f.suffix == ".pdf"]}

@app.post("/ask")
def ask_pdf(request: QueryRequest):
    pdf_path = PDF_DIR / request.pdf_name
    index_path = get_index_path(request.pdf_name)

    if not pdf_path.exists() or not (index_path / "index.faiss").exists():
        return JSONResponse(status_code=404, content={"error": "PDF or index not found"})

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    base_template = """You are an AI assistant helping users understand a PDF.

Use ONLY the context below to answer the user's question.

- If the user asked for a short or 2-line answer, follow that.
- If the mode is 'short', give a brief 1–2 line answer.
- If the mode is 'detailed', give a complete, rich explanation.
- If the answer isn't in the context, say: "I don't know".

Mode: {mode}
Context:
{context}

Question:
{question}

Answer:"""

    def build_prompt(mode_value):
        return PromptTemplate(
            template=base_template.replace("{mode}", mode_value),
            input_variables=["context", "question"]
        )

    prompt = build_prompt(request.mode)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = rag_chain({"query": request.question})
    answer = result["result"]
    sources = result["source_documents"]

    # Save to MongoDB
    chat_record = {
        "pdf_name": request.pdf_name,
        "timestamp": datetime.utcnow(),
        "question": request.question,
        "answer": answer,
        "mode": request.mode
    }
    chats_collection.insert_one(chat_record)

    return {
        "answer": answer,
        "sources": [doc.page_content[:500] for doc in sources]
    }

@app.post("/summarize")
def summarize_pdf(request: QueryRequest):
    pdf_path = PDF_DIR / request.pdf_name
    if not pdf_path.exists():
        return JSONResponse(status_code=404, content={"error": "PDF not found"})

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])[:3000]

    prompt = f"""Summarize the following document content briefly:

{full_text}

Summary:"""

    summary = llm.invoke(prompt)
    return {"summary": summary}

@app.delete("/delete_pdf")
def delete_pdf(pdf_name: str = Form(...)):
    pdf_path = PDF_DIR / pdf_name
    index_path = get_index_path(pdf_name)

    if pdf_path.exists():
        os.remove(pdf_path)
    if index_path.exists():
        shutil.rmtree(index_path)

    # Also delete chat logs from MongoDB
    chats_collection.delete_many({"pdf_name": pdf_name})

    return {"message": f"{pdf_name} and its index deleted."}
