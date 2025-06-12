# process.py
import os
import fitz  # PyMuPDF
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# Setup Transformers model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Embedding model for vector search
embedding_model = HuggingFaceEmbeddings()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Utility: Extract text from PDF
def extract_text_from_pdf(file) -> str:
    file.seek(0)
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Utility: Extract text from TXT
def extract_text_from_txt(file) -> str:
    return file.read().decode("utf-8")

# Chunk and embed documents, return retriever + topic summary dictionary
def chunk_and_embed(files):
    documents = []
    metadatas = []
    topic_dict = {}

    for file in files:
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        elif file.type == "text/plain":
            text = extract_text_from_txt(file)
        else:
            continue

        chunks = text_splitter.split_text(text)
        topic_dict[file.name] = chunks  # Save chunks by filename

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"document": file.name, "chunk": i})

    collection_name = f"collection-{uuid.uuid4().hex[:8]}"
    db = Chroma.from_texts(documents, embedding_model, metadatas=metadatas, collection_name=collection_name)
    return db.as_retriever(), topic_dict

# Answer query using Transformers model and context from retriever
def answer_query_from_documents(query: str, retriever, k: int = 4) -> dict:
    docs = retriever.get_relevant_documents(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "answer": answer,
        "sources": [
            {
                "document": doc.metadata.get("document", "Unknown"),
                "content": doc.page_content
            }
            for doc in docs
        ]
    }
