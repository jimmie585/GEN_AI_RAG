from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import tempfile

app = FastAPI()

# Load the model once
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

retriever_global = None

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global retriever_global
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever_global = vectorstore.as_retriever()

    return {"message": "PDF uploaded and processed successfully."}

@app.get("/ask/")
def ask_question(question: str):
    if not retriever_global:
        return JSONResponse(status_code=400, content={"error": "Please upload a PDF first."})

    relevant_docs = retriever_global.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"Answer the question using only the context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    response = flan_pipeline(prompt, max_new_tokens=200, temperature=0.7, top_k=50, top_p=0.9, do_sample=True)
    return {"answer": response[0]['generated_text']}
