# app.py
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from typing import Optional, List
import requests
import os
import logging
from dotenv import load_dotenv

load_dotenv()
#setup in docker compose file
DATABRICKS_WS = os.environ.get("DATABRICKS_HOST")  # e.g. "https://dbc-XXX.cloud.databricks.com"
VECTOR_ENDPOINT_NAME = os.environ.get("VECTOR_ENDPOINT_NAME")  # e.g. "test-vector-search"
INDEX_NAME = os.environ.get("INDEX_NAME")  # e.g. "testing.default.poc_index_test1"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "default-key")
TOP_K = int(os.environ.get("TOP_K", "3"))
MODEL_ID = os.environ.get("MODEL_ID", "google/flan-t5-small")
CACHED_MODEL_DIR = os.environ.get("CACHED_MODEL_DIR")
logging.basicConfig(level=logging.INFO)


app = FastAPI(title="Local RAG Server")

tokenizer = AutoTokenizer.from_pretrained(CACHED_MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(CACHED_MODEL_DIR, local_files_only=True)

class QueryPayload(BaseModel):
    query: str
    top_k: Optional[int] = None
    query_context: Optional[bool] = False

class ContextPayload(BaseModel):
    query: str
    top_k: Optional[int] = None

def check_auth(x_api_key: Optional[str]):
    if x_api_key != LOCAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def query_databricks_vector(query: str, top_k: int = 3):
    """Call databricks endpoint for vector similarity search"""
    if not (DATABRICKS_WS and VECTOR_ENDPOINT_NAME and INDEX_NAME and DATABRICKS_TOKEN):
        raise RuntimeError("Databricks connection variables not set in env")
    search_client = VectorSearchClient(
        workspace_url=DATABRICKS_WS,
        personal_access_token=DATABRICKS_TOKEN, #allowed also via service principal

    )

    index = search_client.get_index(endpoint_name=VECTOR_ENDPOINT_NAME, index_name=INDEX_NAME)

    items = index.similarity_search(num_results=top_k, columns=["text","id"], query_text=query)
    print(items)
    print(type(items))
    texts = []
    seen = set()
    for row in items.get("result").get("data_array"):
        text = row[0]
        if not text:
            continue
        # Simple dedupe: if text identical, skip
        if text in seen:
            continue
        seen.add(text)
        texts.append({"id": row[1], "text": text})
    return texts

@app.post("/context")
async def get_context(payload: ContextPayload, x_api_key: Optional[str] = Header(None)):
    """Perform similarty search on index"""
    check_auth(x_api_key)
    top_k = payload.top_k or TOP_K
    try:
        docs = query_databricks_vector(payload.query, top_k)
    except Exception as e:
        logging.exception("Error querying Databricks vector endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    return {"query": payload.query, "top_k": top_k, "docs": docs}

@app.post("/query")
async def answer(payload: QueryPayload, x_api_key: Optional[str] = Header(None)):
    """Retrieve context then generate an answer using a small LLM."""
    check_auth(x_api_key)
    top_k = payload.top_k or TOP_K
    if payload.query_context == True:
        try:
            docs = query_databricks_vector(payload.query, top_k)
        except Exception as e:
            logging.exception("Error querying Databricks vector endpoint")
            raise HTTPException(status_code=500, detail=f"vector query: {e}")
    else:
        docs = []
    # Build context string (deduped). Keep it short and labeled.
    context_blocks = []
    for d in docs:
        # include id so you can cite sources later
        context_blocks.append(f"[{d.get('id')}] {d.get('text')}")
    context = "\n\n".join(context_blocks).strip()
    if not context:
        context = "No relevant context found."

    # Build instruction prompt for FLAN-T5
    prompt = (
        "You are a helpful assistant. Use the provided Context to answer the question. Answer with 'I do not have the required information' if contex is empty.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {payload.query}\n\n"
        "Answer concisely and do not repeat the context verbatim.\n"
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    gen = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,   # deterministic for POC
        early_stopping=True
    )
    answer_text = tokenizer.decode(gen[0], skip_special_tokens=True)

    # Return answer plus contexts used
    return {"query": payload.query, "answer": answer_text, "docs": docs}

