# Vector search embedding in databricks
## Plan and documentation
The goal of this PoC is to identify usage and benefit of storing embeddings on databricks to serve a model. We will try implementing a pipeline for the ingestion of new data to feed LLM model. <br>
- [databricks documentation](https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search)
- [databricks vector search modules](https://api-docs.databricks.com/python/vector-search/index.html)
- [faiss](https://faiss.ai/index.html) for vector similarity search

## Workflow
We are going to create a vector search entity in databricks (via UI or python SDK). Creation od the delta table with the data to embed. Creation of the index in the vector search based on some target columns in the delta table. Querying of the index based on the embedding model. Serving of the embedding index to a language model via RAG.
Once all the individual element will be understood we are going to deploy an end to end pipeline that will update the embedding for the target model.

## Setup local model
To setup a SML locally we actually need to load the model into memory and store the cache into a models directory. we will execute this in bash
```
mkdir -p models
python3 -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
import warnings; \
import urllib3; \
from huggingface_hub import snapshot_download; \
from dotenv import load_dotenv; \
import os; \
load_dotenv(); \
MODEL_ID = os.environ.get('MODEL_ID', 'google/flan-t5-small'); \
CACHE_DIR = 'models'; \
os.environ['CURL_CA_BUNDLE'] = ''; \

warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning); \
warnings.filterwarnings('ignore'); \
SNAPSHOT_PATH = snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR); \
print(f'\n\n \nFinal Snapshot Path: {SNAPSHOT_PATH}\n\n');"
```
The final printed row will contain the models cache path to be used to launch the backend API, so copy it and paste it in the .env file

## Run backend server
Having the correct .env configuration we can now spin up the backend server and start making query for benchmarking.

```uvicorn src.backend.main:app --reload --host 0.0.0.0 --port 8000```

To make some test query we can use curl
```
curl -X POST 'http://127.0.0.1:8000/query' \
-H 'Content-Type: application/json' \
-H 'X-Api-Key: default_key' \
-d '{
    "query": "What is a delta lake",
    "top_k": 1,
    "query_context": false
}'
```
This will show the outcome of the query to the defualt untrained model with no additional context. <br>
Let's try now using databricks vector for retrival of additional context.

```
curl -X POST 'http://127.0.0.1:8000/query' \
-H 'Content-Type: application/json' \
-H 'X-Api-Key: default_key' \
-d '{
    "query": "What is a delta lake",
    "top_k": 1,
    "query_context": true
}'
```

We can see now that the output will contains information coming from index. Means that connection was succesful.



#### Issues with certificate
seems that uv is making some strange things