# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch==0.60 databricks-sdk transformers torch sentence-transformers
# COMMAND ----------
# MAGIC %restart_python
# COMMAND ----------
# MAGIC %md ## import and parameters
# COMMAND ----------
import pandas as pd
from pyspark.sql.types import ArrayType, FloatType
from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.reranker import DatabricksReranker
from databricks.sdk import WorkspaceClient
import json
try:
    from transformers import AutoModel, AutoTokenizer
    import torch
except ImportError:
    print("Hugging Face/Torch libraries not found. External model embedding will fail if requested.")
    HUGGING_FACE_ENABLED = False
else:
    HUGGING_FACE_ENABLED = True

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_index_table_source = dbutils.widgets.get("raw_index_table_source")
raw_table_with_embeddings = dbutils.widgets.get("raw_table_with_embeddings")
vector_endpoint_name = dbutils.widgets.get("vector_endpoint_name")
index_name = dbutils.widgets.get("index_name")
embedding_model = dbutils.widgets.get("embedding_model")
embedding_model_hf = dbutils.widgets.get("embedding_model_hf")
embedding_dimension_str = dbutils.widgets.get("embedding_dimension")
use_dbks_client_str = dbutils.widgets.get("use_dbks_client")
embedding_dimension = int(embedding_dimension_str)
use_dbks_client = use_dbks_client_str.lower() == "true"
QUERY_TEXT = dbutils.widgets.get("text_query")

query_vector = None # initialize the vector

full_name = lambda table: f"{catalog}.{schema}.{table}"

print(f"Using Embedding Strategy: {'Databricks Model Serving' if use_dbks_client else 'Hugging Face Transformers'}")

EMBEDDING_SCHEMA = ArrayType(FloatType(), containsNull=False)

# COMMAND ----------
# MAGIC %md ## load model or initialize client
# COMMAND ----------

GLOBAL_MODEL = {}
try:
    GLOBAL_W = WorkspaceClient()
    print("Databricks WorkspaceClient initialized.")
except Exception as e:
    print(f"Client initialization failed: {e}")
    GLOBAL_W = None

if not use_dbks_client:
    #external model
    if not HUGGING_FACE_ENABLED:
        raise ImportError("Hugging Face model requested but libraries not installed.")
        
    try:
        print(f"Loading Hugging Face model: {embedding_model_hf}...")
        DEVICE = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_hf)
        model = AutoModel.from_pretrained(embedding_model_hf).to(DEVICE)
        
        GLOBAL_MODEL['tokenizer'] = tokenizer
        GLOBAL_MODEL['model'] = model
        GLOBAL_MODEL['device'] = DEVICE
        
        print(f"Model '{embedding_model_hf}' loaded successfully.")
    except Exception as e:
        print(f"Driver model load failed: {e}")
        GLOBAL_MODEL = {}

# COMMAND ----------
# MAGIC %md ## Embed query text
# COMMAND ----------

if not use_dbks_client:
    def mean_pooling(model_output, attention_mask):
        """Helper function for mean pooling in Hugging Face models."""
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    if GLOBAL_MODEL:
        try:
            print(f"Embedding query using Hugging Face model: {embedding_model_hf}")
            tokenizer = GLOBAL_MODEL['tokenizer']
            model = GLOBAL_MODEL['model']
            device = GLOBAL_MODEL['device']
            
            encoded_input = tokenizer(
                [QUERY_TEXT], 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                model_output = model(**encoded_input)
            
            # Apply mean pooling and convert to standard Python list
            embeddings_tensor = mean_pooling(model_output, encoded_input['attention_mask'])
            query_vector = embeddings_tensor.cpu().numpy().tolist()[0]
            print(f"Query embedded successfully. Vector size: {len(query_vector)}")

        except Exception as e:
            print(f"Embedding failed for Hugging Face model: {e}")
            query_vector = None

# COMMAND ----------
# MAGIC %md ## Query the index
# COMMAND ----------


if use_dbks_client:
    
    try:
        index = client = VectorSearchClient(disable_notice=True).get_index(index_name=full_name(index_name)+"_client")
        response = index.similarity_search(
            columns ="summary",
            query_text=QUERY_TEXT,
            num_results=2,
            #reranker=DatabricksReranker(columns_to_rerank=["summary", "title", "other_column"])
        )
        result = json.dumps(response, indent=2)
    except Exception as e:
        raise e
    
else:
    try:
        if query_vector is None:
            raise ValueError("Query vector is None, cannot perform similarity search.")
        
        index = client = VectorSearchClient(disable_notice=True).get_index(index_name=full_name(index_name)+"_custom")
        response = index.similarity_search(
            columns ="summary",
            query_vector=query_vector,
            num_results=2,
        )
        result = json.dumps(response, indent=2)
    except Exception as e:
        print(f"Similarity search failed: {e}")
        result = None

print("Query result: \n", result)
