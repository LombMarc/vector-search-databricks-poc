# Databricks notebook source
# MAGIC %md ## library installation
# COMMAND ----------
# MAGIC %pip install databricks-vectorsearch==0.60 databricks-sdk transformers==4.57.3 torch==2.9.1
# COMMAND ----------
# MAGIC %restart_python
# COMMAND ----------
# MAGIC %md ## loading parameters
# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
import pandas as pd
from pyspark.sql.functions import pandas_udf, col, current_timestamp
from pyspark.sql.types import ArrayType, FloatType
# Ensure these are installed via the Environment panel!
from transformers import AutoModel, AutoTokenizer
import torch 

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
raw_index_table_source = dbutils.widgets.get("raw_index_table_source")
embed_index_table_source = dbutils.widgets.get("raw_table_with_embeddings")
vector_endpoint_name = dbutils.widgets.get("vector_endpoint_name")
index_name = dbutils.widgets.get("index_name")
EMBEDDING_ENDPOINT_NAME = dbutils.widgets.get("embedding_model")
HF_MODEL_NAME = dbutils.widgets.get("embedding_model_hf") 
EMBEDDING_DIMENSION = int(dbutils.widgets.get("embedding_dimension"))
use_client = dbutils.widgets.get("use_dbks_client").lower() == "true"

# COMMAND ----------
# MAGIC %md ## Verify creation of catalog and schema
# COMMAND ----------
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

full_name = lambda table: f"{catalog}.{schema}.{table}"

client = VectorSearchClient(disable_notice=True)
if use_client:
    try:
        #delete index to overwrite
        client.delete_index(index_name=full_name(index_name)+"_client")
    except Exception as e:
        print(f"Index not existent, creating new one...\n{e}")

if not use_client:
    try:
        #delete index to overwrite
        client.delete_index(index_name=full_name(index_name)+"_custom")
    except Exception as e:
        print(f"Index not existent, creating new one...\n{e}")

# COMMAND ----------
# MAGIC %md ## Generation of sample data for test
# COMMAND ----------
data = [
    (1, "Delta Lake provides ACID transactions and schema enforcement for data lakes."),
    (2, "Databricks integrates data engineering and AI on a single platform."),
    (3, "Apache Spark is a fast engine for large-scale data processing."),
    (4, "Vector Search allows semantic similarity search across embeddings."),
    (5, "RAG combines retrieval and generation for question answering."),
]

columns = ["id", "text"]

df = spark.createDataFrame(data, columns)
df.write.format("delta").mode("overwrite").saveAsTable(full_name(raw_index_table_source))

#CDF is required for vector search indexes to track changes in the source table
try:
    spark.sql(f"""
        ALTER TABLE {full_name(raw_index_table_source)} 
        SET TBLPROPERTIES (
          'delta.enableChangeDataFeed' = 'true'
        )
    """)

except Exception as e:
    print(f"Error enabling CDF: {e}")

# COMMAND ----------
# MAGIC %md ## Create vector search endpoint
# COMMAND ----------

try:
    client.get_endpoint(name= vector_endpoint_name)
except Exception as e:
    print(f"Vectro search endpoint not existent, creating {vector_endpoint_name}...")
    client.create_endpoint(
        name=vector_endpoint_name,
        endpoint_type="STANDARD"
    )
# COMMAND ----------
# MAGIC %md ## Create vector index
# COMMAND ----------
if use_client:

    index = client.create_delta_sync_index(
    endpoint_name=vector_endpoint_name,
    source_table_name=full_name(raw_index_table_source),
    index_name=full_name(index_name)+"_client",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_source_column="text",
    embedding_dimension=EMBEDDING_DIMENSION,
    embedding_model_endpoint_name=EMBEDDING_ENDPOINT_NAME,
    model_endpoint_name_for_query= EMBEDDING_ENDPOINT_NAME
    )

# COMMAND ----------
# MAGIC %md ## Compute embedding column with external model
# COMMAND ----------

EMBEDDING_SCHEMA = ArrayType(FloatType())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_AND_TOKENIZER = {}

try:
    print(f"Loading Hugging Face model: {HF_MODEL_NAME}...")
    
    # 1. Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModel.from_pretrained(HF_MODEL_NAME).to(DEVICE)
    
    # 2. Store them in the global dictionary
    MODEL_AND_TOKENIZER['tokenizer'] = tokenizer
    MODEL_AND_TOKENIZER['model'] = model
    
    print("Model loaded successfully on the driver.")
    
except Exception as e:
    print(f"Driver model load failed! Check dependencies and internet access: {e}")


@pandas_udf(EMBEDDING_SCHEMA)
def batch_embed_text_hf(text_series: pd.Series) -> pd.Series:
    """Generates embeddings using a globally initialized Hugging Face model.
    """
    if not MODEL_AND_TOKENIZER:
        return pd.Series([None] * len(text_series))
        
    tokenizer = MODEL_AND_TOKENIZER['tokenizer']
    model = MODEL_AND_TOKENIZER['model']

    try:
        #get the token from the input text
        encoded_input = tokenizer(
            text_series.tolist(), 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(DEVICE)

        #compute vector values
        with torch.no_grad():
            model_output = model(**encoded_input)

        #mean pooling to compute average of all token, useful to get general context of the sentence
        token_embeddings = model_output[0] 
        input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()
        
        return pd.Series(embeddings)

    except Exception as e:
        print(f"Batch embedding failed: {e}")
        return pd.Series([None] * len(text_series))


embedded_df = (df
    .withColumn("embedding", batch_embed_text_hf(col("text")))
    .withColumn("timestamp", current_timestamp())
)

if use_client:
    display(embedded_df)
else:
    embedded_df.write.format("delta").mode("overwrite").saveAsTable(full_name(embed_index_table_source))
    #CDF is required for vector search indexes to track changes in the source table
    try:
        spark.sql(f"""
            ALTER TABLE {full_name(embed_index_table_source)} 
            SET TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true'
            )
        """)

    except Exception as e:
        print(f"Error enabling CDF: {e}")
# COMMAND ----------
# MAGIC %md ## Create vector index
# COMMAND ----------
if not use_client:
    try:
        #delete index to overwrite
        client.delete_index(index_name=full_name(index_name)+"_custom")
    except Exception as e:
        print(f"Index not existent, creating new one...\n{e}")

    index = client.create_delta_sync_index(
    endpoint_name=vector_endpoint_name,
    source_table_name=full_name(embed_index_table_source),
    index_name=full_name(index_name)+"_custom",
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_vector_column="embedding",
    embedding_dimension=EMBEDDING_DIMENSION,

    )


# COMMAND ----------
# MAGIC %md ## sync index via pythond sdk
# COMMAND ----------

# index = client.get_index(index_name=full_name(index_name)+"_custom" if not use_client else full_name(index_name)+"_client")
#index.sync()