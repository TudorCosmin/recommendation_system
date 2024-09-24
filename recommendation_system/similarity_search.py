import time
from qdrant_client import models
import numpy as np
import pandas as pd
import torch
from typing import Union, List
from global_config import GlobalConfig

# Initialize the global configuration
config = GlobalConfig()

def create_collection(collection_name: str, vector_len: int) -> None:
    """
    Create a new collection in Qdrant with the specified name and vector length. 
    If the collection already exists, it is deleted and recreated.

    Args:
        collection_name: The name of the collection to be created.
        vector_len: The length of the vectors that will be stored in this collection.
    """

    if config.qclient.collection_exists(collection_name):
        config.qclient.delete_collection(collection_name)

    config.qclient.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_len,
            distance=models.Distance.EUCLID,
        ),
    )

def prepare_qdrant_points(embedding_df: pd.DataFrame, point_details_df: pd.DataFrame, id_column: str) -> List[models.PointStruct]:
    """
    Prepare points for uploading to a Qdrant collection. Each point consists of an embedding vector 
    and its associated metadata (payload).

    Args:
        embedding_df: DataFrame containing embeddings with an identifier column.
        point_details_df: DataFrame containing detailed metadata for each point.
        id_column: The column name used to match embeddings with their corresponding metadata.

    Returns:
        A list of points ready to be uploaded to Qdrant.
    """

    points = []
    for idx, row in embedding_df.iterrows():
        details_dict = point_details_df.loc[point_details_df[id_column] == row[id_column]].iloc[0].to_dict()
        points.append(
            models.PointStruct(
                id=idx,
                vector=np.array(row['embedding']),
                payload=details_dict
            )
        )
    return points

def upload_points(collection_name: str, points: List[models.PointStruct]) -> None:
    """
    Upload a list of points to a specified Qdrant collection.

    Args:
        collection_name: The name of the collection to upload points to.
        points: A list of points to be uploaded.
    """
    
    config.qclient.upload_points(
        collection_name=collection_name,
        points=points
    )

def get_embedding(text: str, model, tokenizer) -> List[float]:
    """
    Generate an embedding for a given text using a pre-trained model and tokenizer.

    Args:
        text: The input text to be converted into an embedding.
        model: The pre-trained model used to generate the embedding.
        tokenizer: The tokenizer used to preprocess the text.

    Returns:
        List: The resulting embedding as a list of floats.
    """

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        hard_skill_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    return hard_skill_embedding.tolist()

def search_similar(query: Union[str, List[float]], collection_name: str, top_k: int, model, tokenizer) -> List[models.ScoredPoint]:
    """
    Search for similar items in a specified Qdrant collection based on a query, which can be either a text 
    string or an embedding vector.

    Args:
        query: The query input, either a string (to be converted to an embedding) or a list of floats (embedding).
        collection_name: The name of the collection to search in.
        top_k: The number of top similar items to retrieve.
        model: The model used for generating embeddings (if query is a string).
        tokenizer: The tokenizer used for processing the text (if query is a string).

    Returns:
        List: A list of the nearest neighbours based on the query embedding.
    """

    if isinstance(query, str):
        query_emb = get_embedding(text=query, model=model, tokenizer=tokenizer)
    elif isinstance(query, list) and all(isinstance(i, float) for i in query):
        query_emb = query

    nearest_neighbours = config.qclient.search(
        collection_name=collection_name,
        query_vector=query_emb,
        limit=top_k
    )

    return nearest_neighbours

def initialize_collections() -> None:
    """
    Initialize the user and product collections in Qdrant by creating the collections, preparing the points, 
    and uploading them.
    """

    init_start_time = time.time()
    print("Starting initializing collections...")

    # Initialize the user collection
    user_data_df = config.user_data_df
    user_embeddings_df = config.user_embeddings_df
    create_collection(
        collection_name=config.user_collection_name,
        vector_len=len(user_embeddings_df.iloc[0]['embedding'])
    )
    user_points = prepare_qdrant_points(
        embedding_df=user_embeddings_df,
        point_details_df=user_data_df,
        id_column="user_id"
    )
    upload_points(config.user_collection_name, user_points)

    # Initialize the product collection
    product_data_df = config.product_data_df
    product_embeddings_df = config.product_embeddings_df
    create_collection(
        collection_name=config.product_collection_name,
        vector_len=len(product_embeddings_df.iloc[0]['embedding'])
    )
    product_points = prepare_qdrant_points(
        embedding_df=product_embeddings_df,
        point_details_df=product_data_df,
        id_column="product_id"
    )
    upload_points(config.product_collection_name, product_points)

    init_elapsed_time = time.time() - init_start_time
    print(f"Initialization done in {init_elapsed_time:.4f} seconds.\n\n")

initialize_collections()