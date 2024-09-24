import os
import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient

class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Ensure the folder structure exists
        self.ensure_folder_structure('data')

        # Input files
        try:
            self.product_data_df = pd.read_csv("data/input/product_data.csv")
            self.user_data_df = pd.read_csv("data/input/user_data.csv")
            self.user_behavior_df = pd.read_csv("data/input/user_behavior_data.csv")
            self.user_ratings_df = pd.read_csv("data/input/user_ratings.csv")
        except FileNotFoundError:
            import data_generation
            data_generation.generate_full_data()
            self.product_data_df = pd.read_csv("data/input/product_data.csv")
            self.user_data_df = pd.read_csv("data/input/user_data.csv")
            self.user_behavior_df = pd.read_csv("data/input/user_behavior_data.csv")
            self.user_ratings_df = pd.read_csv("data/input/user_ratings.csv")

        self.tokenizer = AutoTokenizer.from_pretrained("dunzhang/stella_en_1.5B_v5")
        self.model = AutoModel.from_pretrained("dunzhang/stella_en_1.5B_v5")

        # Embeddings files
        try:
            self.product_embeddings_df = pd.read_csv("data/embeddings/product_embeddings.csv")
            self.user_embeddings_df = pd.read_csv("data/embeddings/user_embeddings.csv")
        except FileNotFoundError:
            import embeddings_generation
            embeddings_generation.generate_full_embeddings()
            self.product_embeddings_df = pd.read_csv("data/embeddings/product_embeddings.csv")
            self.user_embeddings_df = pd.read_csv("data/embeddings/user_embeddings.csv")
        self.product_embeddings_df['embedding'] = self.product_embeddings_df['embedding'].apply(literal_eval)
        self.user_embeddings_df['embedding'] = self.user_embeddings_df['embedding'].apply(literal_eval)

        self.qclient = QdrantClient(":memory:")
        self.user_collection_name = "user_collection"
        self.product_collection_name = "product_collection"
        
        self.MAX_RATING = 5.0

        self.USER_SEARCH_TOP_K = 10
        self.PRODUCT_SEARCH_TOP_K = 100
        self.NUMBER_OF_RECOMMENDED_PRODUCTS = 10
    
    def ensure_folder_structure(self, base_dir: str) -> None:
        folder_structure = ['embeddings', 'input', 'visualisation']
        for folder in folder_structure:
            folder_path = os.path.join(base_dir, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)