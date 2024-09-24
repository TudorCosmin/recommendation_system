from similarity_search import search_similar
from product_approximation import get_approximated_product
from embeddings_generation import stringify_user
import pandas as pd
import time

from global_config import GlobalConfig
config = GlobalConfig()

def select_top_products(user_id: int, products_df: pd.DataFrame, user_behavior_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Select the top k products for a specific user based on the specified rules.

    Args:
        user_id: The ID of the user for whom to select products.
        products_df: A DataFrame containing product information.
        user_behavior_df: A DataFrame containing user behavior data.
        k: The number of top products to return.

    Returns:
        A DataFrame containing the top k products, filtered and selected based on the specified rules.
    """
    # Identify purchased and viewed products by the user
    purchased_products = set(user_behavior_df[(user_behavior_df['user_id'] == user_id) & pd.notnull(user_behavior_df['purchase_timestamp'])]['product_id'])
    viewed_products = set(user_behavior_df[user_behavior_df['user_id'] == user_id]['product_id'])
    
    # Filter out purchased products
    filtered_products_df = products_df[~products_df['product_id'].isin(purchased_products)]
    
    # Split the products into 40% and 60%
    show_anyway_percent = 0.4
    split_index = int(len(filtered_products_df) * show_anyway_percent)
    show_anyway_df = filtered_products_df.iloc[:split_index]
    remaining_unseen_products_df = filtered_products_df.iloc[split_index:][~filtered_products_df.iloc[split_index:]['product_id'].isin(viewed_products)]

    # Concatenate the results and return the top k
    return pd.concat([show_anyway_df, remaining_unseen_products_df]).head(k)
from similarity_search import search_similar
from product_approximation import get_approximated_product
from embeddings_generation import stringify_user
import pandas as pd

from global_config import GlobalConfig
config = GlobalConfig()

def select_top_products(user_id: int, products_df: pd.DataFrame, user_behavior_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Select the top k products for a specific user by filtering out previously purchased items and prioritizing 
    new or less interacted products.

    This function determines the top k product recommendations by first excluding any products the user has 
    already purchased. It then splits the remaining products into two segments: one containing a portion 
    of products that will be shown regardless of previous interactions (to encourage repeated engagement), 
    and another containing products that the user has not viewed yet. The selected products from both segments 
    are combined, and the top k products are returned.

    Args:
        user_id: The ID of the user for whom to select products.
        products_df: A DataFrame containing product information.
        user_behavior_df: A DataFrame containing user behavior data.
        k: The number of top products to return.

    Returns:
        A DataFrame containing the top k products, filtered and selected based on the specified rules.
    """
    # Identify purchased and viewed products by the user
    purchased_products = set(user_behavior_df[(user_behavior_df['user_id'] == user_id) & pd.notnull(user_behavior_df['purchase_timestamp'])]['product_id'])
    viewed_products = set(user_behavior_df[user_behavior_df['user_id'] == user_id]['product_id'])
    
    # Filter out purchased products
    filtered_products_df = products_df[~products_df['product_id'].isin(purchased_products)]
    
    # Split the products into 40% and 60%
    show_anyway_percent = 0.4
    split_index = int(len(filtered_products_df) * show_anyway_percent)
    show_anyway_df = filtered_products_df.iloc[:split_index]
    remaining_unseen_products_df = filtered_products_df.iloc[split_index:][~filtered_products_df.iloc[split_index:]['product_id'].isin(viewed_products)]

    # Concatenate the results and return the top k
    return pd.concat([show_anyway_df, remaining_unseen_products_df]).head(k)

def recommend(user_id: int) -> pd.DataFrame:
    """
    Generate personalized product recommendations for a given user.

    Args:
        user_id: The ID of the user for whom recommendations are to be generated.

    Returns:
        A DataFrame containing the recommended products, with each row representing a product.
    """
    recommend_start_time = time.time()

    user_row = config.user_data_df[config.user_data_df['user_id'] == user_id].iloc[0]
    user_str = stringify_user(user_row)
    
    # Search for similar users in the user collection
    user_ann = search_similar(
        query=user_str,
        collection_name=config.user_collection_name,
        top_k=config.USER_SEARCH_TOP_K,
        model=config.model,
        tokenizer=config.tokenizer
    )

    # Get approximated product preferences based on the most similar users
    similar_user_ids = [neighbour.payload["user_id"] for neighbour in user_ann]
    approximated_products = get_approximated_product(similar_user_ids)

    # Search for similar products in the product collection
    product_ann = search_similar(
        query=approximated_products,
        collection_name=config.product_collection_name,
        top_k=config.PRODUCT_SEARCH_TOP_K,
        model=config.model,
        tokenizer=config.tokenizer
    )

    # Select the top relevant products
    recommendations_df = pd.DataFrame([neighbour.payload for neighbour in product_ann])
    top_recommendations_df = select_top_products(
        user_id=user_id,
        products_df=recommendations_df,
        user_behavior_df=config.user_behavior_df,
        k=config.NUMBER_OF_RECOMMENDED_PRODUCTS
    )

    recommend_elapsed_time = time.time() - recommend_start_time
    print(f"Recommendation done in {recommend_elapsed_time:.4f} seconds.\n\n")

    return top_recommendations_df
