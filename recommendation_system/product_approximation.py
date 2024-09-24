import pandas as pd
import numpy as np
from typing import List, Optional

from global_config import GlobalConfig
config = GlobalConfig()

def calculate_interaction_weight(row: pd.Series) -> float:
    """
    Calculate the interaction weight based on user behavior.

    The base weight for viewing a product is 1.0.
    If the product was purchased, the weight is increased by an additional factor of 1.0.
    
    Args:
        row: A row from the user behavior DataFrame.

    Returns:
        The calculated interaction weight.
    """
    weight = 1.0
    if pd.notnull(row['purchase_timestamp']):
        weight += 1.0
    return weight

def calculate_rating_weight(row: pd.Series, rating_df: pd.DataFrame) -> float:
    """
    Adjust the interaction weight based on the product rating.

    The adjustment is as follows:
    - If the rating is above a certain high_rating_threshold, the weight increases in proportion to how much the rating exceeds this threshold.
    - If the rating is between low_rating_threshold and high_rating_threshold, the weight remains unchanged.
    - If the rating is below a certain low_rating_threshold, the weight decreases in proportion to how much the rating is below this threshold.

    Args:
        row: A row from the user behavior DataFrame.
        rating_df: The DataFrame containing user ratings.

    Returns:
        The adjusted weight after considering the product rating.
    """
    
    # Compute the thresholds based on MAX_RATING
    high_rating_threshold = config.MAX_RATING * 0.6
    low_rating_threshold = config.MAX_RATING * 0.4
    
    # Compute the scaling factors based on MAX_RATING
    high_rating_scale = config.MAX_RATING - high_rating_threshold  # Range above high_rating_threshold
    low_rating_scale = low_rating_threshold  # Range below low_rating_threshold

    user_id, product_id, base_weight = row['user_id'], row['product_id'], row['interaction_weight']
    rating_row = rating_df[(rating_df['user_id'] == user_id) & (rating_df['product_id'] == product_id)]
    
    if not rating_row.empty:
        rating = rating_row.iloc[0]['rating']
        
        if rating > high_rating_threshold: # Rating of 4 or 5
            # Increase the weight based on how much the rating exceeds the high_rating_threshold
            base_weight *= 1 + (rating - high_rating_threshold) / high_rating_scale
        elif low_rating_threshold <= rating <= high_rating_threshold: # Rating of 3
            # No change for neutral ratings
            pass
        else: # Rating of 1 or 2
            # Decrease the weight based on how much the rating is below the low_rating_threshold
            base_weight *= 1 - (low_rating_threshold - rating) / low_rating_scale

    return base_weight

def prepare_user_behavior(user_behavior_df: pd.DataFrame, rating_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the user behavior DataFrame by calculating interaction weights and final weights.

    This function adds two new columns to the user_behavior_df:
    - 'interaction_weight': Base weight depending on whether the product was viewed or purchased.
    - 'final_weight': Adjusted weight based on the user's rating for the product.

    Args:
        user_behavior_df: The DataFrame containing user behavior data.
        rating_df: The DataFrame containing user ratings.

    Returns:
        The updated user_behavior_df with interaction and final weights.
    """
    user_behavior_df['interaction_weight'] = user_behavior_df.apply(calculate_interaction_weight, axis=1)
    user_behavior_df['final_weight'] = user_behavior_df.apply(lambda row: calculate_rating_weight(row, rating_df), axis=1)
    return user_behavior_df

def weighted_average_embedding(user_ids: List[int], user_behavior_df: pd.DataFrame, product_df: pd.DataFrame) -> Optional[List[float]]:
    """
    Compute the weighted average embedding for a list of user IDs.

    The function calculates a single imaginary product embedding based on the interactions of all users in the input list.
    The embeddings are weighted by the interaction type and further adjusted by product ratings.

    Args:
        user_ids: A list of user IDs to compute the embeddings for.
        user_behavior_df: The DataFrame containing user behavior data.
        product_df: The DataFrame containing product embeddings.

    Returns:
        A single weighted average product embedding for all specified users, or None if no interactions are found.
    """
    total_weight = 0
    weighted_embedding_sum = np.zeros(len(product_df.iloc[0]['embedding']))
    
    for user_id in user_ids:
        user_interactions = user_behavior_df[user_behavior_df['user_id'] == user_id]
        if user_interactions.empty:
            continue
        
        for _, row in user_interactions.iterrows():
            product_embedding = product_df[product_df['product_id'] == row['product_id']].iloc[0]['embedding']
            weight = row['final_weight']
            
            weighted_embedding_sum += np.array(product_embedding) * weight
            total_weight += weight
    
    if total_weight > 0:
        weighted_avg_embedding = weighted_embedding_sum / total_weight
        return weighted_avg_embedding.tolist()
    else:
        return None

# Process user behavior to include interaction weights and final weights
config.user_behavior_df = prepare_user_behavior(config.user_behavior_df, config.user_ratings_df)

def get_approximated_product(user_ids: List[int]) -> Optional[List[float]]:
    """
    Compute and return the weighted average product embedding for a list of user IDs.

    Args:
        user_ids: A list of user IDs for which to calculate the weighted average product embedding.

    Returns:
        A single weighted average product embedding representing the combined behavior of all specified users.
        If no interaction data is available for any of the users, the value is None.
    """

    # Compute the combined weighted average embedding for all users in the list
    approximated_product = weighted_average_embedding(user_ids, config.user_behavior_df, config.product_embeddings_df)
    
    return approximated_product