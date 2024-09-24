import pandas as pd
from global_config import GlobalConfig
from recommendation import recommend

config = GlobalConfig()

def main():
    user_id = 1234
    recommendations = recommend(user_id)

    print(f"Top {config.NUMBER_OF_RECOMMENDED_PRODUCTS} recommendations for user {user_id}:")
    print(recommendations)

if __name__ == "__main__":
    main()
