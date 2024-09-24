import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)

# Generate User Data (1000 users)
def generate_users(n=1000):
    return pd.DataFrame({
        'user_id': range(1001, 1001 + n),
        'age': np.random.randint(18, 70, n),
        'gender': np.random.choice(['M', 'F'], n),
        'location': [fake.city() for _ in range(n)],
        'join_date': [fake.date_between(start_date='-3y', end_date='today') for _ in range(n)]
    })

# Generate Product Data (1000 products)
def generate_products(n=1000):
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports', 'Toys',
                  'Food', 'Automotive', 'Health']
    brands = [fake.company() for _ in range(50)]  # 50 unique brands
    return pd.DataFrame({
        'product_id': range(2001, 2001 + n),
        'category': np.random.choice(categories, n),
        'price': np.round(np.random.uniform(5, 1000, n), 2),
        'brand': np.random.choice(brands, n),
        'avg_rating': np.round(np.random.uniform(1, 5, n), 1)
    })


# Generate User Behavior Data (5000 interactions)
def generate_user_behavior(users, products, n=5000):
    user_ids = users['user_id'].tolist()
    product_ids = products['product_id'].tolist()
    data = []
    for _ in range(n):
        user_id = np.random.choice(user_ids)
        product_id = np.random.choice(product_ids)
        view_time = fake.date_time_between(start_date='-1y', end_date='now')
        purchase_time = view_time + timedelta(minutes=np.random.randint(5, 1440)) if np.random.random() < 0.3 else None
        data.append([user_id, product_id, view_time, purchase_time])
        
    return pd.DataFrame(data, columns=['user_id', 'product_id', 'view_timestamp', 'purchase_timestamp'])

# Generate User Ratings (2000 ratings)
def generate_user_ratings(users, products, n=2000):
    user_ids = users['user_id'].tolist()
    product_ids = products['product_id'].tolist()
    data = []
    for _ in range(n):
        user_id = np.random.choice(user_ids)
        product_id = np.random.choice(product_ids)
        rating = np.random.randint(1, 6)
        timestamp = fake.date_time_between(start_date='-1y', end_date='now')
        data.append([user_id, product_id, rating, timestamp])
    return pd.DataFrame(data, columns=['user_id', 'product_id', 'rating', 'timestamp'])

def generate_full_data():
    print("\nData generation has started...")

    # Generate the datasets
    users = generate_users()
    products = generate_products()
    behavior = generate_user_behavior(users, products)
    ratings = generate_user_ratings(users, products)

    # Save the datasets to CSV files
    users.to_csv('data/input/user_data.csv', index=False)
    products.to_csv('data/input/product_data.csv', index=False)
    behavior.to_csv('data/input/user_behavior_data.csv', index=False)
    ratings.to_csv('data/input/user_ratings.csv', index=False)

    print("Datasets generated and saved to CSV files.")
    print(f"Users: {len(users)} records")
    print(f"Products: {len(products)} records")
    print(f"User Behavior: {len(behavior)} records")
    print(f"User Ratings: {len(ratings)} records")

    print("Data generation complete.\n")