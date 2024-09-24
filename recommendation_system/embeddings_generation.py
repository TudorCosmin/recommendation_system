import torch
import csv
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_embedding(text, model, tokenizer):
    """
    Generate an embedding for a given text using the specified model and tokenizer.

    Args:
        text: The input text to be converted into an embedding.
        model: The pre-trained model used to generate the embedding.
        tokenizer: The tokenizer used to preprocess the text.

    Returns:
        The resulting embedding as a list of floats.
    """

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        hard_skill_embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    return hard_skill_embedding.tolist()

def generate_embeddings(model, tokenizer, input_ls, output_file, id_column):
    """
    Generate embeddings for a list of text entries and save them to a CSV file, 
    avoiding duplicates based on an ID column.

    Args:
        model: The pre-trained model used to generate the embeddings.
        tokenizer: The tokenizer used to preprocess the text.
        input_ls: A list of dictionaries, each containing an ID and text to be embedded.
        output_file: The path to the CSV file where the embeddings will be saved.
        id_column: The column name used to identify unique entries in the input list.
    """

    try:
        existing_df = pd.read_csv(output_file)
        existing_elements = set(existing_df[id_column])
    except FileNotFoundError:
        existing_elements = set()

    # Filter out elements that have already been processed
    new_elements_ls = [elem for elem in input_ls if elem[id_column] not in existing_elements]
    total_new_skills = len(new_elements_ls)

    with open(output_file, 'a', newline='') as f_output:
        writer = csv.writer(f_output, quoting=csv.QUOTE_MINIMAL)

        if f_output.tell() == 0:
            writer.writerow([id_column, 'text', 'embedding'])

        # Generate embeddings for new elements and save them to the output file
        with tqdm(total=total_new_skills, desc=f"Processing {id_column}") as pbar:
            for elem in new_elements_ls:
                text = elem["text"]
                embedding_list = get_embedding(text, model, tokenizer)
                embedding_str = json.dumps(embedding_list)
                
                writer.writerow([elem[id_column], text, embedding_str])
                pbar.update(1)
    
    print("Embeddings generation completed.")

def stringify_product(row):
    """
    Convert a product's data into a formatted string.

    Args:
        row: A row of product data.

    Returns:
        A formatted string representing the product.
    """
    return f"CATEGORY: {row['category'].upper()}, price: {row['price']}, brand: {row['brand']}, avg_rating: {row['avg_rating']}"

def format_products(input_file):
    """
    Format product data from a CSV file into a list of dictionaries with IDs and text descriptions.

    Args:
        input_file: The path to the CSV file containing product data.

    Returns:
        A list of dictionaries where each dictionary has a 'product_id' and a 'text' key.
    """
    
    product_df = pd.read_csv(input_file)
    formatted_dicts = product_df.apply(lambda row: {
        "product_id": row['product_id'],
        "text": stringify_product(row=row)
    }, axis=1)

    return formatted_dicts.tolist()

def stringify_user(row):
    """
    Convert a user's data into a formatted string.

    Args:
        row: A row of user data.

    Returns:
        A formatted string representing the user.
    """
    return f"age: {row['age']}, gender: {'male' if row['gender'] == 'M' else 'female'}, location: {row['location']}"

def format_users(input_file):
    """
    Format user data from a CSV file into a list of dictionaries with IDs and text descriptions.

    Args:
        input_file: The path to the CSV file containing user data.

    Returns:
        A list of dictionaries where each dictionary has a 'user_id' and a 'text' key.
    """
    
    user_df = pd.read_csv(input_file)
    formatted_dicts = user_df.apply(lambda row: {
        "user_id": row['user_id'],
        "text": stringify_user(row=row)
    }, axis=1)

    return formatted_dicts.tolist()

def generate_full_embeddings():
    """
    Generate and save embeddings for both products and users.

    This method loads the product and user data, converts them into text descriptions, 
    generates embeddings for each entry, and saves the embeddings to CSV files.

    Args:
        None

    Returns:
        None
    """
    print("\nEmbeddings generation started...")

    # Load the pre-trained model and tokenizer
    model_name = "dunzhang/stella_en_1.5B_v5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate embeddings for products
    product_ls = format_products(input_file="data/input/product_data.csv")
    generate_embeddings(
        model=model,
        tokenizer=tokenizer,
        input_ls=product_ls,
        output_file="data/embeddings/product_embeddings.csv",
        id_column="product_id"
    )

    # Generate embeddings for users
    user_ls = format_users(input_file="data/input/user_data.csv")
    generate_embeddings(
        model=model,
        tokenizer=tokenizer,
        input_ls=user_ls,
        output_file="data/embeddings/user_embeddings.csv",
        id_column="user_id"
    )

    print("Embeddings generation complete.\n")

# if __name__ == "__main__":
#     generate_full_embeddings()