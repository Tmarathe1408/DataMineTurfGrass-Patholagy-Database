from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import sqlite3

# Initialize the embedding model
llm = OllamaLLM(model="llama3.2-8")
embed_model = OllamaEmbeddings(model="mxbai-embed-large-8")

def embed_and_insert_data_from_db(client, collection_name, db_path, table_name):
    data_entries = fetch_data_from_sqlite(db_path, table_name)
    if not data_entries:
        print("No data found in the database.")
        return

    for row in data_entries:
        ids = row[0]
        identi = row[1]
        identifier = sanitize_identifier(identi, mode="remove")[:250]
        paragraph_content = row[2]
        table_content = row[3]

        # Check if identifier exists
        if identifier_exists(client, collection_name, ids):
            print(f"Skipping existing identifier: '{identifier}'\n")
            continue  

        # Generate embeddings using Ollama
        embedding_paragraph = embed_model.embed(paragraph_content)
        embedding_table = embed_model.embed(table_content)

        # Insert data using client
        client.insert(collection_name, [{
            "ids": ids,
            "identifier": identifier,
            "paragraph_contents": embedding_paragraph,
            "table_contents": embedding_table
        }])
        print("Inserted data successfully!")

