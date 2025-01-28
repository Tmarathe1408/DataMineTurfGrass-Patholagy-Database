from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np

# Milvus connection details
_HOST = "127.0.0.1"  # Or localhost
_PORT = "19530"  # Default gRPC port for Milvus Standalone

def connect_to_milvus(db_name="default"):
    print(f"Connecting to Milvus...\n")
    
    # Connect using gRPC
    connections.connect(
        host=_HOST,  
        port=_PORT,        
        timeout=60  # Increased timeout
    )

    # List available databases
    db_list = db.list_database()
    print(db_list)

    # Check if the 'turf_grass' database exists
    if "turf_grass" not in db_list:
        db.create_database("turf_grass")
    
    db.using_database("turf_grass")
    print(f"Using database: turf_grass")

def create_collection():
    # Define the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary key
        FieldSchema(name="identifier", dtype=DataType.VARCHAR, max_length=100),       # Unique identifier
        FieldSchema(name="table_contents", dtype=DataType.FLOAT_VECTOR, dim=384)      # Embedded vector data
        FieldSchema(name="paragraph_contents", dtype=DataType.FLOAT_VECTOR, dim=512) # Embedded vector data
    ]
    
    schema = CollectionSchema(fields=fields, description="Turf grass data collection")

    # Create the collection if it doesn't already exist
    collection_name = "turf_grass_data"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        collection = Collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")

    return collection

def create_index(collection):
    # Check if an index already exists on the 'table_contents' field
    if collection.indexes:
        print("Index already exists on 'table_contents'. Skipping index creation.")
        return
    
    # Define index parameters
    index_params = {
        "metric_type": "IP",  # "L2" for Euclidean; use "IP" for cosine similarity
        "index_type": "IVF_FLAT",
        "params": {"nlist": 384}  # Number of clusters; adjust based on data size and performance needs
    }
    
    # Create the index on the 'table_contents' field
    collection.create_index(field_name="identifier", index_params=index_params)
    print("Index created successfully on 'Identifier'.")



def fetch_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")  # Adjust query as needed
    row = cursor.fetchone()  # Gets the first entry
    conn.close()
    return row

def embed_and_insert_data_from_db(collection, db_path, table_name):
    # Fetch a single row from the SQLite database
    data_entry = fetch_data_from_sqlite(db_path, table_name)
    if data_entry is None:
        print("No data found in the database.")
        return

    # Assuming the data entry structure is (id, identifier, content); adjust as necessary
    identifier = data_entry[1]
    content = data_entry[2]
    print(content)
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
    
    # Generate embedding for the content
    embedding = model.encode(content).tolist()  # Convert embedding to list
    print(f"Embedding: {embedding}") # Prints embedding so comment out if not needed. Using for DEMO purposes

    # Insert data into the Milvus collection
    collection.insert([[identifier], [embedding]])
    print("Single data entry from SQLite inserted successfully.")

def retrieve_all_data(collection):
    # Load the collection into memory
    collection.load()
    
    # Query all data with a specified limit
    results = collection.query(expr="", limit=10)  # Empty expression to get all entries
    print("All data in the collection:")
    for result in results:
        print(result)

# Paths and setup
db_path = "./final_output_completed.db"  # Path to the SQLite database
table_name = "grass"  # table name

# Connect to Milvus
connect_to_milvus()

# Create the collection
collection = create_collection()

# Create an index on the collection
create_index(collection)

# Embed and insert a single entry from SQLite
embed_and_insert_data_from_db(collection, db_path, table_name)

# Retrieve all data
retrieve_all_data(collection)

# Drop collection every run so no need to repeat entries
# Delete this for future uses
collection.drop()



