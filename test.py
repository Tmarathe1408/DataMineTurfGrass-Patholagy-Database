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
        FieldSchema(name="identifier", dtype=DataType.VARCHAR, max_length=500),       # Unique identifier
        FieldSchema(name="paragraph_contents", dtype=DataType.FLOAT_VECTOR, dim=384), # Embedded vector data
        FieldSchema(name="table_contents", dtype=DataType.FLOAT_VECTOR, dim=384)      # Embedded vector data
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
    collection.create_index(field_name="paragraph_contents", index_params=index_params)
    collection.create_index(field_name="table_contents", index_params=index_params)

    print("Indexes created successfully on paragraph and table contents.")



def fetch_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch a single data entry from the specified table
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")  # Adjust query as needed
    row = cursor.fetchall()
    conn.close()
    return row

def embed_and_insert_data_from_db(collection, db_path, table_name):
    # Fetch a single row from the SQLite database
    data_entry = fetch_data_from_sqlite(db_path, table_name)
    if data_entry is None:
        print("No data found in the database.")
        return
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
        
    for row in data_entry:
        # Assuming the data entry structure is (id, identifier, content); adjust as necessary
        identi = row[1]
        identifier = sanitize_identifier(identi[:250], mode="remove")
        paragraph_content = row[2]
        table_content = row[3]

        # Check to see if identifier was already inserted
        if identifier_exists(collection, identifier):
            print(f"Identifier: '{identifier}' already exists in Milvus. Skipping insert.\n")
        else:    
            # Generate embedding for the content
            embedding_paragraph = model.encode(paragraph_content).tolist()  # Convert embedding to list

            embedding_table = model.encode(table_content).tolist()  # Convert embedding to list

            # Insert data into the Milvus collection
            collection.insert([[identifier], [embedding_paragraph], [embedding_table]])
            collection.flush()
            print("Single data entry from SQLite inserted successfully.")

def retrieve_all_data(collection):
    # Load the collection into memory
    collection.load()
    
    # Query all data with a specified limit
    results = collection.query(expr="", limit=10)  # Empty expression to get all entries
    print("All data in the collection:")
    for result in results:
        print(result)

def identifier_exists(collection, identifier_value):
    # Load collection into memory for querying
    collection.load()

    # Build an expression to query by the string field "identifier"
    expr = f"identifier == '{identifier_value}'"
    
    # Return only the "identifier" field, limit=1 to just see if there's at least one match
    results = collection.query(expr=expr, output_fields=["identifier"], limit=1)
    
    # If 'results' is not empty, we have a match
    return len(results) > 0


def print_collection_info(collection):
    print("=== Collection Info ===")
    print("Name:", collection.name)
    print("Description:", collection.description)
    print("Number of entities:", collection.num_entities)
    print("\nSchema:")
    for field in collection.schema.fields:
        print(f"  - {field.name}, type: {field.dtype}, is_primary: {field.is_primary}")

    print("\nIndexes:")
    if collection.indexes:
        for idx in collection.indexes:
            print(f"  - Field name: {idx.field_name}")
            print(f"    - Index type: {idx.index_name}")
            print(f"    - Params: {idx.params}")
    else:
        print("  No indexes found.")

    print("=======================\n")

def sanitize_identifier(identifier: str, mode="remove") -> str:
    """
    Removes or replaces single quotes in the given identifier string.
    mode="remove": deletes all single quotes.
    mode="double": replaces single quotes with double quotes.
    """
    if mode == "remove":
        # Completely remove single quotes
        return identifier.replace("'", "")
    elif mode == "double":
        # Replace single quotes with double quotes
        return identifier.replace("'", '"')
    else:
        return identifier  # No change if mode isn't recognized

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
print_collection_info(collection)

