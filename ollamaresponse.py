from pymilvus import connections, db, utility, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import ollama

_HOST = "127.0.0.1"  # Or localhost
_PORT = "19530"  # Default gRPC port for Milvus Standalone
# Connect to Milvus
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
# Load your collection

connect_to_milvus()
collection_name = "turf_grass_data"
collection = Collection(collection_name)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model outputs 384-dimensional embeddings
def get_embedding(text):
    """Generate an embedding for the query using Ollama."""
    response = model.encode(text).tolist()  # Adjust model as needed
    return response

# Query the database
def askOllama(text):
    query_text = " "
    query_embedding = get_embedding(query_text)
      #  "index_type": "IVF_FLAT",
    
    search_params = {"metric_type": "IP", "params": {"nlist": 384}}

    results = collection.search(
        data=[query_embedding], 
        anns_field="paragraph_contents",  # Adjust to match your schema
        param=search_params, 
        limit=5
    )

    # Print retrieved results
    for result in results[0]:
        print(f"ID: {result.id}, Distance: {result.distance}")
    # Extract the top results as context
    context = "\n".join([str(res.id) for res in list(results[0])])
    return context

def generateresponse(text) : # Generate a response using Ollama
    context = askOllama(text)
    response = ollama.chat(
        model="llama_3.2",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are an expert in turfgrass and plant diseases."},
            {"role": "user", "content": f"Based on the following data, explain:\n{context}"}
        ]
    )

    print(response["message"]["content"])

#ask question
generateresponse("Explain the effects of DMI fungicides on Bermuda grass")
