import chromadb

# Connect to the same persistent Chroma database
client = chromadb.PersistentClient(path="rag/chroma_db")

# Load the existing collection
collection = client.get_collection(name="network_security")

def retrieve_info(query):
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    return results["documents"][0][0]


# Quick test
if __name__ == "__main__":
    query = "How to troubleshoot DoS Hulk attack?"
    response = retrieve_info(query)
    print(response)