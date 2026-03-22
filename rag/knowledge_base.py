import chromadb

# Create a persistent local Chroma client
client = chromadb.PersistentClient(path="rag/chroma_db")

# Create or load collection
collection = client.get_or_create_collection(name="network_security")

# Add attack descriptions and troubleshooting steps
documents = [
    "DoS Hulk attack: high traffic flood. Troubleshoot by rate limiting, blocking suspicious IPs, and monitoring request spikes.",
    "Slowloris attack: keeps many connections open for a long time. Troubleshoot by setting connection timeouts and limiting concurrent sessions.",
    "Heartbleed vulnerability: SSL memory exposure issue. Troubleshoot by patching OpenSSL and rotating certificates or keys.",
    "GoldenEye attack: HTTP flooding attack. Troubleshoot by checking request patterns and applying web firewall filtering."
]

# Store each document with a unique ID
for i, doc in enumerate(documents):
    collection.upsert(
        documents=[doc],
        ids=[str(i)]
    )

print("Knowledge base created successfully")