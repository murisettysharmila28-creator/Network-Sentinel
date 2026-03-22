from retriever import retrieve_info

# Simulated model output
predicted_label = "DoS Hulk"

# Build a query using the predicted attack label
query = f"How to troubleshoot {predicted_label} attack?"

# Retrieve explanation and troubleshooting guidance
response = retrieve_info(query)

print("Predicted Label:", predicted_label)
print("Agent Response:", response)