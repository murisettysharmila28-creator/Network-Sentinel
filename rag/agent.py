from retriever import retrieve_info
from src.models.predict import predict_attack

def run_agent(input_df, user_query=None):
    predicted_label = predict_attack(input_df)

    if user_query is None:
        user_query = f"How to troubleshoot {predicted_label} attack?"

    response = retrieve_info(user_query)

    return predicted_label, response