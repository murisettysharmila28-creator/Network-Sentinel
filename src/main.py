from src.data.loader import load_dataset
from src.agent.incident_agent import run_agent

def main():
    df = load_dataset("data/raw/sample.csv")

    predicted_label, response = run_agent(df)

    print("Predicted Attack:", predicted_label)
    print("Explanation:", response)


if __name__ == "__main__":
    main()