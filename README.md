# Network Sentinel

**Network Sentinel** is an end-to-end **Machine Learning + Retrieval-Augmented Generation (RAG)** based **Network Intrusion Detection System** built for CICIDS-style network traffic data.  
It combines a high-performing **XGBoost classifier** for attack detection with a **ChromaDB-backed knowledge base** and an **LLM/fallback reasoning layer** to generate security-focused incident summaries and guidance in a **Streamlit dashboard**.

## Live Demo
[Network Sentinel App](https://network-sentinel-sharmila.streamlit.app/)

---

## Project Overview

The system is designed to:

- Accept uploaded network traffic CSV files
- Clean and preprocess the input data
- Apply feature engineering
- Use a trained ML model to predict the attack type
- Retrieve relevant cybersecurity knowledge using RAG
- Generate an incident summary and recommended action
- Present results in an interactive Streamlit interface

This project focuses on both **prediction accuracy** and **practical explainability**, making it more than just a classifier. It acts as a lightweight analyst-assistance system.

---

## Problem Statement

Traditional intrusion detection systems often stop at classification. In real-world security workflows, that is not enough. Analysts also need:

- context about the detected threat
- likely indicators
- severity interpretation
- mitigation guidance

Network Sentinel addresses this by combining:

1. **Machine Learning** for attack classification  
2. **RAG** for contextual cybersecurity knowledge retrieval  
3. **LLM/fallback generation** for incident-style explanation  

---

## Dataset

**Source file used in the project:**

`data/raw/Wednesday-workingHours.pcap_ISCX.csv`

### Dataset summary
- Original shape: approximately **692,703 rows × 79 columns**
- Cleaned shape: approximately **610,794 rows**
- Features after processing: **80**
- Target classes:
  - `BENIGN`
  - `DoS GoldenEye`
  - `DoS Hulk`
  - `DoS Slowhttptest`
  - `DoS slowloris`
  - `Heartbleed`

---

## Preprocessing Pipeline

The preprocessing workflow ensures consistency between training-time and inference-time transformations.

### Steps performed
1. **Column cleanup**
   - stripped column names
   - standardized input handling

2. **Duplicate removal**
   - removed duplicate records from the raw dataset

3. **Missing value handling**
   - filled missing values in `Flow Bytes/s` with `0`

4. **Infinite value handling**
   - replaced `+inf` and `-inf` values with valid numeric values

5. **Numeric coercion**
   - converted feature columns to numeric where applicable

6. **Feature engineering**
   - `Total Packets = Total Fwd Packets + Total Backward Packets`
   - `Packets per Second = Total Packets / (Flow Duration + 1)`

7. **Final numeric sanity checks**
   - revalidated numeric values
   - clipped extreme values
   - verified no non-finite values remained

8. **Label encoding**
   - encoded the target label for model training

9. **Feature scaling**
   - applied `StandardScaler`

---

## Machine Learning Pipeline

### Train/Test setup
- Train/Test split: **80/20**
- Stratified split: **Yes**
- Cross-validation: **5-fold**

### Models compared
1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

---

## Model Validation and Performance

### Logistic Regression
- CV Mean Accuracy: **0.9759**
- CV Std Dev: **0.0241**
- Test Accuracy: **0.9919**

### Random Forest
- CV Mean Accuracy: **0.9938**
- CV Std Dev: **0.0054**
- Test Accuracy: **0.9995**

### XGBoost
- CV Mean Accuracy: **0.9965**
- CV Std Dev: **0.0019**
- Test Accuracy: **0.9997**

### Final selected model
**XGBoost** was selected because it achieved:

- the **highest test accuracy**
- the **highest cross-validation mean**
- the **lowest cross-validation standard deviation**

This made it the strongest model in both **performance** and **stability** for tabular intrusion detection data.

---

## Why XGBoost Was Chosen

XGBoost was selected as the deployed model because:

- intrusion detection here is a **tabular classification problem**
- boosted trees are highly effective on structured numerical features
- it handled the feature patterns better than Logistic Regression
- it slightly outperformed Random Forest in both accuracy and CV stability
- it provided the best balance between **generalization** and **deployment readiness**

---

## Retrieval-Augmented Generation (RAG) Layer

The RAG layer enriches predictions with domain knowledge.

### Knowledge base
- Vector database: **ChromaDB**
- Collection: `network_security`

### Stored knowledge includes
- attack descriptions
- indicators
- impact context
- mitigation guidance

### Retrieval flow
1. Model predicts attack label
2. A retrieval query is generated dynamically
3. ChromaDB returns the most relevant cybersecurity context
4. The LLM or fallback agent uses that context to produce an incident summary

---

## LLM and Fallback Logic

The application uses an OpenAI-powered response generator for structured incident explanation.

### Behavior
- Primary path: **OpenAI response generation**
- Fallback path: activated when API fails, quota is unavailable, or generation errors occur

### Why fallback exists
This makes the app more reliable during:
- quota exhaustion
- deployment environments
- demo scenarios
- unstable API availability

The fallback response still provides:
- attack summary
- severity
- key indicators
- retrieved guidance
- recommended action

---

## Streamlit Application Features

The Streamlit dashboard provides a simplified analyst workflow.

### Current UI capabilities
- upload network traffic CSV
- run analysis
- view:
  - predicted attack
  - confidence
  - severity
  - rows processed
  - incident summary
  - technical details
- optional diagnostics panel
- clean minimal layout for presentation/demo use

---

## Project Architecture

### End-to-end flow
1. User uploads CSV in Streamlit
2. Data is read into a DataFrame
3. Preprocessing and feature engineering are applied
4. Saved model artifacts are loaded
5. XGBoost predicts the attack type
6. Confidence is calculated
7. Retrieval query is generated
8. ChromaDB retrieves relevant context
9. LLM or fallback agent generates the incident summary
10. Results are shown in the dashboard

---

## Folder Structure

```text
Network-Sentinel/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   └── Wednesday-workingHours.pcap_ISCX.csv
│   └── processed/
│
├── logs/
│   └── network_sentinel.log
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
│
├── rag/
│   ├── knowledge_base.py
│   ├── retriever.py
│   └── chroma_db/
│
└── src/
    ├── agent/
    │   ├── incident_agent.py
    │   └── llm_agent.py
    │
    ├── data/
    │   ├── loader.py
    │   └── preprocessing.py
    │
    ├── models/
    │   ├── train.py
    │   └── predict.py
    │
    └── utils/
        └── logger.py

```
## Saved Model Artifacts

The trained pipeline saves the following artifacts in `/models`:

- **model.pkl**  
  Trained XGBoost model used for inference  

- **scaler.pkl**  
  Saved `StandardScaler` used to transform features at inference time  

- **label_encoder.pkl**  
  Saved label encoder for converting model outputs back to original attack labels  

- **feature_columns.pkl**  
  Saved ordered list of training feature columns to ensure inference consistency  


---

## Logging

Application events are logged to:
logs/network_sentinel.log


### Logging covers
- artifact loading  
- preprocessing  
- feature transformation  
- prediction  
- retrieval  
- LLM fallback behavior  
- exception handling  

This makes debugging and deployment validation easier.


---

## How to Run the Project Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Network-Sentinel

### 2. Install dependencies
pip install -r requirements.txt

### Train the model
python -m src.models.train

### Run the Streamlit application
streamlit run app.py
```
## Limitations

- Currently summarizes overall input instead of full distribution analysis  
- RAG knowledge base is limited in size  
- LLM output depends on API availability (fallback mitigates this)  
- Evaluation mainly focuses on accuracy and CV stability  
  *(precision/recall/F1/confusion matrix can be expanded)*  


---

## Future Improvements

- Add batch-level attack distribution visualization  
- Highlight top suspicious rows  
- Include confusion matrix and classification report  
- Expand knowledge base for richer RAG responses  
- Add analyst follow-up query interface  
- Enable downloadable incident reports  
- Improve deployment environment and secret handling  


---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- scikit-learn  
- XGBoost  
- Streamlit  
- ChromaDB  
- OpenAI API  
- Joblib  

## Author

**Sharmila Murisetty**  
Data Analytics, Machine Learning, and BI Systems  
