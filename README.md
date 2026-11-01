# Banking77 Customer Query Intent Classification

## Project Overview
This project uses a fine-tuned DistilBERT model to classify banking customer queries into 77 different intent categories.

The application allows users to enter a banking-related query and predicts the customer's intent along with a confidence score.

## Dataset
- **Name:** Banking77
- **Source:** [PolyAI Banking77 Dataset](https://huggingface.co/datasets/PolyAI/banking77)
- **Training examples:** 10,003
- **Testing examples:** 3,080
- **Categories:** 77
- **Columns:** Text and Label

The dataset contains pre-labelled customer banking queries.

## Pretrained Model
- **Model:** DistilBERT Base Uncased
- **Source:** [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased)
- **Fine-tuning:** Full Fine-Tuning
- **Epochs:** 3

## Fine-Tuned Model
[View My Hugging Face Model](https://huggingface.co/hafeezaismail/banking77-distilbert-intent-classifier)

## Performance
The model achieved **84.64% accuracy** on the Banking77 test split evaluation.

## Application Features
- Enter a banking customer query.
- Predict one of 77 customer intent categories.
- Display prediction confidence.
- Simple and interactive Streamlit interface.

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Banking77 Dataset
- Streamlit

## Repository Files
- `app.py` – Streamlit application code.
- `requirements.txt` – Required Python libraries.
- `README.md` – Project documentation.

## Run Locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Author
Hafeeza Ismail
