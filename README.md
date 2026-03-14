# Personality Prediction Using Text 🧠

A complete end-to-end machine learning project to predict whether a person is an **Introvert** or **Extrovert** based on their text input.

## Project Overview
This project uses classical Machine Learning models (Logistic Regression, Naive Bayes, SVM, and Random Forest) to classify text into personality types. It includes a full pipeline from data generation and preprocessing to model training and a web-based user interface.

## Project Structure
```
personality_prediction_project/
│
├── data/
│   └── personality_data.csv    # Synthetic dataset
├── notebooks/                  # Exploration notebooks
├── src/
│   ├── preprocessing.py        # Text cleaning and lemmatization
│   ├── train_model.py          # Model training script
│   └── predict.py              # Prediction interface
│
├── models/                     # Saved model files
├── app.py                      # Streamlit web application
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

## Setup and Installation

### 1. Prerequisites
- Python 3.8+ (Tested on Python 3.14)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Data & Train Models
If you want to retrain the models:
```bash
cd src
python train_model.py
```

### 4. Run the Web App
```bash
streamlit run app.py
```

## Features
- **Preprocessing Pipeline**: Cleaning, tokenization, stopword removal, and lemmatization.
- **Model Comparison**: Compare performance across different ML algorithms.
- **Interactive UI**: Real-time prediction with confidence scores.
- **Visualizations**: Word clouds and distribution charts.

## Deployment
### Streamlit Cloud
1. Push the code to a GitHub repository.
2. Connect the repository to [Streamlit Cloud](https://streamlit.io/cloud).
3. Set the main file path to `personality_prediction_project/app.py`.

### Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```
Build and run:
```bash
docker build -t personality-predictor .
docker run -p 8501:8501 personality-predictor
```

## Example Usage
- **Input**: "I love meeting new people and talking a lot." 
- **Output**: Extrovert (Confidence: 100%)
