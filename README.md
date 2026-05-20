# Titanic Survival Prediction - ML Deployment Project

## Project Overview

This project is an end-to-end Machine Learning Deployment project using the Titanic dataset.

The model predicts whether a passenger survived or not using:
- Support Vector Machine (SVM)
- Flask API
- Docker containerization
- AWS deployment workflow

---

# Tech Stack

- Python
- Scikit-learn
- Flask
- Docker
- Git & GitHub
- AWS EC2

---

# Machine Learning Workflow

1. Data preprocessing
2. Missing value handling
3. Feature engineering
4. Model training using SVM
5. Model evaluation
6. Model serialization using Pickle
7. Flask API creation
8. Docker containerization
9. Cloud deployment on AWS EC2

---

# Project Structure

```bash
titanic-svm/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── model.pkl
├── scaler.pkl
├── train_model.py
├── train.csv
├── test.csv
└── README.md
```

---

# Model Performance

- Algorithm: Support Vector Machine (SVM)
- Accuracy: ~82%
- Evaluation Metrics:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

---

# Docker Commands

## Build Docker Image

```bash
docker build -t titanic-svm .
```

## Run Docker Container

```bash
docker run -p 5000:5000 titanic-svm
```

---

# Flask API Endpoint

## Home Route

```bash
GET /
```

## Prediction Route

```bash
POST /predict
```

### Sample JSON Input

```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 25,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 100
}
```

### Sample Output

```json
{
  "prediction": "Survived"
}
```

---

# AWS Deployment Workflow

1. Launch EC2 instance
2. Install Docker
3. Clone GitHub repository
4. Build Docker image
5. Run Docker container
6. Access public API using EC2 Public IP

---

# Future Improvements

- Add frontend UI
- Use FastAPI
- Add SHAP explainability
- CI/CD pipeline
- Kubernetes deployment
- Model monitoring

---

# Author

Gayatri P3