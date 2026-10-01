# Smart-Credit-Automating-Financial-Risk-Brackets-with-Predictive-Machine-Learning-Models



Credit Score Classification API

A machine learning model that classifies a customer's credit score as Poor, Standard, or Good from their banking and credit data, served through a REST API built with FastAPI.

This project was built as a final certification assignment: acting as a data scientist at a global finance company, the goal is to turn raw credit-related records into a reliable credit score classifier.

Features
Data cleaning and feature preparation for messy real-world bank data
Comparison of several classifiers (Random Forest, XGBoost, LightGBM, Logistic Regression)
Model explainability with feature importance and SHAP
REST API for single and batch predictions, returning class probabilities
Interactive API docs (Swagger UI) out of the box
Optional Docker deployment
Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, FastAPI, Uvicorn

Project Structure
.
├── Model.ipynb        # Data cleaning, EDA, model comparison, explainability
├── main.py            # FastAPI application
├── requirements.txt   # Python dependencies
├── train.csv          # Training data (not included in the repo, see below)
├── test.csv           # Test data (not included in the repo, see below)
└── model.joblib       # Trained model, created on first run
The Model

The notebook walks through the full workflow:

Cleaning: removes junk characters and placeholder values, fixes impossible values, and fills gaps.
EDA: distributions by credit class, categorical breakdowns, and a correlation heatmap.
Modelling: class imbalance is handled with SMOTE, and four models are compared on validation macro F1 using a customer-grouped train/validation split so the same customer never appears in both sets.
Explainability: feature importance, SHAP values, and a confusion matrix show what drives each rating.

The model served by the API is a Logistic Regression classifier (C=0.1, balanced class weights) trained on the full training set with label-encoded categorical features and median imputation, as in the final cell of the notebook.

Model	Validation Macro F1
Add your results here	
Getting Started
Prerequisites
Python 3.12 or 3.13 recommended
train.csv placed in the project root
Installation
bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
Run the API
bash
uvicorn main:app --reload

On the first start the model is trained from train.csv and saved to model.joblib. Later starts load the saved model. Set RETRAIN=1 to force retraining.

The API runs at http://127.0.0.1:8000, and the interactive docs are at http://127.0.0.1:8000/docs.

API Reference
Method	Endpoint	Description
GET	/health	Check that the service and model are loaded
POST	/predict	Predict the credit score for one record
POST	/predict/batch	Predict for up to 1000 records at once

Fields you leave out are treated as missing and filled with the training median.

Example request
bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Month": "January",
    "Age": 28,
    "Occupation": "Scientist",
    "Annual_Income": 19114.12,
    "Monthly_Inhand_Salary": 1824.84,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Interest_Rate": 3,
    "Num_of_Loan": 4,
    "Delay_from_due_date": 3,
    "Num_of_Delayed_Payment": 7,
    "Changed_Credit_Limit": 11.27,
    "Num_Credit_Inquiries": 4,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 809.98,
    "Credit_Utilization_Ratio": 26.82,
    "Credit_History_Age": "22 Years and 9 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 49.57,
    "Amount_invested_monthly": 80.42,
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Monthly_Balance": 312.49
  }'
Example response
json
{
  "credit_score": "Standard",
  "class_id": 1,
  "probabilities": {
    "Poor": 0.1832,
    "Standard": 0.5127,
    "Good": 0.3041
  }
}

The probabilities above are illustrative. Real values depend on the trained model.

Docker
dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY main.py model.joblib ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

Run the app once locally first so model.joblib exists, then:

bash
docker build -t credit-score-api .
docker run -p 8000:8000 credit-score-api
Data

The dataset (train.csv, test.csv) contains customer-month banking records with fields such as income, debt, number of loans, payment delays, credit mix, and credit history age. The target is Credit_Score (Poor, Standard, Good). The data files are not included in this repository. Place them in the project root before running.

Notes and Limitations
The model is for learning and demonstration. It should not be used to make real lending decisions.
Categories not seen during training are encoded as unknown, and missing values are filled with the training median.
The model and the saved model.joblib should be built with the same scikit-learn version you deploy with.
License

Add a license of your choice (for example MIT) and state it here.

Author

Mohsin Mubarok

#Exploratory Data Analysis

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-9.png)
![alt text](image-10.png)
![alt text](image-11.png)
![alt text](image-12.png)
