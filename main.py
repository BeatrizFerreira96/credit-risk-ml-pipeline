from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()   # ✅ MUST come before decorators
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# load model
model = joblib.load("src/model.pkl")

# input schema
class Customer(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

@app.get("/")
def root():
    return {"message": "Credit Risk ML API"}

@app.post("/predict")
def predict(customer: Customer):

    data = pd.DataFrame([customer.dict()])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    return {
        "prediction": "high risk" if pred == 1 else "low risk",
        "risk_probability": round(float(prob), 3),
        "confidence": round(float(prob if pred == 1 else 1 - prob), 3)
    }