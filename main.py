from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional

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
class InputData(BaseModel):
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    age: int
    existing_credits: int

    # optional fields
    personal_status: Optional[str] = None

@app.get("/")
def root():
    import os
    print("Serving file from:", os.getcwd())
    return FileResponse(os.path.join(os.getcwd(), "index.html"))


@app.post("/predict")
def predict(customer: InputData):

    data = pd.DataFrame([customer.dict()])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    return {
        "prediction": "high risk" if pred == 1 else "low risk",
        "risk_probability": round(float(prob), 3),
        "confidence": round(float(prob if pred == 1 else 1 - prob), 3)
    }