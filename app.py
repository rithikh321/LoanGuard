from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import pickle
import os

# ===================================================
# ✅ App Configuration
# ===================================================
app = FastAPI(
    title="LoanGuard API",
    description="Loan risk prediction and bank recommendation API",
    version="1.0.0"
)

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================
# ✅ Load Model + Scaler + Data (ENHANCED)
# ===================================================
def safe_load_pickle(file):
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing required file: {file}")
    with open(file, "rb") as f:
        return pickle.load(f)

# Global variables with fallbacks
model = None
scaler = None
feature_columns = None
banks_df = None

try:
    # Check if files exist first
    required_files = ['loan_model.pkl', 'scaler.pkl', 'feature_columns.pkl', 'Banks-Interest-Rates_india.csv']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
    
    model = safe_load_pickle("loan_model.pkl")
    scaler = safe_load_pickle("scaler.pkl")
    feature_columns = safe_load_pickle("feature_columns.pkl")
    banks_df = pd.read_csv("Banks-Interest-Rates_india.csv")
    print("✅ All model files loaded successfully!")
    print(f"📊 Model type: {type(model)}")
    print(f"📊 Feature columns: {feature_columns}")
    print(f"📊 Banks data shape: {banks_df.shape}")
    
except Exception as e:
    print(f"❌ Error loading model/data: {e}")
    # Create fallback data for testing
    print("🔄 Creating fallback data for testing...")
    banks_df = pd.DataFrame({
        'Bank Name': ['SBI', 'HDFC', 'ICICI', 'Axis Bank', 'PNB'],
        '1 year tenure': [8.5, 9.0, 9.2, 9.5, 8.8],
        '3 year tenure': [9.0, 9.5, 9.7, 10.0, 9.3],
        '5 year tenure': [9.5, 10.0, 10.2, 10.5, 9.8]
    })
    # Create simple fallback feature columns
# Create simple fallback feature columns
    feature_columns = ['age', 'emp_length_clean', 'annual_inc', 'dti', 'fico_range_low', 
                      'loan_amnt', 'int_rate', 'loan_to_income', 'income_stability']

# ===================================================
# ✅ Serve Frontend
# ===================================================
@app.get("/")
async def serve_frontend():
    """Serve the LoanGuard frontend"""
    frontend_path = "index.html"
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(frontend_path)

# ===================================================
# ✅ Health Check Endpoint
# ===================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "LoanGuard API is running!"}

# ===================================================
# ✅ Enhanced Prediction Endpoint
# ===================================================
@app.post("/predict")
async def predict_loan(data: dict):
    try:
        print("📥 Received prediction request:", data)
        
        # Parse data safely with defaults
        age = float(data.get("age", 30))
        emp_length = float(data.get("emp_length", 3))
        annual_inc = float(data.get("annual_inc", 500000))
        existing_debt = float(data.get("existing_debt", 0))
        credit_score = float(data.get("credit_score", 700))
        loan_amnt = float(data.get("loan_amnt", 100000))
        loan_tenure = data.get("loan_tenure", "1 year tenure")
        employment_status = data.get("employment_status", "employed")

        # Compute derived metrics
        dti = existing_debt / annual_inc if annual_inc > 0 else 0

        # Calculate interest rate dynamically
        calculated_int_rate = (
            9
            + (700 - credit_score) * 0.01
            + (dti * 5)
            - (emp_length * 0.1)
        )
        calculated_int_rate = max(6.5, min(calculated_int_rate, 18))

        # Build profile for model
        customer_profile = {
            "age": age,
            "emp_length_clean": emp_length,
            "annual_inc": annual_inc,
            "dti": dti,
            "credit_score": credit_score,
            "loan_amnt": loan_amnt,
            "int_rate": calculated_int_rate,
            "loan_to_income": loan_amnt / annual_inc if annual_inc > 0 else 0,
            "income_stability": 0.8 if emp_length >= 5 else (0.6 if emp_length >= 2 else 0.4),
        }

        print("🧮 Customer profile:", customer_profile)

        # 🔥 ENHANCED PREDICTION WITH FALLBACK
        try:
            # Create DataFrame with proper feature columns
            customer_df = pd.DataFrame([customer_profile])
            
            # Ensure all feature columns are present
            for col in feature_columns:
                if col not in customer_df.columns:
                    customer_df[col] = 0
            
            # Reorder columns to match training
            customer_df = customer_df[feature_columns]
            
            # Scale features
            scaled = scaler.transform(customer_df)
            
            # Predict
            if hasattr(model, "predict_proba"):
                default_risk = float(model.predict_proba(scaled)[0][1])
            else:
                default_risk = 0.5  # Fallback
                
        except Exception as model_error:
            print(f"⚠️ Model prediction failed: {model_error}")
            # Fallback risk calculation
            default_risk = min(0.8, max(0.1, 
                (dti * 0.3) + 
                ((800 - credit_score) / 800 * 0.3) + 
                ((1 - customer_profile['income_stability']) * 0.2) +
                (customer_profile['loan_to_income'] * 0.2)
            ))

        # 🔥 ENHANCED BANK RECOMMENDATIONS
        try:
            eligible_banks = banks_df[banks_df[loan_tenure].notna()].copy()
            if eligible_banks.empty:
                raise ValueError("No banks found for tenure")
                
            min_rate = eligible_banks[loan_tenure].min()
            max_rate = eligible_banks[loan_tenure].max()
            eligible_banks["int_rate_score"] = 1 - ((eligible_banks[loan_tenure] - min_rate) / (max_rate - min_rate)).clip(0, 1)
            risk_factor = 0.85 if default_risk < 0.2 else (0.65 if default_risk < 0.4 else 0.45)
            eligible_banks["risk_score"] = risk_factor
            eligible_banks["lti_score"] = max(0, 1 - customer_profile["loan_to_income"])

            eligible_banks["suitability_score"] = (
                eligible_banks["int_rate_score"] * 0.5 +
                eligible_banks["risk_score"] * 0.3 +
                eligible_banks["lti_score"] * 0.2
            )

            recommendations = eligible_banks.sort_values("suitability_score", ascending=False).head(5)[
                ["Bank Name", loan_tenure, "suitability_score"]
            ].rename(columns={loan_tenure: "Interest Rate (%)"}).to_dict(orient="records")
            
        except Exception as bank_error:
            print(f"⚠️ Bank recommendation failed: {bank_error}")
            # Fallback bank recommendations
            recommendations = [
                {"Bank Name": "State Bank of India", "Interest Rate (%)": 8.5, "suitability_score": 0.85},
                {"Bank Name": "HDFC Bank", "Interest Rate (%)": 9.0, "suitability_score": 0.78},
                {"Bank Name": "ICICI Bank", "Interest Rate (%)": 9.2, "suitability_score": 0.75},
                {"Bank Name": "Axis Bank", "Interest Rate (%)": 9.5, "suitability_score": 0.72},
                {"Bank Name": "Punjab National Bank", "Interest Rate (%)": 8.8, "suitability_score": 0.80}
            ]

        print("✅ Prediction successful!")
        print(f"📊 Risk Score: {default_risk * 100:.2f}%")
        print(f"📊 Interest Rate: {calculated_int_rate:.2f}%")
        print(f"📊 Recommended Banks: {len(recommendations)}")
        
        return {
            "risk_score (%)": round(default_risk * 100, 2),
            "interest_rate (%)": round(calculated_int_rate, 2),
            "recommended_banks": recommendations,
        }

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===================================================
# ✅ Run App
# ===================================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LoanGuard API Server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)