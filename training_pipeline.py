import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_loan_model(customer_csv, banks_csv):
    # Step 1: Load and preprocess
    df = pd.read_csv(customer_csv)
    banks_df = pd.read_csv(banks_csv)
    # Simplify: Keep relevant cols
    cols = ['age', 'emp_length', 'annual_inc', 'dti', 'fico_range_low', 'loan_amnt', 'term', 'int_rate', 'loan_status']
    df = df[cols].copy()
    df.fillna(df.median(), inplace=True)
    df['is_default'] = (df['loan_status'] == 1).astype(int)
    # Feature engineering
    df['emp_length_clean'] = df['emp_length'].str.extract('(\d+)', expand=False).fillna(0).astype(int)
    df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
    df['income_stability'] = np.where(df['emp_length_clean'] >= 5, 0.8,
                                      np.where(df['emp_length_clean'] >= 2, 0.6, 0.4))
    
    features = ['age', 'emp_length_clean', 'annual_inc', 'dti', 'fico_range_low',
                'loan_amnt', 'int_rate', 'loan_to_income', 'income_stability']
    X = df[features]
    y = df['is_default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("✅ Model trained and saved!")

if __name__ == "__main__":
    train_loan_model("loans_dataset_500_records_INR.csv", "Banks-Interest-Rates_india.csv")