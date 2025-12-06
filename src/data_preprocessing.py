import pandas as pd
import numpy as np

def load_data(filepath):
    """CSV faylı oxu."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Data cleaning: TotalCharges, encoding, target."""
    # 1. TotalCharges: ' ' → NaN, sonra float-a çevir
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 2. NaN dəyərləri median ilə doldur
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # 3. Churn: 'Yes' → 1, 'No' → 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. ID və useless sütunları sil
    df.drop(['customerID'], axis=1, inplace=True)
    
    return df

def encode_features(df):
    """Kategorik dəyişənləri kodla."""
    # Binary columns (Yes/No)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-Hot Encoding üçün qalan kategoriklər
    categorical_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod', 'gender'
    ]
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df