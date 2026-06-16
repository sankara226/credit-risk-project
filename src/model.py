import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def train_credit_model(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Extracts features, normalizes scales, and fits a high-performance predictive risk engine."""
    features = ["balance", "income"]
    if "balance_to_income" in df.columns:
        features.append("balance_to_income")  # Fixed your original syntax hyphen bug
        
    X = df[features].copy()
    y = df["default"].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )  # Fixed your missing comma bug
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Fixed your missing predictive variable assignment bugs
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": features,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }