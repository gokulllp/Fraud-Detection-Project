import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import os
from scripts.feature_engineering import feature_engineering # Import the function

def train_and_evaluate():
    print("Starting model training and evaluation...")
    # Load data
    try:
        df = pd.read_csv('data/transactions_data.csv')
    except FileNotFoundError:
        print("Error: data/transactions_data.csv not found. Please run generate_data.py first.")
        return

    # Apply feature engineering
    df_features = feature_engineering(df.copy())

    # Prepare data for modeling
    X = df_features.drop(columns=['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD'])
    y = df_features['TX_FRAUD']

    # Handle any remaining infinite or NaN values from feature engineering
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0) # Simple imputation, consider more advanced methods for real data

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Fraud in train set: {y_train.sum() / len(y_train) * 100:.2f}%")
    print(f"Fraud in test set: {y_test.sum() / len(y_test) * 100:.2f}%")

    # --- Model Training ---

    # RandomForestClassifier
    print("\nTraining RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("RandomForestClassifier trained.")

    # GradientBoostingClassifier
    print("\nTraining GradientBoostingClassifier...")
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train, y_train)
    print("GradientBoostingClassifier trained.")

    # --- Evaluation ---
    def evaluate_model(model, X_test, y_test, model_name):
        print(f"\n--- {model_name} Evaluation ---")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        print(f"Precision-Recall AUC: {pr_auc:.4f}")

    evaluate_model(rf_model, X_test, y_test, "RandomForestClassifier")
    evaluate_model(gb_model, X_test, y_test, "GradientBoostingClassifier")

    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(gb_model, 'models/gradient_boosting_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nModels and scaler saved successfully to 'models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()