import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from scripts.feature_engineering import feature_engineering # Import the function

def predict_new_transaction(new_tx_data):
    """
    Predicts if a new transaction is fraudulent.
    new_tx_data: A dictionary containing details of a new transaction.
                 Example: {'TRANSACTION_ID': 'TX123456', 'TX_DATETIME': '2023-02-15 10:30:00',
                           'CUSTOMER_ID': 'C0001', 'TERMINAL_ID': 'T0005', 'TX_AMOUNT': 150.00}
    """
    print("\n--- Predicting New Transaction ---")
    # Load the trained model and scaler
    try:
        loaded_rf_model = joblib.load('models/random_forest_model.pkl')
        loaded_scaler = joblib.load('models/scaler.pkl')
        print("Models and scaler loaded successfully.")
    except FileNotFoundError:
        print("Error: Models not found. Please run train_model.py first.")
        return

    # Convert new transaction data to DataFrame
    new_tx_df = pd.DataFrame([new_tx_data])
    new_tx_df['TX_DATETIME'] = pd.to_datetime(new_tx_df['TX_DATETIME'])

    # --- Important Note for Real-time Prediction ---
    # For a true real-time prediction, the feature_engineering function
    # would need access to historical data for the specific customer and terminal.
    # In this simplified example, we're assuming 'new_tx_df' conceptually contains
    # enough data to compute these features, or that the feature engineering
    # function is adapted to work with external historical data stores.
    # For demonstration, we'll append to a dummy historical dataframe.

    # This part is a simplification. In reality, you'd query a feature store
    # for the customer's/terminal's recent history.
    # For this script to run, let's create a minimal history, or you can load
    # the full training data again for a simplified demonstration.
    try:
        # Load the full historical data to ensure rolling windows can compute
        historical_df = pd.read_csv('data/transactions_data.csv')
        historical_df['TX_DATETIME'] = pd.to_datetime(historical_df['TX_DATETIME'])
        # Append the new transaction to history for feature calculation
        combined_df = pd.concat([historical_df, new_tx_df], ignore_index=True)
        combined_df = combined_df.sort_values(by='TX_DATETIME').reset_index(drop=True)
    except FileNotFoundError:
        print("Warning: Historical data not found. Feature engineering will be limited for single transaction.")
        combined_df = new_tx_df.copy()

    # Apply feature engineering to the combined (or just new) data
    df_with_features = feature_engineering(combined_df)

    # Extract features for the *new* transaction (the last row after sorting)
    # Ensure the feature names match those used during training
    features_for_prediction = df_with_features.iloc[[-1]].drop(
        columns=['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_FRAUD'], errors='ignore'
    )

    # Handle potential inf/NaN if the single transaction scenario caused them
    features_for_prediction = features_for_prediction.replace([np.inf, -np.inf], np.nan)
    features_for_prediction = features_for_prediction.fillna(0)


    # Scale the features
    scaled_features = loaded_scaler.transform(features_for_prediction)

    # Make prediction
    fraud_probability = loaded_rf_model.predict_proba(scaled_features)[:, 1]
    is_fraud = (fraud_probability[0] > 0.5) # Default threshold 0.5

    print(f"Transaction ID: {new_tx_data['TRANSACTION_ID']}")
    print(f"Predicted Fraud Probability: {fraud_probability[0]:.4f}")
    if is_fraud:
        print("Prediction: POTENTIAL FRAUD (Flagged for Review)")
    else:
        print("Prediction: Legitimate Transaction")

if __name__ == "__main__":
    # Example usage:
    example_new_transaction_1 = {
        'TRANSACTION_ID': 'NEW_TX_001',
        'TX_DATETIME': '2023-03-01 10:05:00',
        'CUSTOMER_ID': 'C0001', # Customer who might have a history of normal transactions
        'TERMINAL_ID': 'T0005',
        'TX_AMOUNT': 300.00, # Example of amount > 220
        'TX_FRAUD': 0 # We don't know the true label yet for new transactions
    }
    predict_new_transaction(example_new_transaction_1)

    example_new_transaction_2 = {
        'TRANSACTION_ID': 'NEW_TX_002',
        'TX_DATETIME': '2023-03-01 11:15:00',
        'CUSTOMER_ID': 'C0050',
        'TERMINAL_ID': 'T0002',
        'TX_AMOUNT': 50.00, # Normal amount
        'TX_FRAUD': 0
    }
    predict_new_transaction(example_new_transaction_2)

    # Example simulating a potentially compromised customer's high value transaction
    example_new_transaction_3 = {
        'TRANSACTION_ID': 'NEW_TX_003',
        'TX_DATETIME': '2023-03-01 12:30:00',
        'CUSTOMER_ID': 'C0123', # Assuming C0123 had recent normal transactions
        'TERMINAL_ID': 'T0010',
        'TX_AMOUNT': 750.00, # A very high amount for a potentially compromised customer
        'TX_FRAUD': 0
    }
    predict_new_transaction(example_new_transaction_3)