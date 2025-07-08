# Fraud Transaction Detection System

This project demonstrates a machine learning pipeline for detecting fraudulent transactions based on a simulated dataset. The system incorporates feature engineering tailored to specific fraud scenarios and employs common classification models.

## Project Structure:

- `data/`: Contains the simulated transaction dataset.
- `scripts/`: Python scripts for data generation, feature engineering, model training, and prediction.
- `models/`: Stores the trained machine learning models and scaler.

## Fraud Scenarios Simulated:

1.  **Amount-based Fraud:** Transactions with amounts greater than 220 are flagged as fraud.
2.  **Compromised Terminals:** Two random terminals are compromised daily. All transactions on these terminals for the next 28 days are fraudulent.
3.  **Compromised Customers:** Three random customers are compromised daily. For the next 14 days, 1/3 of their transactions have their amounts multiplied by 5 and are marked as fraudulent.

## How to Run:

1.  **Clone this repository** (or create the file structure manually).
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Generate the dataset:**
    ```bash
    python scripts/generate_data.py
    ```
    This will create `data/transactions_data.csv`.
4.  **Perform Feature Engineering and Train Models:**
    ```bash
    python scripts/train_model.py
    ```
    This script will load the data, engineer features, train RandomForest and GradientBoosting models, evaluate them, and save the models (`.pkl` files) into the `models/` directory.
5.  **Make Predictions on New Data (Conceptual):**
    You can adapt `scripts/predict_new_transaction.py` to simulate new transactions and test the deployed model. This requires careful handling of historical features.

## Files Description:

-   **`generate_data.py`**: Contains the `generate_simulated_data` function to create the synthetic transaction dataset based on the specified fraud rules.
-   **`feature_engineering.py`**: (Can be merged into `train_model.py` for simplicity, or kept separate for modularity). Contains the `feature_engineering` function which creates new features like rolling averages and counts for customers and terminals.
-   **`train_model.py`**: Orchestrates data loading, calls feature engineering, splits data, trains RandomForest and GradientBoosting classifiers, evaluates their performance, and saves the trained models and the StandardScaler.
-   **`predict_new_transaction.py`**: An example script demonstrating how to load a saved model and scaler, process new (unseen) transaction data, and make fraud predictions.