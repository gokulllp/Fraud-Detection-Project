import pandas as pd

def feature_engineering(df):
    print("Performing feature engineering...")
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_DATE'] = df['TX_DATETIME'].dt.date
    df['TX_TIME_OF_DAY'] = df['TX_DATETIME'].dt.hour

    # Scenario 1: Amount-based feature
    df['AMOUNT_GT_220'] = (df['TX_AMOUNT'] > 220).astype(int)

    # Sort by datetime for correct rolling window calculations
    df = df.sort_values(by='TX_DATETIME').reset_index(drop=True)

    # Scenario 2: Terminal-based features
    # Number of fraudulent transactions on a terminal in the last 7 days
    # Use apply with lambda for rolling to handle groupby correctly
    df['TERMINAL_FRAUD_COUNT_7D'] = df.groupby('TERMINAL_ID').apply(
        lambda x: x['TX_FRAUD'].rolling(window='7D', on='TX_DATETIME', closed='left').sum()
    ).reset_index(level=0, drop=True)
    df['TERMINAL_FRAUD_COUNT_7D'] = df['TERMINAL_FRAUD_COUNT_7D'].fillna(0)

    # Scenario 3: Customer-based features
    # Average spending of a customer in the last 7 days
    df['CUSTOMER_AVG_AMOUNT_7D'] = df.groupby('CUSTOMER_ID').apply(
        lambda x: x['TX_AMOUNT'].rolling(window='7D', on='TX_DATETIME', closed='left').mean()
    ).reset_index(level=0, drop=True)
    df['CUSTOMER_AVG_AMOUNT_7D'] = df['CUSTOMER_AVG_AMOUNT_7D'].fillna(df['TX_AMOUNT']) # Fill initial NaNs with current amount

    # Spending deviation from customer's average
    df['CUSTOMER_AMOUNT_DEVIATION'] = df['TX_AMOUNT'] / df['CUSTOMER_AVG_AMOUNT_7D']
    # Handle cases where CUSTOMER_AVG_AMOUNT_7D might be zero, leading to inf
    df['CUSTOMER_AMOUNT_DEVIATION'] = df['CUSTOMER_AMOUNT_DEVIATION'].replace([float('inf'), -float('inf')], 0)


    # Number of transactions by customer in last 7 days
    df['CUSTOMER_TX_COUNT_7D'] = df.groupby('CUSTOMER_ID').apply(
        lambda x: x['TX_AMOUNT'].rolling(window='7D', on='TX_DATETIME', closed='left').count()
    ).reset_index(level=0, drop=True)
    df['CUSTOMER_TX_COUNT_7D'] = df['CUSTOMER_TX_COUNT_7D'].fillna(0)


    print("Feature engineering complete.")
    return df.drop(columns=['TX_DATE']) # Drop the intermediate date column