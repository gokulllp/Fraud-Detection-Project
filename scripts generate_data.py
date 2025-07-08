import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_simulated_data(num_transactions=100000, num_customers=5000, num_terminals=1000, start_date='2023-01-01'):
    print("Generating simulated data...")
    transactions = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')

    customer_ids = [f'C{i:04d}' for i in range(num_customers)]
    terminal_ids = [f'T{i:04d}' for i in range(num_terminals)]

    compromised_terminals_data = {}
    compromised_customers_data = {}

    for i in range(num_transactions):
        transaction_id = f'TX{i:06d}'
        # Ensure increasing time for rolling window features
        tx_datetime = current_date + timedelta(minutes=random.randint(0, 5))
        current_date = tx_datetime

        customer_id = random.choice(customer_ids)
        terminal_id = random.choice(terminal_ids)
        tx_amount = round(random.uniform(10, 300), 2)
        tx_fraud = 0

        # Scenario 1: Amount > 220 is fraud
        if tx_amount > 220:
            tx_fraud = 1

        # Scenario 2: Compromised Terminals
        day_key = tx_datetime.date()
        if day_key not in compromised_terminals_data:
            # Draw new compromised terminals once per day
            compromised_terminals_data[day_key] = random.sample(terminal_ids, 2)

        for draw_date, term_list in compromised_terminals_data.items():
            if draw_date <= day_key < draw_date + timedelta(days=28) and terminal_id in term_list:
                tx_fraud = 1
                break

        # Scenario 3: Compromised Customers
        if day_key not in compromised_customers_data:
            # Draw new compromised customers once per day
            compromised_customers_data[day_key] = random.sample(customer_ids, 3)

        for draw_date, cust_list in compromised_customers_data.items():
            if draw_date <= day_key < draw_date + timedelta(days=14) and customer_id in cust_list:
                if random.random() < (1/3):
                    tx_amount *= 5 # Amount multiplied by 5
                    tx_fraud = 1
                break

        transactions.append({
            'TRANSACTION_ID': transaction_id,
            'TX_DATETIME': tx_datetime,
            'CUSTOMER_ID': customer_id,
            'TERMINAL_ID': terminal_id,
            'TX_AMOUNT': tx_amount,
            'TX_FRAUD': tx_fraud
        })

    df = pd.DataFrame(transactions)
    print(f"Generated {len(df)} transactions.")
    print(df['TX_FRAUD'].value_counts())

    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions_data.csv', index=False)
    print("Data saved to data/transactions_data.csv")
    return df

if __name__ == "__main__":
    generate_simulated_data()