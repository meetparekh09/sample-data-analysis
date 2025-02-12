import pandas as pd
import random
from datetime import datetime, timedelta

# Parameters for the dataset
num_records = 17000
branches = ['A', 'B']
customer_types = ['Member', 'Normal']
payments = ['Cash', 'Credit Card', 'E-wallet']

# Generate random data
data = {
    'Date': [(datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d') for _ in range(num_records)],
    'Branch': [random.choice(branches) for _ in range(num_records)],
    'Customer_Type': [random.choice(customer_types) for _ in range(num_records)],
    'Payment': [random.choice(payments) for _ in range(num_records)],
    'Total': [round(random.uniform(10, 500), 2) for _ in range(num_records)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('supermarket_sales.csv', index=False)

print("Synthetic supermarket sales dataset generated and saved as 'supermarket_sales.csv'.")