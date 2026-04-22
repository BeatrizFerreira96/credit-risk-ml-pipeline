import pandas as pd
import sqlite3

# Load raw dataset
df = pd.read_csv("data/raw/credit_data.csv", sep=" ", header=None)

# Add column names (simplified)
df.columns = [f"feature_{i}" for i in range(20)] + ["target"]

# Convert target (1 = good, 2 = bad → make 0/1)
df["target"] = df["target"].map({1: 0, 2: 1})

# Save to SQLite
conn = sqlite3.connect("data/credit.db")
df.to_sql("credit_data", conn, if_exists="replace", index=False)

conn.close()

print("Data loaded into SQL database!")