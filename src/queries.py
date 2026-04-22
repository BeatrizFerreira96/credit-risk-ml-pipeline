import sqlite3
import pandas as pd
from pathlib import Path

# project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "credit.db"

def get_credit_data():
    conn = sqlite3.connect(DB_PATH)

    query = "SELECT * FROM credit_data"

    df = pd.read_sql(query, conn)
    conn.close()

    return df