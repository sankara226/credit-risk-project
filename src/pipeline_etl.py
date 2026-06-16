import os
import sqlite3
import pandas as pd
import kagglehub
from pathlib import Path

def extract_credit_data(dataset_name: str = "creepycrap/finance-dataset") -> pd.DataFrame:
    """Downloads repository data via KaggleHub and loads into a standardized Pandas DataFrame."""
    # Attempt to download dataset via kagglehub
    path = kagglehub.dataset_download(dataset_name)
    files = os.listdir(path)
    csv_files = [f for f in files if f.lower().endswith('.csv')]

    # If a local CSV exists in the project data folder, prefer it (helpful for offline runs)
    local_csv = Path(__file__).resolve().parent.parent / 'data' / 'credit.csv'
    if local_csv.exists():
        df = pd.read_csv(local_csv)
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    # If Kaggle download produced CSVs, load the first one
    if csv_files:
        csv_path = Path(path) / csv_files[0]
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        return df

    # Support Excel files (.xls, .xlsx) commonly used in some datasets
    excel_files = [f for f in files if f.lower().endswith(('.xls', '.xlsx', '.xlsm'))]
    if excel_files:
        excel_path = Path(path) / excel_files[0]
        df = pd.read_excel(excel_path, sheet_name=0)
        # drop index-like unnamed columns often present in Excel exports
        df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith('unnamed')]]
        df.columns = [c.lower().strip() for c in df.columns]

        # Validate expected schema for this ETL (it's built specifically for the finance dataset)
        required = {"default", "student", "balance", "income"}
        present = set(df.columns)
        if not required.issubset(present):
            raise ValueError(
                f"Dataset schema mismatch: expected columns {sorted(required)}, "
                f"found {sorted(present)}. Ensure you're using the '{dataset_name}' dataset or provide a compatible CSV at 'data/credit.csv'."
            )

        return df

    # No CSV available: provide a clear diagnostic error and stop the ETL
    available = ', '.join(files)
    raise FileNotFoundError(
        f"No CSV found in Kaggle dataset '{dataset_name}'. Downloaded cache path: {path}. "
        f"Available files: {available}.\nPlease ensure the dataset contains at least one .csv file, or place a CSV at '{local_csv}'."
    )
    df.columns = [c.lower().strip() for c in df.columns]
    return df

def transform_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes target columns, maps discrete categories to flags, and builds analytics indicators."""
    df_clean = df.copy()
    
    # Standardize string maps to binary flags
    for col in ["default", "student"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map({"Yes": 1, "No": 0})
            
    # Cast numerical primitives cleanly
    for col in ["balance", "income"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            
    df_clean = df_clean.dropna().reset_index(drop=True)
    
    if "balance" in df_clean.columns and "income" in df_clean.columns:
        df_clean["balance_to_income"] = df_clean["balance"] / df_clean["income"]
        
    return df_clean

def load_credit_to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str = "credit_data") -> bool:
    """Streams the cleaned dataset into a localized structural relational SQLite cluster database."""
    db_path.parent.mkdir(exist_ok=True)
    try:
        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        return True
    except Exception as e:
        print(f"[ERROR] Warehouse upload failed for table {table_name}: {e}")
        return False