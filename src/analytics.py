import sqlite3
import pandas as pd
from pathlib import Path

def process_credit_chunks(db_path: Path, chunk_size: int = 2000) -> list[dict]:
    """Processes historical records out of SQL using memory-safe offset chunks."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM credit_data")
        total_rows = cursor.fetchone()[0]
        
        results = []
        for offset in range(0, total_rows, chunk_size):
            query = f"SELECT * FROM credit_data LIMIT {chunk_size} OFFSET {offset}"
            df_chunk = pd.read_sql_query(query, conn)
            
            results.append({
                "chunk": (offset // chunk_size) + 1,
                "mean_balance": float(df_chunk["balance"].mean()),
                "mean_income": float(df_chunk["income"].mean()),
                "default_rate": float(df_chunk["default"].mean()),
                "student_rate": float(df_chunk["student"].mean()),
                "rows": len(df_chunk)
            })
    return results

def aggregate_credit_chunks(results: list[dict]) -> dict:
    """Assembles operational chunk dictionaries back into unified global parameters."""
    df = pd.DataFrame(results)
    return {
        "global_mean_balance": float(df["mean_balance"].mean()),
        "global_mean_income": float(df["mean_income"].mean()),
        "global_default_rate": float(df["default_rate"].mean()),
        "global_student_rate": float(df["student_rate"].mean()),
        "total_rows_processed": int(df["rows"].sum()),
        "chunks_processed": len(df)
    }

def generate_credit_insights(df: pd.DataFrame) -> dict:
    """Generates explicit macro groupings for financial executive summary matrices."""
    return {
        "default_rate": float(df["default"].mean()),
        "default_rate_by_student": df.groupby("student")["default"].mean().to_dict(),
        "average_balance": float(df["balance"].mean()),
        "average_income": float(df["income"].mean()),
        "balance_by_default": df.groupby("default")["balance"].mean().to_dict(),
        "income_by_default": df.groupby("default")["income"].mean().to_dict(),
        "correlation_with_default": df.select_dtypes(include='number').corr()["default"].sort_values(ascending=False).to_dict()
    }