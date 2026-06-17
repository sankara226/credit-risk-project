import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Credit Risk Framework Dashboard", layout="wide")

def load_warehouse_data() -> pd.DataFrame:
    db_file = Path(__file__).resolve().parent / "data" / "credit.db"
    if not db_file.exists():
        st.error(f"Localized system database workspace missing at {db_file}. Execute 'python main.py' to generate artifacts.")
        st.stop()
    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("SELECT * FROM credit_data", conn)
    return df

df = load_warehouse_data()

st.title("Credit Risk & Portfolio Delinquency Dashboard")
st.markdown("Automated diagnostic system analyzing default probabilities across balances, income tiers, and student cohorts.")
st.hr()

# KPI Metrics Metrics Layout
st.header("Corporate Portfolio High-Level KPIs")
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
kpi_col1.metric("Total Observed Credit Records", f"{len(df):,}")
kpi_col2.metric("Portfolio Default Rate (Arrears Factor)", f"{df['default'].mean()*100:.2f}%")
kpi_col3.metric("Active Student Roster Exposure Mix", f"{df['student'].mean()*100:.2f}%")
st.hr()

# Display Columns
v_col1, v_col2 = st.columns(2)

with v_col1:
    st.subheader("Balance Distribution Profile")
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df["balance"], kde=True, bins=40, color="royalblue", ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Default Distribution by Demographic Profile")
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x="student", y="default", data=df, ax=ax2, palette="Set1")
    ax2.set_xticklabels(["Non-Student Portfolio", "Student Allocation"])
    st.pyplot(fig2)

with v_col2:
    st.subheader("Income Distribution Profile")
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    sns.histplot(df["income"], kde=True, bins=40, color="emerald", ax=ax3)
    st.pyplot(fig3)
    
    st.subheader("Outstanding Balance vs Delinquency State")
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(x="default", y="balance", data=df, ax=ax4, palette="Set2")
    ax4.set_xticklabels(["Active Clear Ledgers", "Delinquent Defaulters"])
    st.pyplot(fig4)
