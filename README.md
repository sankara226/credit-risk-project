# High-Scale Credit Risk ETL Pipeline & Classification Dashboard

This project is an asynchronous, production-grade credit risk modeling and portfolio data engineering framework. This system extracts financial transaction datasets from Kaggle Hub, maps string categories into binary indicator tags, tracks execution memory footprints via chunk-based processing matrices, and builds machine learning predictive models alongside live data visualization channels.

## Operational Pipeline Architecture

1. **ETL ingestion Engine:** Download mechanisms via `kagglehub` pipeline extraction protocols.
2. **Data Warehouse Layer:** Memory-isolated validation loops using programmatic SQLite data streaming protocols.
3. **Machine Learning Model Engine:** Normalizes metrics using a standard scaler and fits a predictive risk Logistic Regression classifier to evaluate customer default risk metrics.
4. **Interface Serving Core:** Real-time analytics charts serving through a local interactive `streamlit` dashboard server interface.

## Execution Sequence Setup

### 1. Pin System Virtual Packages Environment

```bash
pip install -r requirements.txt
```
