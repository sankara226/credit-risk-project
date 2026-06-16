import datetime
from pathlib import Path
from fpdf import FPDF
from src.pipeline_etl import extract_credit_data, transform_credit_data, load_credit_to_sqlite
from src.analytics import process_credit_chunks, aggregate_credit_chunks, generate_credit_insights
from src.model import train_credit_model
from src.plots import generate_report_plots, generate_model_plots

class PortfolioReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 10, "Risk Operations Automated Verification Registry", ln=True, border="B")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} | Corporate Analytical Assets Portfolio", align="C")

def build_pdf_dashboard(insights: dict, model_res: dict, plots: dict, model_plots: dict, output_path: Path):
    """Generates an executive-ready auditing PDF framework profile reporting validation metrics."""
    pdf = PortfolioReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Cover Sheet Abstract Section
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 15, "Credit Risk Exposure & Validation Audit Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 5, f"Analysis Pipeline Compilation Date: {datetime.date.today()}", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "This automated report tracks the operational credit parameters of the data core. "
                         "By running stratified lookups across balances and income brackets, we build risk evaluation metrics.")
    
    # KPI Grid Append
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Macro Profile Aggregates Summary", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Global Ledger Arrears Factor (Default Rate) : {insights['default_rate']*100:.2f}%", ln=True)
    pdf.cell(0, 7, f"Cohort Student Risk Allocation Factor     : {insights['default_rate_by_student'].get(1,0)*100:.2f}%", ln=True)
    pdf.cell(0, 7, f"Average Outstanding Portfolio Balances    : ${insights['average_balance']:,.2f}", ln=True)
    
    # Model Assessment Layer Append
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Predictive Risk Modeling Performance (Test Matrix Summary)", ln=True)
    pdf.ln(3)
    
    m = model_res["metrics"]
    pdf.cell(0, 6, f" - Classification Precision Level : {m['precision']:.3f}", ln=True)
    pdf.cell(0, 6, f" - Classification Recall Rate       : {m['recall']:.3f}", ln=True)
    pdf.cell(0, 6, f" - Strategic ROC-AUC Performance    : {m['roc_auc']:.3f}", ln=True)
    
    pdf.ln(5)
    pdf.image(model_plots["confusion_matrix"], x=15, w=90)
    pdf.image(model_plots["roc_curve"], x=110, y=38, w=90)
    
    pdf.output(str(output_path))

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "data" / "credit.db"
    OUT_DIR = BASE_DIR / "outputs"
    
    print("[1/6] Extracting data from source registries...")
    df_raw = extract_credit_data()
    
    print("[2/6] Running data clean-mapping profiles...")
    df_clean = transform_credit_data(df_raw)
    
    print(f"[3/6] Streaming transactional matrices straight into warehouse: {DB_PATH}")
    load_credit_to_sqlite(df_clean, DB_PATH)
    
    print("[4/6] Parsing partition layers via chunk allocations...")
    chunks = process_credit_chunks(DB_PATH)
    global_aggregates = aggregate_credit_chunks(chunks)
    insights = generate_credit_insights(df_clean)
    
    print("[5/6] Executing Logistic Regression validation sequences...")
    model_results = train_credit_model(df_clean)
    
    print("[6/6] Packaging visual charts and assembling executive PDF documents...")
    report_plots = generate_report_plots(df_clean, OUT_DIR / "report_plots")
    model_plots = generate_model_plots(model_results, OUT_DIR / "model_plots")
    
    pdf_out = OUT_DIR / "credit_risk_report.pdf"
    build_pdf_dashboard(insights, model_results, report_plots, model_plots, pdf_out)
    print(f"\n[✓] Operational automation run success. PDF safely stored at: {pdf_out}")

if __name__ == "__main__":
    main()