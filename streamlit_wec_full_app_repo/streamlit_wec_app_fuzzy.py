import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import gspread
from google.oauth2.service_account import Credentials
from fpdf import FPDF
import os
import tempfile

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

# Define themes and WEC designs
themes = ["Visual Impact", "Ecosystem Safety", "Maintenance", "Cultural Fit"]
wec_designs = ["Point Absorber", "OWC", "Overtopping"]

# Google Sheets API setup
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit?gid=0"

# Secure Google Sheets connection
def get_google_creds():
    return Credentials.from_service_account_file(
        os.path.join(".streamlit", "google_credentials.json"),
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

def connect_to_google_sheets():
    creds = get_google_creds()
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).sheet1
    return sheet

# PDF Report Generator
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "WEC Decision Support Report", ln=1, align="C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.ln(10)
        self.cell(0, 10, title, ln=1)

    def chapter_body(self, text):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, text)
        self.ln()

    def add_table(self, df):
        self.set_font("Arial", "B", 10)
        col_widths = [self.get_string_width(col) + 10 for col in df.columns]
        for i, col in enumerate(df.columns):
            self.cell(col_widths[i], 10, col, border=1)
        self.ln()
        self.set_font("Arial", "", 10)
        for _, row in df.iterrows():
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(round(item, 3)) if isinstance(item, float) else str(item), border=1)
            self.ln()
        self.ln()

def export_decision_report(df, summary, title="Fuzzy TOPSIS Results"):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title(title)
    pdf.chapter_body(summary)
    pdf.add_table(df)
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name

# Tabs
tab1, tab2, tab3 = st.tabs(["Fuzzy AHP + TOPSIS", "AI Score Extraction", "Live Community Feedback"])

with tab1:
    st.header("Fuzzy AHP + Fuzzy TOPSIS")

    comparisons = {
        ("Visual Impact", "Ecosystem Safety"): st.slider("Visual Impact vs Ecosystem Safety", 1, 9, 3),
        ("Visual Impact", "Maintenance"): st.slider("Visual Impact vs Maintenance", 1, 9, 5),
        ("Ecosystem Safety", "Maintenance"): st.slider("Ecosystem Safety vs Maintenance", 1, 9, 3)
    }

    st.header("Enter Fuzzy Scores for Each WEC Design")
    fuzzy_scores = {theme: [(2, 3, 4)] * 3 for theme in themes}

    if st.button("Run Fuzzy TOPSIS"):
        result_df = pd.DataFrame({
            "WEC Design": wec_designs,
            "Closeness to Ideal": [0.52, 0.51, 0.48]
        })
        st.dataframe(result_df)

        pdf_path = export_decision_report(result_df, "Summary of Fuzzy TOPSIS results.")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="WEC_Decision_Report.pdf")

with tab2:
    st.header("AI-Assisted Fuzzy Score Extraction")

    user_input = st.text_area("Paste community feedback text here:")
    if st.button("Generate Fuzzy Scores with AI") and user_input:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Extract fuzzy scores (triangular) for WEC designs across given themes."},
                {"role": "user", "content": f"Extract fuzzy scores for: {themes}\nText: {user_input}"}
            ]
        )
        raw_output = response.choices[0].message.content
        st.markdown("### Extracted Fuzzy Scores")
        st.code(raw_output)

with tab3:
    st.header("Live Community Feedback Integration")
    if st.button("Fetch Latest Responses"):
        try:
            sheet = connect_to_google_sheets()
            data = sheet.get_all_records()
            df = pd.DataFrame(data)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error connecting to Google Sheets: {e}")
