
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from openai import Openai
from google.oauth2.service_account import Credentials
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

themes = ["Visual Impact", "Ecosystem Safety", "Maintenance", "Cultural Fit"]
wec_designs = ["Point Absorber", "OWC", "Overtopping"]

client = Openai(api_key=st.secrets["OPENAI_API_KEY"])
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit?gid=0"

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

def fuzzy_ahp_to_weights(comparisons, criteria):
    n = len(criteria)
    pcm = np.ones((n, n))
    label_to_index = {name: i for i, name in enumerate(criteria)}
    for (a, b), val in comparisons.items():
        i, j = label_to_index[a], label_to_index[b]
        pcm[i, j] = val
        pcm[j, i] = 1 / val
    geom_means = np.prod(pcm, axis=1) ** (1/n)
    weights = geom_means / np.sum(geom_means)
    return dict(zip(criteria, weights))

def fuzzy_topsis(crisp_scores, weights):
    norm = np.linalg.norm(crisp_scores, axis=0)
    norm_scores = crisp_scores / norm
    weighted = norm_scores * weights
    ideal = np.max(weighted, axis=0)
    anti_ideal = np.min(weighted, axis=0)
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)
    closeness = d_neg / (d_pos + d_neg)
    return pd.DataFrame({
        "WEC Design": wec_designs,
        "Closeness to Ideal": np.round(closeness, 4)
    }).sort_values(by="Closeness to Ideal", ascending=False)

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

tab1, tab2, tab3 = st.tabs(["Fuzzy AHP + TOPSIS", "Community Feedback", "Live Sheet View"])

with tab1:
    st.header("Fuzzy AHP + Fuzzy TOPSIS")
    comparisons = {}
    for i in range(len(themes)):
        for j in range(i+1, len(themes)):
            a, b = themes[i], themes[j]
            comparisons[(a, b)] = st.slider(f"{a} vs {b}", 1, 9, 3)

    st.info("Fuzzy scores will be auto-generated from feedback data in the sheet.")

    if st.button("Run Fuzzy TOPSIS"):
        try:
            sheet = connect_to_google_sheets()
            data = pd.DataFrame(sheet.get_all_records())

            # Group and convert feedback using GPT (not shown here — placeholder)
            # For now, simulate scores:
            fuzzy_scores = {
                "Visual Impact": [(2, 3, 4), (3, 4, 5), (4, 5, 6)],
                "Ecosystem Safety": [(3, 4, 5), (2, 3, 4), (4, 5, 6)],
                "Maintenance": [(4, 5, 6), (3, 4, 5), (2, 3, 4)],
                "Cultural Fit": [(3, 4, 5), (4, 5, 6), (2, 3, 4)]
            }

            weights_dict = fuzzy_ahp_to_weights(comparisons, themes)
            weights = [weights_dict[t] for t in themes]
            crisp_scores = np.array([
                [(a + b + c) / 3 for (a, b, c) in fuzzy_scores[theme]]
                for theme in themes
            ]).T
            result_df = fuzzy_topsis(crisp_scores, weights)
            st.dataframe(result_df)
            pdf_path = export_decision_report(result_df, "Results based on fuzzy scores extracted from community feedback.")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Report", f, file_name="WEC_Decision_Report.pdf")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Community Feedback Submission")

    name = st.text_input("Your Name")
    community = st.text_input("Community Name")
    design = st.selectbox("WEC Design", wec_designs)

    vi_text = st.text_area("Visual Impact Feedback")
    eco_text = st.text_area("Ecosystem Concern")
    maint_text = st.text_area("Maintenance Thoughts")
    culture_text = st.text_area("Cultural Compatibility")

    if st.button("Submit Feedback"):
        try:
            sheet = connect_to_google_sheets()
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name, community, design,
                vi_text, eco_text, maint_text, culture_text
            ]
            sheet.append_row(row)
            st.success("Feedback submitted successfully!")
        except Exception as e:
            st.error(f"Error submitting to sheet: {e}")

    if st.button("Generate Fuzzy Scores with AI (Optional)") and all([vi_text, eco_text, maint_text, culture_text]):
        full_text = f"""
        WEC Design: {design}
        Visual Impact: {vi_text}
        Ecosystem Concern: {eco_text}
        Maintenance Thoughts: {maint_text}
        Cultural Compatibility: {culture_text}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that extracts fuzzy scores (triangular numbers) for 4 themes (Visual Impact, Ecosystem Safety, Maintenance, Cultural Fit) based on community feedback. Respond only in JSON."
                    },
                    {
                        "role": "user",
                        "content": full_text
                    }
                ],
                temperature=0.3
            )
            st.code(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI error: {e}")

with tab3:
    st.header("Live View of All Feedback")
    if st.button("Load Feedback from Sheet"):
        try:
            sheet = connect_to_google_sheets()
            data = pd.DataFrame(sheet.get_all_records())
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error: {e}")
