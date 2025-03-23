
import streamlit as st
import pandas as pd
import numpy as np
import gspread
from openai import OpenAI
from google.oauth2.service_account import Credentials
from fpdf import FPDF
import tempfile
import os

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

themes = ["Visual Impact", "Ecosystem Safety", "Maintenance", "Cultural Fit"]
wec_designs = ["Point Absorber", "OWC", "Overtopping"]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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

def fuzzy_topsis(fuzzy_scores, weights):
    scores = []
    for i in range(len(wec_designs)):
        v = []
        for theme in themes:
            v.append(fuzzy_scores[theme][i])
        scores.append(v)

    scores = np.array(scores)  # (n_alts, n_criteria, 3)

    norm = np.sqrt(np.sum(scores ** 2, axis=0))
    norm_scores = scores / norm

    weighted = np.array([w * norm_scores[:, i] for i, w in enumerate(weights)]).T

    ideal = np.max(weighted, axis=0)
    anti_ideal = np.min(weighted, axis=0)

    d_pos = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

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

tab1, tab2, tab3 = st.tabs(["Fuzzy AHP + TOPSIS", "AI Score Extraction", "Live Community Feedback"])

with tab1:
    st.header("Fuzzy AHP + Fuzzy TOPSIS")

    comparisons = {}
    for i in range(len(themes)):
        for j in range(i+1, len(themes)):
            a, b = themes[i], themes[j]
            comparisons[(a, b)] = st.slider(f"{a} vs {b}", 1, 9, 3)

    fuzzy_scores = {
        "Visual Impact": [(2, 3, 4), (3, 4, 5), (4, 5, 6)],
        "Ecosystem Safety": [(3, 4, 5), (2, 3, 4), (4, 5, 6)],
        "Maintenance": [(4, 5, 6), (3, 4, 5), (2, 3, 4)],
        "Cultural Fit": [(3, 4, 5), (4, 5, 6), (2, 3, 4)]
    }

    if st.button("Run Fuzzy TOPSIS"):
    # Validate fuzzy_scores completeness
    if not all(len(fuzzy_scores[theme]) == len(wec_designs) for theme in themes):
        st.error("Each theme must have fuzzy scores for all WEC designs.")
    else:
        weights_dict = fuzzy_ahp_to_weights(comparisons, themes)
        weights = [weights_dict[t] for t in themes]

        crisp_scores = np.array([
            [(a+b+c)/3 for (a, b, c) in fuzzy_scores[theme]]
            for theme in themes
        ]).T

                result_df = fuzzy_topsis(crisp_scores, weights)
        st.dataframe(result_df)

        pdf_path = export_decision_report(result_df, "Results based on real fuzzy AHP weights and fuzzy TOPSIS.")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF Report", f, file_name="WEC_Decision_Report.pdf")

with tab2:
    st.header("AI-Assisted Fuzzy Score Extraction")

    user_input = st.text_area("Paste community feedback text here:")
    if st.button("Generate Fuzzy Scores with AI") and user_input:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that extracts fuzzy scores (as triangular numbers) for WEC designs across the themes: Visual Impact, Ecosystem Safety, Maintenance, Cultural Fit."
                    },
                    {
                        "role": "user",
                        "content": f"""Text: {user_input}

Please provide fuzzy scores in this format:
{{'Visual Impact': {{'OWC': (1, 2, 3), ...}}, ...}}"""
                    }
                ],
                temperature=0.3
            )

            raw_output = response.choices[0].message.content
            st.markdown("### Extracted Fuzzy Scores")
            st.code(raw_output)

        except Exception as e:
            st.error(f"OpenAI API error: {e}")

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
