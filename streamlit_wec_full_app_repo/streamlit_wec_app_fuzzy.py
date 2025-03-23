
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import openai
from google.oauth2.service_account import Credentials
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

themes = ["Visual Impact", "Ecosystem Safety", "Maintenance", "Cultural Fit"]
wec_designs = ["Point Absorber", "OWC", "Overtopping"]

openai.api_key = st.secrets["OPENAI_API_KEY"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit?gid=0"

def get_google_creds():
    return Credentials.from_service_account_info(
        st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

def connect_to_google_sheets():
    creds = get_google_creds()
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SPREADSHEET_URL).sheet1
    return sheet

tab1, tab2, tab3 = st.tabs(["Fuzzy AHP + TOPSIS", "Community Feedback", "Live Sheet View"])

with tab2:
    st.header("Submit Community Feedback")

    name = st.text_input("Your Name")
    community = st.text_input("Community Name")
    design = st.selectbox("WEC Design", wec_designs)
    general_feedback = st.text_area("What are your thoughts about this WEC design for your community?")

    if st.button("Submit and Analyze"):
        if not all([name, community, design, general_feedback]):
            st.warning("Please fill out all fields before submitting.")
        else:
            with st.spinner("Interpreting feedback with AI..."):
                try:
                    prompt = f"""
You are an assistant that reads community feedback about a wave energy converter (WEC) design and fills out structured insights for four categories:
- Visual Impact Feedback
- Ecosystem Concern
- Maintenance Thoughts
- Cultural Compatibility

Given this raw feedback, extract the relevant points for each theme and return a JSON with the following structure:
{{
  "Visual Impact Feedback": "...",
  "Ecosystem Concern": "...",
  "Maintenance Thoughts": "...",
  "Cultural Compatibility": "..."
}}

Raw Feedback:
"{general_feedback}"
"""

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert assistant that extracts structured information from community feedback."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )

                    structured = response.choices[0].message.content.strip()
                    st.markdown("### AI-Extracted Insights")
                    st.code(structured, language="json")

                    # Attempt to convert AI output to dictionary
                    try:
                        import json
                        result = json.loads(structured)

                        # Prepare row for Google Sheets
                        row = [
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            name,
                            community,
                            design,
                            result.get("Visual Impact Feedback", ""),
                            result.get("Ecosystem Concern", ""),
                            result.get("Maintenance Thoughts", ""),
                            result.get("Cultural Compatibility", "")
                        ]

                        sheet = connect_to_google_sheets()
                        sheet.append_row(row)
                        st.success("Feedback submitted and saved successfully!")

                    except Exception as parse_err:
                        st.error(f"Failed to parse AI output: {parse_err}")

                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

with tab1:
    st.header("Fuzzy AHP + Fuzzy TOPSIS")
    st.info("This will soon use the structured feedback stored in the sheet for scoring and decision-making.")

with tab3:
    st.header("Live View of All Feedback")
    if st.button("Load Feedback from Sheet"):
        try:
            sheet = connect_to_google_sheets()
            data = pd.DataFrame(sheet.get_all_records())
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error: {e}")
