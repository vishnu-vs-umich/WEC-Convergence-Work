
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import openai
import json
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

themes = ["Visual Impact", "Ecosystem Concern", "Maintenance Thoughts", "Cultural Compatibility"]
wec_designs = ["Point Absorber", "OWC", "Overtopping"]

# Google & OpenAI setup
openai.api_key = st.secrets["OPENAI_API_KEY"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit#gid=0"

import json

def get_google_creds():
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )

def connect_to_google_sheets():
    client = gspread.authorize(get_google_creds())
    return client.open_by_url(SPREADSHEET_URL).sheet1

def extract_fuzzy_score_from_text(text, theme):
    prompt = f"""
Convert the following community feedback into a fuzzy score (triangular number) for the theme: {theme}.

Respond only with a Python tuple like: (2, 3, 4)

Feedback: "{text}"
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're an expert in qualitative-to-quantitative transformation using fuzzy logic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        score = eval(content)
        if isinstance(score, tuple) and len(score) == 3:
            return score
    except:
        pass
    return (2, 3, 4)  # fallback default

def fuzzy_ahp_to_weights(comparisons, criteria):
    n = len(criteria)
    pcm = np.ones((n, n))
    idx = {name: i for i, name in enumerate(criteria)}
    for (a, b), val in comparisons.items():
        i, j = idx[a], idx[b]
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

tab1, tab2, tab3 = st.tabs(["Fuzzy AHP + TOPSIS", "Community Feedback", "Live Sheet View"])

with tab1:
    st.header("Fuzzy AHP + Fuzzy TOPSIS")
    comparisons = {}
    for i in range(len(themes)):
        for j in range(i + 1, len(themes)):
            a, b = themes[i], themes[j]
            comparisons[(a, b)] = st.slider(f"{a} vs {b}", 1, 9, 3)

    if st.button("Run Fuzzy TOPSIS"):
        sheet = connect_to_google_sheets()
        data = pd.DataFrame(sheet.get_all_records())

        # Create structure: {theme: [score_per_design]}
        fuzzy_scores = {theme: [(0, 0, 0)] * len(wec_designs) for theme in themes}
        count_matrix = {theme: [0] * len(wec_designs) for theme in themes}

        for _, row in data.iterrows():
            design = row["WEC Design"]
            if design not in wec_designs:
                continue
            idx = wec_designs.index(design)
            for theme in themes:
                text = row.get(theme, "")
                if text.strip():
                    score = extract_fuzzy_score_from_text(text, theme)
                    prev = fuzzy_scores[theme][idx]
                    new_score = tuple(p + s for p, s in zip(prev, score))
                    fuzzy_scores[theme][idx] = new_score
                    count_matrix[theme][idx] += 1

        for theme in themes:
            for i in range(len(wec_designs)):
                count = count_matrix[theme][i]
                if count > 0:
                    fuzzy_scores[theme][i] = tuple(x / count for x in fuzzy_scores[theme][i])
                else:
                    fuzzy_scores[theme][i] = (2, 3, 4)  # default

        weights_dict = fuzzy_ahp_to_weights(comparisons, themes)
        weights = [weights_dict[t] for t in themes]
        crisp_scores = np.array([
            [(a + b + c) / 3 for (a, b, c) in fuzzy_scores[theme]]
            for theme in themes
        ]).T

        result_df = fuzzy_topsis(crisp_scores, weights)
        st.dataframe(result_df)

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
You are an assistant that reads community feedback about a wave energy converter (WEC) and fills out these 4 themes:
- Visual Impact Feedback
- Ecosystem Concern
- Maintenance Thoughts
- Cultural Compatibility

Return a JSON like:
{{"Visual Impact Feedback": "...", "Ecosystem Concern": "...", "Maintenance Thoughts": "...", "Cultural Compatibility": "..."}}

Feedback:
"{general_feedback}"
"""
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You extract structured values from raw feedback."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    structured = response.choices[0].message.content.strip()
                    result = json.loads(structured)

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
                    st.success("Feedback saved successfully!")
                    st.markdown("### Interpreted Themes")
                    st.json(result)
                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.header("Live Feedback Sheet")
    if st.button("Refresh Sheet"):
        sheet = connect_to_google_sheets()
        data = pd.DataFrame(sheet.get_all_records())
        st.dataframe(data)
