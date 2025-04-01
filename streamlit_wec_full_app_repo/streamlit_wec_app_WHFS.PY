import streamlit as st
import pandas as pd
import numpy as np
import gspread
import json
from datetime import datetime
from google.oauth2.service_account import Credentials
from openai import OpenAI
import matplotlib.pyplot as plt
from PIL import Image
import ast

# === Config ===
THEMES = ["Visual Impact", "Ecosystem Concern", "Maintenance Thoughts", "Cultural Fit"]
WEC_DESIGNS = ["Point Absorber", "OWC", "Oscillating Surge Flap"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit#gid=0"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === Google Sheets Connection ===
def get_google_creds():
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"].encode().decode("unicode_escape"))
    return Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def connect_to_google_sheets():
    client = gspread.authorize(get_google_creds())
    return client.open_by_url(SPREADSHEET_URL)

def create_sheet_if_missing(sheet, title):
    if title not in [ws.title for ws in sheet.worksheets()]:
        new_sheet = sheet.add_worksheet(title=title, rows="100", cols="20")

        # Define dynamic headers based on tab title
        if title == "Community Feedback Tab":
            headers = ["Timestamp", "Name", "Community"] + THEMES
        elif title == "Expert Tab":
            headers = ["Theme"] + WEC_DESIGNS
        elif title == "Final Rankings":
            headers = ["WEC Design", "Closeness to Ideal"]
        else:
            headers = []

        if headers:
            new_sheet.append_row(headers)

# === Consistency Check ===
def calculate_consistency_index(pcm):
    n = len(pcm)
    eigvals, _ = np.linalg.eig(pcm)
    max_eigval = np.max(np.real(eigvals))
    CI = (max_eigval - n) / (n - 1)
    RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12}
    RI = RI_dict.get(n, 1.12)
    CR = CI / RI if RI != 0 else 0
    return round(CR, 4), round(max_eigval, 3)

def ahp_weights(pcm):
    geom_mean = np.prod(pcm, axis=1) ** (1 / len(pcm))
    weights = geom_mean / np.sum(geom_mean)
    return weights

# === Defuzzification ===
def defuzzify(tfn): return sum(tfn)/3

def aggregate_whfs_scores(df):
    theme_scores = {theme: [] for theme in THEMES}
    for i, row in df.iterrows():
        for theme in THEMES:
            raw = row.get(theme, "")
            try:
                score_obj = ast.literal_eval(raw) if isinstance(raw, str) else raw
                if isinstance(score_obj, list):  # WHFS is a list of TFNs
                    for tfn in score_obj:
                        if isinstance(tfn, list) and len(tfn) == 3:
                            theme_scores[theme].append(defuzzify(tfn))
            except Exception as e:
                print(f"Error parsing fuzzy score for row {i}, theme {theme}: {e}")
                continue
    weights = {t: np.mean(vals) for t, vals in theme_scores.items() if vals}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else {}

def topsis(matrix, weights):
    norm = np.linalg.norm(matrix, axis=0)
    norm_matrix = matrix / norm
    weighted = norm_matrix * weights
    ideal = np.max(weighted, axis=0)
    anti = np.min(weighted, axis=0)
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        closeness = d_neg / (d_pos + d_neg)
        closeness = np.nan_to_num(closeness, nan=0.0, posinf=0.0, neginf=0.0)


    # ✅ Fix NaN or infinite values
    closeness = np.nan_to_num(closeness, nan=0.0, posinf=0.0, neginf=0.0)
    return closeness.round(4)

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("🌊 WEC Decision Support Tool")
tabs = st.tabs(["1️⃣ Expert Input", "2️⃣ Community Feedback", "3️⃣ Final Ranking"])

# === TAB 1: EXPERT INPUT ===
with tabs[0]:
    st.subheader("Expert Scoring: Pairwise Comparisons of WEC Designs")
    sheet = connect_to_google_sheets()
    create_sheet_if_missing(sheet, "Expert Tab")

    expert_scores = {}
    show_warning = False
    inconsistent_themes = []

    for theme in THEMES:
        st.markdown(f"**{theme}**")
        pcm = np.ones((len(WEC_DESIGNS), len(WEC_DESIGNS)))
        for i in range(len(WEC_DESIGNS)):
            for j in range(i + 1, len(WEC_DESIGNS)):
                val = st.slider(f"{WEC_DESIGNS[i]} vs {WEC_DESIGNS[j]} ({theme})", 1, 9, 3)
                pcm[i, j] = val
                pcm[j, i] = 1 / val
        cr, lam_max = calculate_consistency_index(pcm)
        st.markdown(f"λₘₐₓ = {lam_max:.3f}, CI = ..., **CR = {cr:.3f}**")
        
        if cr > 0.1:
            st.warning(f"⚠️ Theme '{theme}' has inconsistent comparisons (CR > 0.1). Please revise.")
            inconsistent_themes.append(theme)
        
        expert_scores[theme] = ahp_weights(pcm)

    # ✅ Only show button if all themes are consistent
    if not inconsistent_themes:
        if st.button("✅ Save Expert Scores"):
            expert_df = pd.DataFrame(expert_scores, index=WEC_DESIGNS)
            ws = sheet.worksheet("Expert Tab")
            ws.clear()
            ws.update([["WEC"] + list(expert_df.columns)] + expert_df.reset_index().values.tolist())
            st.success("Expert scores saved successfully!")
    else:
        st.error("❌ One or more themes have inconsistent AHP matrices. Fix them before saving.")


# === TAB 2: COMMUNITY INPUT ===
with tabs[1]:
    st.subheader("Community Input: General Feedback")
    col1, col2 = st.columns([2, 1])  # Wider left column for form, right for image
    create_sheet_if_missing(sheet, "Community Feedback Tab")
    with col1:
        name = st.text_input("Name")
        community = st.text_input("Community")
        feedback = st.text_area("What are your thoughts about WEC in your community?")
        #wec_type = st.selectbox("Select a WEC Design", WEC_DESIGNS)

        if st.button("✉️ Submit Feedback"):
            try:
                prompt = f"""
                You are an expert assistant in converting qualitative community feedback into Weighted Hesitant Fuzzy Scores (WHFS) for 4 themes:

                {THEMES}

                For each theme, return:
                1. A short explanation (1-2 sentences)
                2. A list of 2 to 3 triangular fuzzy numbers (TFNs), each in the format [a, b, c], where 0 ≤ a ≤ b ≤ c ≤ 9

                Respond in this JSON format:
                {{
                "Visual Impact": {{
                    "explanation": "May obstruct scenic views during sunrise.",
                    "scores": [[2, 3, 4], [3, 4, 5]]
                }},
                "Ecosystem Concern": {{
                    "explanation": "Might affect fish breeding areas.",
                    "scores": [[1, 2, 3]]
                }},
                ...
                }}

                Community Feedback:
                \"\"\"{feedback}\"\"\"
                """

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                result = json.loads(response.choices[0].message.content)

                def fuzzy_list_to_str(x):
                    return json.dumps(x['scores']) if isinstance(x['scores'], list) else ""

                row = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name,
                    community
                ] + [fuzzy_list_to_str(result.get(theme)) for theme in THEMES]

                
                ws = sheet.worksheet("Community Feedback Tab")
                ws.append_row(row)
                st.success("Feedback saved successfully!")
                # Show extracted themes with markdown formatting
                st.markdown("### AI-Assited Feedback Breakdown by Theme")

                for theme in THEMES:
                    explanation = result[theme]["explanation"]
                    scores = result[theme]["scores"]

                    st.markdown(f"**{theme}**")
                    st.markdown(f"- 📝 *Explanation:* {explanation}")
                    st.markdown(f"- 🔢 *Fuzzy Score:* `{scores}`")

            except Exception as e:
                st.error(f"GPT failed: {e}")

    with col2:
        spacer, image_col = st.columns([1, 4])  # Add left-padding using spacer column
        with image_col:
            try:
                image = Image.open("WEC Classification.png")
                st.image(image, caption="RM3, RM5, RM6 - WEC Types", width=500)
            except:
                st.warning("WEC Classification image not found.")

# === TAB 3: FINAL RANKING ===
with tabs[2]:
    st.subheader("Final WEC Ranking using WHFS-TOPSIS")
    create_sheet_if_missing(sheet, "Final Rankings")
    if st.button("🏁 Run Ranking"):
        try:
            exp_df = pd.DataFrame(sheet.worksheet("Expert Tab").get_all_records())
            comm_df = pd.DataFrame(sheet.worksheet("Community Feedback Tab").get_all_records())

            exp_matrix = exp_df.set_index("WEC").loc[WEC_DESIGNS][THEMES].astype(float).values
            theme_weights = aggregate_whfs_scores(comm_df)
            weights = np.array([theme_weights[t] for t in THEMES])

            closeness = topsis(exp_matrix, weights)
            result_df = pd.DataFrame({
                "WEC Design": WEC_DESIGNS,
                "Closeness to Ideal": closeness
            }).sort_values(by="Closeness to Ideal", ascending=False)

            st.markdown("### 📊 Theme Weights")
            st.dataframe(pd.DataFrame.from_dict(theme_weights, orient='index', columns=["Weight"]))

            st.markdown("### 🔍 Expert Score Matrix")
            st.dataframe(exp_df.set_index("WEC"))

            st.markdown("### 🏆 TOPSIS Results")
            st.dataframe(result_df)

            # Save final ranking
            sheet.worksheet("Final Rankings").clear()
            sheet.worksheet("Final Rankings").update(
                [["WEC Design", "Closeness to Ideal"]] + result_df.values.tolist()
            )
        except Exception as e:
            st.error(f"Error during ranking: {e}")
