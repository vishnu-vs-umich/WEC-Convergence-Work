
import streamlit as st
# © 2025 Vishnu Vijayasankar. All rights reserved.

import pandas as pd
import numpy as np
import gspread
from openai import OpenAI

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
import json
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(layout="wide")
st.title("Wave Energy Converter Decision Support Tool")

themes = ["Visual Impact", "Ecosystem Concern", "Maintenance Thoughts", "Cultural Compatibility"]
wec_designs = ["Point Absorber", "OWC", "Oscillating Surge Flap"]

# Google & OpenAI setup
OpenAI.api_key = st.secrets["OPENAI_API_KEY"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit#gid=0"

import json
from PIL import Image

def get_google_creds():
    creds_dict = json.loads(
        st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"].encode().decode('unicode_escape')
    )
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
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're an expert in qualitative-to-quantitative transformation using fuzzy logic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        score = eval(content)
        if isinstance(score, tuple) and len(score) == 3:
            return score
    except:
        pass
    st.warning("⚠️ No valid fuzzy score returned for this feedback. Skipping score calculation.")
    return None


def map_to_fuzzy_scale(value):
    scale = {
        1: (1, 1, 1),
        2: (1, 2, 3),
        3: (2, 3, 4),
        4: (3, 4, 5),
        5: (4, 5, 6),
        6: (5, 6, 7),
        7: (6, 7, 8),
        8: (7, 8, 9),
        9: (8, 9, 9)
    }
    return scale.get(value, (1, 1, 1))

def fuzzy_ahp_to_weights(comparisons, criteria):
    n = len(criteria)
    L = np.ones((n, n))
    M = np.ones((n, n))
    U = np.ones((n, n))
    index = {c: i for i, c in enumerate(criteria)}

    for (a, b), v in comparisons.items():
        i, j = index[a], index[b]
        l, m, u = map_to_fuzzy_scale(v)
        L[i, j], M[i, j], U[i, j] = l, m, u
        L[j, i], M[j, i], U[j, i] = 1/u, 1/m, 1/l

    G_L, G_M, G_U = [], [], []
    for i in range(n):
        gm_L = np.prod(L[i]) ** (1/n)
        gm_M = np.prod(M[i]) ** (1/n)
        gm_U = np.prod(U[i]) ** (1/n)
        G_L.append(gm_L)
        G_M.append(gm_M)
        G_U.append(gm_U)

    sum_L, sum_M, sum_U = sum(G_L), sum(G_M), sum(G_U)
    weights = []
    for i in range(n):
        wl = G_L[i]/sum_U
        wm = G_M[i]/sum_M
        wu = G_U[i]/sum_L
        weights.append(((wl + wm + wu)/3))

    total = sum(weights)
    return dict(zip(criteria, [w/total for w in weights]))


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

    
    # Calculate crisp pairwise comparison matrix and CR
    def calculate_crisp_pcm(comparisons, criteria):
        n = len(criteria)
        pcm = np.ones((n, n))
        idx = {name: i for i, name in enumerate(criteria)}
        for (a, b), val in comparisons.items():
            i, j = idx[a], idx[b]
            pcm[i, j] = val
            pcm[j, i] = 1 / val
        return pcm

    def check_consistency(pcm):
        n = pcm.shape[0]
        eigvals, _ = np.linalg.eig(pcm)
        lambda_max = np.max(np.real(eigvals))
        ci = (lambda_max - n) / (n - 1)
        RI_dict = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                   7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        RI = RI_dict.get(n, 1.49)
        cr = ci / RI if RI != 0 else 0
        return cr, ci, lambda_max

    pcm = calculate_crisp_pcm(comparisons, themes)
    cr, ci, lambda_max = check_consistency(pcm)

    st.subheader("📏 AHP Consistency Check")
    st.markdown(f"**Consistency Ratio (CR):** `{cr:.3f}`  <br>**Lambda Max (λₘₐₓ):** `{lambda_max:.3f}`", unsafe_allow_html=True)

    if cr > 0.1:
        st.error("⚠️ Your pairwise comparisons are inconsistent (CR > 0.10).")
        st.markdown("""
        **Suggestions to reduce inconsistency:**
        - Recheck your sliders — do they contradict each other?
        - Ensure logical transitivity: If A > B and B > C, then A > C
        - Reduce extreme ratings unless you're confident
        """)
        can_run_topsis = False
    else:
        st.success("✅ Pairwise matrix is consistent (CR ≤ 0.10). You may run Fuzzy AHP + TOPSIS.")
        can_run_topsis = True

    run_button = st.button("Run Fuzzy TOPSIS", disabled=not can_run_topsis)
    if run_button and can_run_topsis:

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

        
        
        
        
        
        # Live theme weight visualization (minimal + visually balanced)
        st.subheader("🔍 Theme Priority Weights (Fuzzy AHP)")
        import matplotlib.pyplot as plt
        from textwrap import fill

        labels = list(weights_dict.keys())
        values = list(weights_dict.values())
        wrapped_labels = [fill(label, width=12) for label in labels]

        fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Control plot size visually
        bars = ax.bar(wrapped_labels, values, color='skyblue')
        ax.set_title("Theme Weights", fontsize=8)
        ax.set_ylabel("Weight", fontsize=6)
        ax.set_ylim(0, 1)

        plt.xticks(rotation=45, ha='right', fontsize=6)
        ax.tick_params(axis='y', labelsize=6)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{val:.2f}", ha='center', fontsize=5)

        plt.tight_layout()

        # Use Streamlit layout (column) to center it
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig)


        weights = [weights_dict[t] for t in themes]
        crisp_scores = np.array([
            [(a + b + c) / 3 for (a, b, c) in fuzzy_scores[theme]]
            for theme in themes
        ]).T

        result_df = fuzzy_topsis(crisp_scores, weights)
        st.dataframe(result_df)

        # SAVE TO GOOGLE SHEET - NEW TAB WITH FORMATTING
        spreadsheet = gspread.authorize(get_google_creds()).open_by_url(SPREADSHEET_URL)
        try:
            sheet_results = spreadsheet.worksheet("Fuzzy Results")
            spreadsheet.del_worksheet(sheet_results)
        except:
            pass
        sheet_results = spreadsheet.add_worksheet(title="Fuzzy Results", rows="100", cols="20")

        # Title: Fuzzy Scores
        sheet_results.update("A1", [["Fuzzy Scores (Theme × WEC)"]])
        sheet_results.format("A1", {"textFormat": {"bold": True, "fontSize": 12}})
        sheet_results.update("A2", [["Theme", "WEC Design", "L", "M", "U"]])
        sheet_results.format("A2:E2", {"textFormat": {"bold": True}})

        fuzzy_rows = []
        for theme in themes:
            for i, design in enumerate(wec_designs):
                L, M, U = fuzzy_scores[theme][i]
                fuzzy_rows.append([theme, design, L, M, U])
        sheet_results.append_rows(fuzzy_rows, value_input_option="USER_ENTERED")

        # Title: Crisp Scores
        start_row = len(fuzzy_rows) + 4
        sheet_results.update(f"A{start_row}", [["Crisp Score Matrix"]])
        sheet_results.format(f"A{start_row}", {"textFormat": {"bold": True, "fontSize": 12}})
        sheet_results.update(f"A{start_row+1}", [["Theme"] + wec_designs])
        sheet_results.format(f"A{start_row+1}:{chr(65+len(wec_designs))}{start_row+1}", {"textFormat": {"bold": True}})

        for i, theme in enumerate(themes):
            row = [theme] + list(np.round(crisp_scores[:, i], 3))
            sheet_results.append_row(row, value_input_option="USER_ENTERED")

        # Title: Closeness to Ideal
        start_row = start_row + len(themes) + 4
        sheet_results.update(f"A{start_row}", [["Fuzzy TOPSIS Closeness to Ideal Ranking"]])
        sheet_results.format(f"A{start_row}", {"textFormat": {"bold": True, "fontSize": 12}})
        sheet_results.update(f"A{start_row+1}", [["WEC Design", "Closeness to Ideal"]])
        sheet_results.format(f"A{start_row+1}:B{start_row+1}", {"textFormat": {"bold": True}})

        for _, row in result_df.iterrows():
            sheet_results.append_row([row["WEC Design"], row["Closeness to Ideal"]], value_input_option="USER_ENTERED")


with tab2:
    st.header("Submit Community Feedback")

    col1, col2 = st.columns([2, 1])  # Wider left column for form, right for image

    with col1:
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
                        response = client.chat.completions.create(
                            model="gpt-4",
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

    with col2:
        spacer, image_col = st.columns([1, 4])  # Add left-padding using spacer column
        with image_col:
            try:
                image = Image.open("WEC Classification.png")
                st.image(image, caption="RM3, RM5, RM6 - WEC Types", width=500)
            except:
                st.warning("WEC Classification image not found.")


with tab3:
    st.header("Live Feedback Sheet")
    if st.button("Refresh Sheet"):
        sheet = connect_to_google_sheets()
        data = pd.DataFrame(sheet.get_all_records())
        st.dataframe(data)



# Inject footer at bottom of app
st.markdown(
    "<hr style='margin-top: 50px;'><div style='text-align: center; font-size: 18px; color: gray;'>© 2025 Vishnu Vijayasankar. All rights reserved.</div>",
    unsafe_allow_html=True
)

#streamlit run streamlit_wec_app_fuzzy.py --server.runOnSave true
