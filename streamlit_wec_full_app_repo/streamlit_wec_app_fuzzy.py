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
import matplotlib.pyplot as plt
import seaborn as sns


# ✅ FIRST Streamlit command
st.set_page_config(layout="wide")

# === Config and setup continues below ===
WEC_DESIGNS = ["Point Absorber", "OWC", "Oscillating Surge Flap"]
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1mVOU66Ab-AlZaddRzm-6rWar3J_Nmpu69Iw_L4GTXq0/edit#gid=0"

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data(ttl=600)
def load_themes_and_subcriteria(_sheet):
    """
    Loads theme-subcriterion pairs from the Themes Tab.
    Returns:
        - themes: list of unique themes
        - mapping: dict {theme: [subcriterion1, subcriterion2, ...]}
    """
    ws = _sheet.worksheet("Themes Tab")
    rows = ws.get_all_records()
    theme_map = {}
    for row in rows:
        theme = row.get("Theme", "").strip()
        sub = row.get("Subcriterion", "").strip()
        if theme and sub:
            theme_map.setdefault(theme, []).append(sub)
    themes = list(theme_map.keys())
    return themes, theme_map

# === Google Sheets Connection ===
def get_google_creds():
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"].encode().decode("unicode_escape"))
    return Credentials.from_service_account_info(creds_dict, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def connect_to_google_sheets():
    client = gspread.authorize(get_google_creds())
    return client.open_by_url(SPREADSHEET_URL)

def create_themes_tab_with_subcriteria(sheet):
    """
    Creates the 'Themes Tab' with the final themes and subcriteria (one row per pair).
    """
    if "Themes Tab" not in [ws.title for ws in sheet.worksheets()]:
        ws = sheet.add_worksheet(title="Themes Tab", rows="100", cols="2")
        ws.append_row(["Theme", "Subcriterion"])  # Header row

        theme_data = [
            # Functional Efficiency
            ("Functional Efficiency", "Electricity reliability and security"),
            ("Functional Efficiency", "Electricity affordability"),

            # Environmental Sustainability
            ("Environmental Sustainability", "Habitat of marine animals"),
            ("Environmental Sustainability", "Birds"),
            ("Environmental Sustainability", "Health"),
            ("Environmental Sustainability", "Noise"),

            # Sense of Place
            ("Sense of Place", "Personal identity / connection to place"),
            ("Sense of Place", "Ocean / landscape view"),

            # Community Prosperity
            ("Community Prosperity", "Community benefits"),
            ("Community Prosperity", "Job opportunities"),
            ("Community Prosperity", "Tax revenues"),
            ("Community Prosperity", "Indirect economic effects"),
            ("Community Prosperity", "Tourism"),

            # Marine Space Utilization
            ("Marine Space Utilization", "Recreational fishing"),
            ("Marine Space Utilization", "Recreational boating")
        ]

        for theme, sub in theme_data:
            ws.append_row([theme, sub])

sheet = connect_to_google_sheets()
create_themes_tab_with_subcriteria(sheet)  # Make sure the sheet exists
THEMES, THEME_SUBCRITERIA = load_themes_and_subcriteria(sheet)

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

def ensure_headers_if_missing(sheet, title):
    ws = sheet.worksheet(title)
    existing_rows = ws.get_all_values()

    if not existing_rows or all([all(cell == "" for cell in row) for row in existing_rows]):
        # Clear and rewrite appropriate headers
        ws.clear()

        if title == "Community Feedback Tab":
            headers = ["Timestamp", "Name", "Community"] + THEMES
        elif title == "Expert Tab":
            headers = ["Theme"] + WEC_DESIGNS
        elif title == "Final Rankings":
            headers = ["WEC Design", "Closeness to Ideal"]
        elif title == "Expert Contributions Tab":
            headers = ["Timestamp", "Name", "Theme", "Expertise Level"]
            for i in range(len(WEC_DESIGNS)):
                for j in range(i + 1, len(WEC_DESIGNS)):
                    headers.append(f"PCM_{WEC_DESIGNS[i]}_vs_{WEC_DESIGNS[j]}")
        else:
            headers = []

        if headers:
            ws.append_row(headers)

def clear_all_google_sheet_tabs():
    """
    Connects to the Google Sheet and clears all worksheets (tabs) without deleting them.
    """
    try:
        sheet = connect_to_google_sheets()  # Uses your existing connection function
        for ws in sheet.worksheets():
            ws.clear()
        st.success("✅ All sheets have been cleared successfully.")
    except Exception as e:
        st.error(f"❌ Failed to clear sheets: {e}")

# === Consistency Check ===
def calculate_consistency_index(pcm):
    n = len(pcm)
    eigvals, _ = np.linalg.eig(pcm)
    lam_max = np.max(np.real(eigvals))
    CI = (lam_max - n) / (n - 1) if n > 1 else 0

    # Saaty's RI values (Robles-Algarin 2017)
    RI_lookup = {
        1: 0.0,
        2: 0.0,
        3: 0.52,
        4: 0.89,
        5: 1.11,
        6: 1.25,
        7: 1.35,
        8: 1.40,
        9: 1.45,
        10: 1.49
    }

    RI = RI_lookup.get(n, 1.49)  # Default to 1.49 if n > 10
    CR = CI / RI if RI != 0 else 0
    return round(CR, 4), round(lam_max, 3), round(CI, 4)


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
st.markdown("""
<h1 style='font-size: 40px;'>
    <span style='color:#00274C; font-weight: bold; font-family: cursive;'>Blue</span>
    <span style='color:#FFCB05; font-weight: bold; font-family: cursive;'>Choice </span>
    <span style='font-weight: normal;'>: A Community Driven WEC Decision Support Tool</span>
</h1>
""", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
tabs = st.tabs(["1️⃣ Expert Input", "2️⃣ Community Feedback", "3️⃣ Final Ranking"])

# === TAB 1: EXPERT INPUT ===
with tabs[0]:
    st.warning("*All fields are mandatory")
    sheet = connect_to_google_sheets()
    create_themes_tab_with_subcriteria(sheet)
    create_sheet_if_missing(sheet, "Expert Tab")
    ensure_headers_if_missing(sheet, "Expert Tab")
    create_sheet_if_missing(sheet, "Expert Contributions Tab")
    ensure_headers_if_missing(sheet, "Expert Contributions Tab")

    expert_scores = {}
    expert_pcms = {}
    inconsistent_themes = []
    expertise_levels = {}

    st.subheader("Expert Scoring: Expertise Level")
    col1, spacer, col2 = st.columns([3, 0.3, 2])

    with col1:
        expert_name = st.text_input("👤 Your Name")
        st.markdown("### 📊 Self-Rated Expertise")
        for theme in THEMES:
            expertise_levels[theme] = st.slider(
                f"Rate your expertise in scoring **{theme}** (1 = low, 5 = high)", 1, 5, 3
            )

    with col2:
        st.markdown("### 📘 Theme Descriptions")
        theme_descriptions = {
            "Functional Efficiency": "Ability of the technology to provide reliable and affordable electricity to the community.",
            "Environmental Sustainability": "Minimization of negative impacts on marine ecosystems, wildlife, and human health.",
            "Sense of Place": "Respect for the cultural, visual, and emotional identity of the community and natural surroundings.",
            "Community Prosperity": "Opportunities for local economic development, job creation, tourism, and indirect financial benefits.",
            "Marine Space Utilization": "Efficient and harmonious coexistence with existing maritime activities such as fishing and recreation."
        }

        table_html = """
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }
            th, td {
                border: 1px solid #DDD;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #00274C;
                color: #FFCB05;
            }
        </style>
        <table>
            <thead>
                <tr><th>Theme</th><th>Description</th></tr>
            </thead>
            <tbody>
        """

        for theme, desc in theme_descriptions.items():
            table_html += f"<tr><td>{theme}</td><td>{desc}</td></tr>"

        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

    st.subheader("Expert Scoring: Pairwise Comparisons of WEC Designs")
    st.info("Scroll to the bottom to see the Pairwise Comparison Slider Guide")

    st.markdown("""
    <div style="padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; font-size: 15px">
        All criteria and subcriteria are treated as <strong>benefit criteria</strong>. 
        <br>That means <span style="color: green;"><strong>higher values are always better</strong></span>.
        No inversions or cost adjustments are applied.
    </div>
    """, unsafe_allow_html=True)

    for theme in THEMES:
        st.markdown(
            f"<h4 style='color:#00274C; background-color:#FFCB05; padding:6px'>{theme}</h4>",
            unsafe_allow_html=True,
        )
        for subcriterion in THEME_SUBCRITERIA[theme]:
            st.markdown(
                f"<div style='font-weight:600; margin-left:10px;'>🔹 {subcriterion}</div>",
                unsafe_allow_html=True,
            )
            pcm = np.ones((len(WEC_DESIGNS), len(WEC_DESIGNS)))
            for i in range(len(WEC_DESIGNS)):
                for j in range(i + 1, len(WEC_DESIGNS)):
                    st.caption(f"⬅️ Slide left if **{WEC_DESIGNS[j]}** is better, ➡️ slide right if **{WEC_DESIGNS[i]}** is better.")
                    label = f"In terms of **{subcriterion}**, how much better is **{WEC_DESIGNS[i]}** than **{WEC_DESIGNS[j]}**?"
                    
                    scale_options = [-9, -7, -5, -3, 1, 3, 5, 7, 9]

                    descriptors = {
                        -9: f"{WEC_DESIGNS[j]} ≫ {WEC_DESIGNS[i]}",
                        -7: f"{WEC_DESIGNS[j]} > {WEC_DESIGNS[i]}",
                        -5: f"{WEC_DESIGNS[j]} > {WEC_DESIGNS[i]}",
                        -3: f"{WEC_DESIGNS[j]} > {WEC_DESIGNS[i]}",
                        1: f"{WEC_DESIGNS[i]} = {WEC_DESIGNS[j]}",
                        3: f"{WEC_DESIGNS[i]} > {WEC_DESIGNS[j]}",
                        5: f"{WEC_DESIGNS[i]} > {WEC_DESIGNS[j]}",
                        7: f"{WEC_DESIGNS[i]} > {WEC_DESIGNS[j]}",
                        9: f"{WEC_DESIGNS[i]} ≫ {WEC_DESIGNS[j]}"
                    }

                    val = st.select_slider(
                        label,
                        options=scale_options,
                        value=1,
                        format_func=lambda x: f"{abs(x)} ({descriptors[x]})",
                        key=f"{theme}_{subcriterion}_{i}_{j}"
                    )

                    if val == 1:
                        pcm[i, j] = 1
                        pcm[j, i] = 1
                    elif val > 1:
                        pcm[i, j] = val
                        pcm[j, i] = 1 / val
                    elif val < 1:
                        pcm[i, j] = 1 / abs(val)
                        pcm[j, i] = abs(val)

            cr, lam_max, ci = calculate_consistency_index(pcm)
            st.markdown(
                f"<div style='margin-left:10px;'>λₘₐₓ = {lam_max:.3f}, CI = {ci:.3f}, <strong>CR = {cr:.3f}</strong></div>",
                unsafe_allow_html=True,
            )

            if cr > 0.1:
                st.warning(
                    f"⚠️ Subcriterion '{subcriterion}' under theme '{theme}' is inconsistent (CR > 0.1). Please revise."
                )
                inconsistent_themes.append(f"{theme} - {subcriterion}")
            else:
                # ✅ Save weights
                expert_scores[(theme, subcriterion)] = ahp_weights(pcm)
                expert_pcms[(theme, subcriterion)] = pcm

                # ✅ Show PCM as dataframe
                df_pcm = pd.DataFrame(pcm, index=WEC_DESIGNS, columns=WEC_DESIGNS)
                st.markdown(f"**Pairwise Comparison Matrix for:** _{subcriterion}_")
                st.dataframe(df_pcm.style.format("{:.3f}"))

                # ✅ Show Heatmap
                # fig, ax = plt.subplots(figsize=(4.5, 3.5))
                # sns.heatmap(df_pcm, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax)
                # ax.set_title(f"Heatmap: {subcriterion}", fontsize=10)
                # ax.tick_params(axis='x', labelrotation=45)
                # st.pyplot(fig)


            expert_scores[(theme, subcriterion)] = ahp_weights(pcm)
            expert_pcms[(theme, subcriterion)] = pcm

    if not inconsistent_themes:
        if st.button("✅ Save Expert Scores"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet_ws = sheet.worksheet("Expert Contributions Tab")
            header = ["Timestamp", "Name", "Theme", "Subcriterion", "Expertise Level"]
            for i in range(len(WEC_DESIGNS)):
                for j in range(i + 1, len(WEC_DESIGNS)):
                    header.append(f"PCM_{WEC_DESIGNS[i]}_vs_{WEC_DESIGNS[j]}")
            existing_rows = sheet_ws.get_all_values()
            if len(existing_rows) == 0 or existing_rows == [[]]:
                sheet_ws.append_row(header)

            for (theme, subcriterion), pcm in expert_pcms.items():
                flat_pcm = []
                for i in range(len(WEC_DESIGNS)):
                    for j in range(i + 1, len(WEC_DESIGNS)):
                        flat_pcm.append(pcm[i, j])
                level = expertise_levels[theme]
                row = [timestamp, expert_name, theme, subcriterion, level] + flat_pcm
                sheet_ws.append_row(row)

            st.success("✅ Expert input saved successfully!")
    else:
        st.error("❌ One or more subcriteria are inconsistent. Fix them before saving.")

    st.markdown("""
    <style>
        .comparison-guide {
            font-size: 14px;
            margin-top: 30px;
        }
        .comparison-guide table {
            border-collapse: collapse;
            width: 100%;
        }
        .comparison-guide th, .comparison-guide td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .comparison-guide th {
            background-color: #003366;
            color: #FFCB05;
        }
        .comparison-guide caption {
            caption-side: top;
            text-align: left;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }
        .example {
            background-color: #f1f8ff;
            border-left: 4px solid #1f77b4;
            padding: 10px;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>

    <div class="comparison-guide">
    <h4>Pairwise Comparison Slider Guide</h4>
    <table>
        <tr>
            <th>Slider Value</th>
            <th>Meaning</th>
        </tr>
        <tr><td><b>1</b></td><td>Both WEC designs perform equally under this subcriterion</td></tr>
        <tr><td><b>3</b></td><td>First design performs slightly better than the second</td></tr>
        <tr><td><b>5</b></td><td>First design clearly performs better than the second</td></tr>
        <tr><td><b>7</b></td><td>First design strongly outperforms the second</td></tr>
        <tr><td><b>9</b></td><td>First design is overwhelmingly better under this subcriterion</td></tr>
        <tr><td><b>2, 4, 6, 8</b></td><td>Intermediate judgments between the above levels</td></tr>
    </table>
    </div>

    <div class="example">
    <b>📌 Example:</b><br>
    If you set <b>Design A vs Design B (Noise) = 5</b>, you’re saying <i>"Design A"</i> is <b>clearly better</b> than <i>"Design B"</i> in minimizing noise impact.
    </div>
    """, unsafe_allow_html=True)


# === TAB 2: COMMUNITY INPUT ===
with tabs[1]:
    st.subheader("Community Input: General Feedback")
    col1, col2 = st.columns([2, 1])  # Wider left column for form, right for image
    create_sheet_if_missing(sheet, "Community Feedback Tab")
    ensure_headers_if_missing(sheet, "Community Feedback Tab")

    with col1:
        name = st.text_input("Name")
        community = st.text_input("Community")
        feedback = st.text_area("What are your thoughts about WEC in your community? (Don't include any personal information below)")
        #wec_type = st.selectbox("Select a WEC Design", WEC_DESIGNS)

        if st.button("✉️ Submit Feedback"):
            try:
                prompt = f"""
                You are an expert assistant in converting qualitative community feedback into Weighted Hesitant Fuzzy Scores (WHFS) for 4 themes:

                {THEMES}

                For each theme, return:
                1. A short explanation (1-2 sentences)
                2. A list of 2 to 3 triangular fuzzy numbers (TFNs), each in the format [a, b, c], where 0 ≤ a ≤ b ≤ c ≤ 9
                The scoring should reflect **level of community concern/impact**, with:
                - 0 = No concern/impact
                - 9 = Very high concern/impact
                {{
                    "Scenic Value": {{
                        "explanation": "The design does not obstruct scenic views.",
                        "scores": [[1, 2, 3], [2, 3, 4]]
                    }},
                    "Community Benefits": {{
                        "explanation": "Expected to create jobs and educational opportunities.",
                        "scores": [[5, 6, 7], [6, 7, 8]]
                    }},
                    ...
                }}

                Community Feedback:
                \"\"\"{feedback}\"\"\"
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )

                import re
                raw_response = response.choices[0].message.content.strip()

                # Remove code block markers if present
                if raw_response.startswith("```"):
                    raw_response = re.sub(r"^```(?:json)?", "", raw_response)
                    raw_response = re.sub(r"```$", "", raw_response)
                    raw_response = raw_response.strip()

                try:
                    result = json.loads(raw_response)
                except Exception as e:
                    st.error(f"GPT failed to return valid JSON. Error: {e}")
                    st.stop()

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
    ensure_headers_if_missing(sheet, "Final Rankings")
    
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

            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
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

            # Theme Weight Visualization (similar to previous version)
            st.markdown("### 📉 Visual Theme Weights (Community WHFS)")

            labels = list(theme_weights.keys())
            values = list(theme_weights.values())

            from textwrap import fill
            wrapped_labels = [fill(label, width=12) for label in labels]

            fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Compact size
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

        except Exception as e:
            st.error(f"Error during ranking: {e}")

st.markdown("---")
st.markdown("### ⚠️ Danger Zone")
if st.button("🧹 Clear All Data from Google Sheets"):
    clear_all_google_sheet_tabs()

st.markdown(
    "<hr style='margin-top: 50px;'><div style='text-align: center; font-size: 18px; color: gray;'>© 2025 Vishnu Vijayasankar. All rights reserved.</div>",
    unsafe_allow_html=True
)