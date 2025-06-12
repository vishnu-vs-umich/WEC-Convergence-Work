import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from openai import OpenAI
import matplotlib.pyplot as plt
from PIL import Image
import ast
import seaborn as sns
import openpyxl
from openpyxl import Workbook, load_workbook
import os


EXCEL_FILE = "bluechoice_data.xlsx"

def read_excel(sheet_name):
    if not os.path.exists(EXCEL_FILE):
        wb = Workbook()
        wb.save(EXCEL_FILE)
    wb = load_workbook(EXCEL_FILE)
    if sheet_name not in wb.sheetnames:
        wb.create_sheet(sheet_name)
        wb.save(EXCEL_FILE)
    ws = wb[sheet_name]
    return ws, wb

def write_excel(sheet_name, data, mode='append'):
    ws, wb = read_excel(sheet_name)
    if mode == 'overwrite':
        ws.delete_rows(1, ws.max_row)
    for row in data:
        ws.append(row)
    wb.save(EXCEL_FILE)

# ✅ FIRST Streamlit command
st.set_page_config(layout="wide")

# === Config and setup continues below ===
WEC_DESIGNS = ["Point Absorber", "OWC", "Oscillating Surge Flap"]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_data(ttl=600)
def load_themes_and_subcriteria(sheet=None):
    ws, _ = read_excel("Themes Tab")
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = rows[1:]
    theme_map = {}
    for row in data:
        theme, sub = row
        if theme and sub:
            theme_map.setdefault(theme.strip(), []).append(sub.strip())
    return list(theme_map.keys()), theme_map

def connect_to_google_sheets():
    return EXCEL_FILE

def create_themes_tab_with_subcriteria(sheet=None):
    ws, wb = read_excel("Themes Tab")
    if ws.max_row > 1:
        return  # Already initialized
    ws.append(["Theme", "Subcriterion"])
    theme_data = [
        ("Functional Efficiency", "Electricity reliability and security"),
        ("Functional Efficiency", "Electricity affordability"),
        ("Environmental Sustainability", "Habitat of marine animals"),
        ("Environmental Sustainability", "Birds"),
        ("Environmental Sustainability", "Health"),
        ("Environmental Sustainability", "Noise"),
        ("Sense of Place", "Personal identity / connection to place"),
        ("Sense of Place", "Ocean / landscape view"),
        ("Community Prosperity", "Community benefits"),
        ("Community Prosperity", "Job opportunities"),
        ("Community Prosperity", "Tax revenues"),
        ("Community Prosperity", "Indirect economic effects"),
        ("Community Prosperity", "Tourism"),
        ("Marine Space Utilization", "Recreational fishing"),
        ("Marine Space Utilization", "Recreational boating")
    ]
    for theme, sub in theme_data:
        ws.append([theme, sub])
    wb.save(EXCEL_FILE)


sheet = connect_to_google_sheets()
create_themes_tab_with_subcriteria(sheet)  # Make sure the sheet exists
THEMES, THEME_SUBCRITERIA = load_themes_and_subcriteria(sheet)

def create_sheet_if_missing(sheet_path, title):
    """Creates a new sheet in the local Excel file if it doesn't exist."""
    if not os.path.exists(sheet_path):
        # Create workbook and add the requested sheet
        wb = openpyxl.Workbook()
        wb.remove(wb.active)  # remove default sheet
        wb.create_sheet(title)
        wb.save(sheet_path)
        return

    wb = openpyxl.load_workbook(sheet_path)
    if title not in wb.sheetnames:
        wb.create_sheet(title)
        wb.save(sheet_path)


def ensure_headers_if_missing(title):
    ws, wb = read_excel(title)  # Load worksheet and workbook

    existing_rows = list(ws.iter_rows(values_only=True))

    if not existing_rows or all(all(cell in [None, ""] for cell in row) for row in existing_rows):
        ws.delete_rows(1, ws.max_row)  # Clear existing content if any

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
            ws.append(headers)
            wb.save(EXCEL_FILE)

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
    ensure_headers_if_missing("Expert Tab")
    create_sheet_if_missing(sheet, "Expert Contributions Tab")
    ensure_headers_if_missing("Expert Contributions Tab")

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
        for sub_idx, subcriterion in enumerate(THEME_SUBCRITERIA[theme], 1):
            st.markdown(
                f"<h5 style='font-weight:600; margin-left:10px;'>{sub_idx}. {subcriterion}</h5>",
                unsafe_allow_html=True,
            )

            pcm = np.ones((len(WEC_DESIGNS), len(WEC_DESIGNS)))
            for i in range(len(WEC_DESIGNS)):
                for j in range(i + 1, len(WEC_DESIGNS)):
                    st.caption(f"⬅️ Slide left if **{WEC_DESIGNS[j]}** is better, ➡️ slide right if **{WEC_DESIGNS[i]}** is better.")

                    comparison_question = f"""
                        <h3 style='margin-top: 10px; margin-bottom: 6px; font-size: 18px; color:#00274C;'>
                        🔍 In terms of <span style="color:#FFCB05;"><strong>{subcriterion}</strong></span>, how much better is <span style="color:#1f77b4;"><strong>{WEC_DESIGNS[i]}</strong></span> than 
                        <span style="color:#d62728;"><strong>{WEC_DESIGNS[j]}</strong></span>?
                        </h3>
                        """
                    st.markdown(comparison_question, unsafe_allow_html=True)
                    st.markdown("""
                    <div style='
                        display: flex;
                        justify-content: space-between;
                        background-color: #f9f9f9;
                        padding: 6px 0;
                        margin-bottom: -10px;
                        border: 1px solid #ccc;
                        border-radius: 6px;
                        font-size: 13px;
                        font-weight: 500;
                        color: #333;
                    '>
                        <div>-9</div>
                        <div>-7</div>
                        <div>-5</div>
                        <div>-3</div>
                        <div>1</div>
                        <div>3</div>
                        <div>5</div>
                        <div>7</div>
                        <div>9</div>
                    </div>
                    """, unsafe_allow_html=True)

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
                        label="",
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
            sheet_name = "Expert Contributions Tab"

            # Define headers
            header = ["Timestamp", "Name", "Theme", "Subcriterion", "Expertise Level"]
            for i in range(len(WEC_DESIGNS)):
                for j in range(i + 1, len(WEC_DESIGNS)):
                    header.append(f"PCM_{WEC_DESIGNS[i]}_vs_{WEC_DESIGNS[j]}")

            # Construct new rows
            new_rows = []
            for (theme, subcriterion), pcm in expert_pcms.items():
                flat_pcm = []
                for i in range(len(WEC_DESIGNS)):
                    for j in range(i + 1, len(WEC_DESIGNS)):
                        flat_pcm.append(pcm[i, j])
                level = expertise_levels[theme]
                row = [timestamp, expert_name, theme, subcriterion, level] + flat_pcm
                new_rows.append(row)

            new_df = pd.DataFrame(new_rows, columns=header)

            try:
                # Try to load existing data
                existing_df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except FileNotFoundError:
                # File doesn't exist; create new
                combined_df = new_df
            except ValueError:
                # Sheet doesn't exist; treat as new
                combined_df = new_df

            # Write back to Excel
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

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

                # === Construct the row to be saved ===
                row = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name,
                    community
                ] + [fuzzy_list_to_str(result.get(theme)) for theme in THEMES]

                # === Read the local worksheet and workbook ===
                ws, wb = read_excel("Community Feedback Tab")

                # === Check if headers are missing (first time write) ===
                if ws.max_row == 0:
                    headers = ["Timestamp", "Name", "Community"] + THEMES
                    ws.append(headers)

                # === Append the data row and save ===
                ws.append(row)
                wb.save(EXCEL_FILE)

                st.success("✅ Feedback saved successfully!")

                # === Display parsed AI-extracted feedback ===
                st.markdown("### AI-Assisted Feedback Breakdown by Theme")


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
            # === Load Expert Scores and Community Feedback from Excel ===
            exp_ws, _ = read_excel("Expert Tab")
            comm_ws, _ = read_excel("Community Feedback Tab")

            exp_df = pd.DataFrame(exp_ws.values)
            comm_df = pd.DataFrame(comm_ws.values)

            # Clean and assign headers
            exp_df.columns = exp_df.iloc[0]
            exp_df = exp_df[1:]
            comm_df.columns = comm_df.iloc[0]
            comm_df = comm_df[1:]

            exp_df.set_index("WEC", inplace=True)
            exp_matrix = exp_df.loc[WEC_DESIGNS][THEMES].astype(float).values

            theme_weights = aggregate_whfs_scores(comm_df)
            weights = np.array([theme_weights[t] for t in THEMES])

            closeness = topsis(exp_matrix, weights)
            result_df = pd.DataFrame({
                "WEC Design": WEC_DESIGNS,
                "Closeness to Ideal": closeness
            }).sort_values(by="Closeness to Ideal", ascending=False)

            # === Show Outputs ===
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("### 📊 Theme Weights")
                st.dataframe(pd.DataFrame.from_dict(theme_weights, orient='index', columns=["Weight"]))

            st.markdown("### 🔍 Expert Score Matrix")
            st.dataframe(exp_df)

            st.markdown("### 🏆 TOPSIS Results")
            st.dataframe(result_df)

            # === Save to Final Rankings sheet ===
            ws, wb = read_excel("Final Rankings")
            ws.delete_rows(1, ws.max_row)  # Clear existing content
            ws.append(["WEC Design", "Closeness to Ideal"])
            for row in result_df.values.tolist():
                ws.append(row)
            wb.save(EXCEL_FILE)

            # === Theme Weight Visualization ===
            st.markdown("### 📉 Visual Theme Weights (Community WHFS)")
            labels = list(theme_weights.keys())
            values = list(theme_weights.values())

            from textwrap import fill
            wrapped_labels = [fill(label, width=12) for label in labels]

            fig, ax = plt.subplots(figsize=(3.5, 2.5))
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