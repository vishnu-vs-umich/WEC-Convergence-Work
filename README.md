BlueChoice: Community-Driven WEC Decision Support Tool
======================================================

BlueChoice is a decision-support application for Wave Energy Converter (WEC) selection. It integrates community-derived preferences with expert evaluations through multi-criteria decision-making techniques such as Analytic Hierarchy Process (AHP) and Weighted Hesitant Fuzzy Set-TOPSIS (WHFS-TOPSIS). The tool is built with Python and Streamlit and provides interactive visualizations and data management through Excel storage.

------------------------------------------------------
Features
------------------------------------------------------

- Multi-tab Streamlit interface:
  - Expert Input (AHP pairwise comparisons with consistency checks)
  - Community Feedback (AI-assisted WHFS fuzzy scoring)
  - Final Rankings (WHFS-TOPSIS integration with plots and Monte Carlo sensitivity analysis)
  - 3D Metrics Visualization (TRL vs 1/LCOE vs Social Acceptance)
- Local Excel storage (bluechoice_data.xlsx) with automatic sheet creation and header validation.
- Publication-quality plots (radar charts, stacked bar charts, scatter plots).
- Monte Carlo sensitivity with Kendall’s tau ranking stability analysis.
- Branding with University of Michigan colors (maize and blue).

------------------------------------------------------
Requirements
------------------------------------------------------

- Python 3.9 or higher
- Streamlit
- Pandas
- NumPy
- OpenPyXL
- Matplotlib
- Plotly
- Seaborn (optional, only for some analyses)
- OpenAI Python client (if AI-assisted scoring is enabled)

------------------------------------------------------
Installation
------------------------------------------------------

1. Clone the Repository

   git clone https://github.com/<your-username>/BlueChoice.git
   cd BlueChoice

2. Create a Virtual Environment

   Windows (PowerShell):
       python -m venv venv
       venv\Scripts\activate

   Linux / macOS (bash/zsh):
       python3 -m venv venv
       source venv/bin/activate

3. Install Dependencies

   pip install -r requirements.txt

   If requirements.txt is not present, create it with the following content:

   streamlit>=1.36.0
   pandas>=2.0.0
   numpy>=1.24.0
   openpyxl>=3.1.0
   matplotlib>=3.7.0
   plotly>=5.20.0
   seaborn>=0.13.0
   openai>=1.0.0

------------------------------------------------------
Running the Application
------------------------------------------------------

Windows:
    streamlit run Blue_Choice.py

Linux / macOS:
    streamlit run Blue_Choice.py

The application will start a local server and provide a URL (e.g., http://localhost:8501). Open this link in a web browser.

------------------------------------------------------
Data Storage
------------------------------------------------------

All application data is stored locally in an Excel file:

    bluechoice_data.xlsx

Tabs:
  - Expert Contributions Tab
  - Community Feedback Tab
  - AHP Decision Matrix
  - Final Rankings
  - Final AHP Theme Weights

Ensure the Excel file is writable in the working directory. If the file is missing, the application will create it automatically.

------------------------------------------------------
Cross-Platform Notes
------------------------------------------------------

- Windows: Use PowerShell or Command Prompt to run the application. Ensure Excel files are not open in another program, as this may lock them.
- Linux: Install Python 3.9+ using your package manager (e.g., sudo apt install python3 python3-venv).
- macOS: Use the system Python (python3) or install via Homebrew (brew install python).

------------------------------------------------------
Troubleshooting
------------------------------------------------------

- Port already in use: Run with a different port:
      streamlit run Blue_Choice.py --server.port 8502

- Excel file permission error: Close Excel or any program that has bluechoice_data.xlsx open.

- Dependency issues: Upgrade pip and reinstall requirements:
      python -m pip install --upgrade pip
      pip install -r requirements.txt

------------------------------------------------------
License
------------------------------------------------------

This project is licensed under the MIT License. See the LICENSE file for details.
