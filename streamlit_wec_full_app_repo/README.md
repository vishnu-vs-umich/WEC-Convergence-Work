# Streamlit App: WEC Decision Support Tool (Fully Integrated)

This Streamlit app evaluates Wave Energy Converter (WEC) designs using **Fuzzy AHP + TOPSIS**, AI-assisted fuzzy score extraction, and **live community feedback from Google Sheets**.

## Features
- **Fuzzy AHP for theme weighting**
- **Fuzzy TOPSIS for ranking WECs**
- **AI-based fuzzy score extraction from community feedback**
- **Live data integration from Google Forms → Google Sheets**
- **PDF report generation for final decision output**

## Deployment
1. Upload this repo to GitHub
2. Deploy via [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Set your OpenAI API key in Streamlit Cloud settings → Secrets:
```toml
OPENAI_API_KEY = "your-api-key"
```
4. Set up Google Sheets API credentials for live community feedback.

---
**Developed as part of a PhD research decision support tool.**
