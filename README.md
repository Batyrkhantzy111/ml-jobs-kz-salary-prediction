# KZ IT JobScope - IT job market analysis in Kazakhstan

Analysis of 1000+ IT job postings from hh.kz across Kazakhstan,
with exploratory data analysis, skill extraction, an interactive salary prediction app based on tech stack, and role features

**[Live Demo](#)** · **[Dataset](data/processed/vacancies_clean_2026-04-14.csv)**

---

## Overview
This project analyzes vacancies from hh.kz to explore the Kazakhstan IT job market

The pipeline includes:
- data collection from hh.kz
- data cleaning and preprocessing
- exploratory data analysis
- skill extraction from vacancy descriptions
- salary prediction using machine learning
- interactive Streamlit app for market insights and salary estimation

## Stack
`Python` `pandas` `scikit-learn` `XGBoost` `SHAP`  
`Plotly` `kaleido` `Streamlit` `BeautifulSoup` `requests`

## Results *(will be updated)*
- **1177 (already processed) vacancies** collected from hh.kz (april 2026)
- Best model: R², MAE
- Top salary predictors:

## How to run
```bash
pip install -r requirements.txt

# Collect data
notebooks/00_extract_data.ipynb
# Explore and
notebooks/01_eda.ipynb

# Run app (will be then)
streamlit run app.py
```

## Project Structure
```
├── notebooks/
│   ├── 00_extract_data.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_nlp.ipynb
│   └── 03_model.ipynb
├── data/
│ ├── raw/
│ │ └── raw_vacancies.csv
│ ├── processed/
│ │ └── vacancies_clean.csv
│ │ └── features_nlp.csv
│ └── vizualizations/
│ ├── viz_1.png
│ ├── viz_1.html
│ ├── viz_2.png
│ ├── viz_2.html
│ └── ...
├── models/salary_model.pkl
├── app.py
└── requirements.txt
```
