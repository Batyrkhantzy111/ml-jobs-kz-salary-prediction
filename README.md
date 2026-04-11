# Python related Jobs predictions in Kazakhstan - salary prediction

Analysis of 1000+ Data Science/ML/AI/Python/DevOps etc. job postings from hh.kz  
with an interactive salary prediction app based on tech stack

**[Live Demo](#)** · **[Dataset](#)**

---

## Stack
`Python` `pandas` `scikit-learn` `XGBoost` `SHAP`  
`Plotly` `Streamlit` `BeautifulSoup` `requests`

## Results *(will be updated)*
- **2342 vacancies** collected from hh.kz (april 2026)
- Best model: R², MAE
- Top salary predictors:

## How to run
```bash
pip install -r requirements.txt

# Collect data
notebooks/01_extract_data.ipynb

# Run app (will be then)
streamlit run app.py
```

## Project Structure
```
├── notebooks/
│   ├── 01_extract_data.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_nlp.ipynb
│   └── 04_model.ipynb
├── data/raw/vacancies.csv
├── models/salary_model
├── app.py
└── requirements.txt
```
