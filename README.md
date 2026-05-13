# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![ML](https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

Dataset:  
[Movie Dataset](https://drive.google.com/file/d/1iJIxjywm5HxoYytJc0Wn_n-VqLXlIRTJY/view?usp=drive_link)

---

## 📌 Project overview
This is a content-based movie recommendation system built with Python.  
It analyzes movie plots, genres, directors, and cast to recommend similar movies.

---
##  How it works
- Text preprocessing (cleaning metadata)
- Feature engineering ("soup")
- TF-IDF vectorization
- Cosine similarity computation
- Genre-based filtering
- Top-10 recommendations output

## 📁 Project Structure
app.py
movies.db
tfidf_matrix.pkl
indices.pkl
movie_dataset.csv
README.md

---
##  How to run

### 1. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn
### 2. Run the app
streamlit run app.py
### 3. Open in browser
http://localhost:8501

