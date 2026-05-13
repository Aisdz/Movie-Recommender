# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![ML](https://img.shields.io/badge/ML-TF--IDF%20%7C%20Cosine%20Similarity-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Dataset:** [Movie Dataset](https://drive.google.com/file/d/1iJIxjywm5HxoYytJc0Wn_n-VqLXlIRTJY/view?usp=drive_link)

---

## 📌 Project Overview

A content-based movie recommendation system built with Python.
Analyzes movie plots, genres, directors, and cast to recommend similar movies.

---

## How it works

1. Text preprocessing — cleaning metadata
2. Feature engineering — building a weighted "soup" string
3. TF-IDF vectorization
4. Cosine similarity computation
5. Genre-based filtering
6. Top-10 recommendations output

---

## 📁 Project structure

```
├── app.py
├── movies.db
├── tfidf_matrix.pkl
├── indices.pkl
├── movie_dataset.csv
└── README.md
```

---

## How to run

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

```
http://localhost:8501
```


<img width="1680" height="960" alt="Screenshot 2026-05-13 at 14 52 44" src="https://github.com/user-attachments/assets/c98a594c-5d4e-4377-bd57-a8efdb6d5610" />
<img width="1680" height="961" alt="Screenshot 2026-05-13 at 14 54 37" src="https://github.com/user-attachments/assets/118dbbec-fe59-4ed1-a84c-1b1eba3b7f86" />



