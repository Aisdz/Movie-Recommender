# 🎬 Movie Recommendation System

## Project Overview
A content-based recommendation engine built with Python. It analyzes movie plots, genres, and directors to suggest similar movies.

## Key Features
* **Smart Filtering:** Enforces genre consistency (e.g., Action movies recommend other Action movies).
* **NLP Engine:** Uses TF-IDF and Cosine Similarity.
* **Interactive UI:** Built with Streamlit for easy user interaction.

## Files
* `Data_Preprocessing.ipynb`: The code for cleaning data and training the model.
* `app.py`: The main script for the web application.

## How to Run
1. Install libraries: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`