import streamlit as st
import pandas as pd
import pickle
import sqlite3
from sklearn.metrics.pairwise import linear_kernel

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data():
    """Load the pre-computed TF-IDF matrix and index mapping."""
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    with open('indices.pkl', 'rb') as f:
        indices = pickle.load(f)
        
    return tfidf_matrix, indices

tfidf_matrix, indices = load_data()

def load_db():
    """Load movie metadata from SQL to ensure perfect index alignment."""
    conn = sqlite3.connect('movies.db')
    df = pd.read_sql("SELECT * FROM movies", conn)
    conn.close()
    return df

df_movies = load_db()

# --- 2. RECOMMENDATION ENGINE ---
def get_recommendations(title):
    """
    Finds similar movies using Cosine Similarity + Genre Filtering.
    Returns: List of integer indices for the recommended movies.
    """
    try:
        # Get index of the selected movie
        idx = indices[title]
        
        # Determine the primary genre of the source movie
        source_genre = str(df_movies.iloc[idx]['Genre']).lower()
        
        # Priority list for "Smart Filtering"
        priority_genres = ['action', 'comedy', 'drama', 'horror', 'romance', 
                           'superhero', 'sci-fi', 'animation', 'thriller', 'crime']
        
        required_tag = None
        for genre in priority_genres:
            if genre in source_genre:
                required_tag = genre
                break
        
        # Calculate similarity scores (linear_kernel is faster than cosine_similarity)
        cosine_sim_vector = linear_kernel(tfidf_matrix[idx], tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim_vector[0]))
        
        # Sort by similarity (Highest score first)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Filter top 50 results based on Genre match
        final_recs = []
        for item in sim_scores[1:51]:
            movie_idx = item[0]
            target_genre = str(df_movies.iloc[movie_idx]['Genre']).lower()
            
            # Strict Filter: Enforce matching genre for key categories
            if required_tag and required_tag not in target_genre:
                continue
                
            final_recs.append(movie_idx)
            
            # Stop after finding 5 good matches
            if len(final_recs) >= 5:
                break
        
        return final_recs
        
    except Exception:
        return []

# --- 3. FRONTEND INTERFACE ---
st.title("🎬 Movie Recommendation System")

# Dropdown for movie selection
selected_movie = st.selectbox("Type or Select a Movie:", df_movies['Title'].values)

if st.button('Show Recommendations'):
    with st.spinner('Thinking...'):
        rec_indices = get_recommendations(selected_movie)
        
    if rec_indices:
        st.success(f"Because you liked **{selected_movie}**, you might like:")
        
        # Create 5 columns for the results
        cols = st.columns(5)
        for i, col in enumerate(cols):
            if i < len(rec_indices):
                idx = rec_indices[i]
                row = df_movies.iloc[idx]
                
                with col:
                    st.subheader(row['Title'])
                    st.caption(f"Genre: {row['Genre']}")
                    st.text(f"Dir: {row['Director']}")
                    
                    # Expandable plot summary
                    with st.expander("Plot"):
                        st.write(str(row['Plot'])[:150] + "...")
    else:
        st.error("Movie not found or no good matches!")