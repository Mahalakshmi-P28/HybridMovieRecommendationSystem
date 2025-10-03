import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🎬 Hybrid Movie Recommender", page_icon="🎥", layout="centered")

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_movielens():
    # u.item has | separator
    movies = pd.read_csv("data/u.item", sep='|', header=None, encoding='latin-1',
                         names=["movieId","title","release_date","video_release_date","IMDb_URL",
                                "unknown","Action","Adventure","Animation","Children's",
                                "Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir",
                                "Horror","Musical","Mystery","Romance","Sci-Fi","Thriller",
                                "War","Western"])
    ratings = pd.read_csv("data/u.data", sep='\t', header=None, names=['userId','movieId','rating','timestamp'])
    return movies, ratings

movies, ratings = load_movielens()

# ----------------------------
# Prepare content features (genres)
# ----------------------------
genre_cols = movies.columns[6:]  # from 'unknown' to 'Western'
tags_scaled = pd.DataFrame(StandardScaler().fit_transform(movies[genre_cols]),
                           index=movies['movieId'], columns=genre_cols)

# ----------------------------
# Compute content similarity
# ----------------------------
@st.cache_data
def compute_content_similarity(matrix):
    sim = cosine_similarity(matrix, matrix)
    return pd.DataFrame(sim, index=matrix.index, columns=matrix.index)

content_sim_df = compute_content_similarity(tags_scaled)

# ----------------------------
# Hybrid Recommendation Function
# ----------------------------
def hybrid_recommendation(movie_name, top_n=10, alpha=0.5):
    movie_name_lower = movie_name.lower().strip()
    
    # Remove year from titles for matching
    movies['title_clean'] = movies['title'].str.replace(r"\(\d{4}\)","", regex=True).str.strip().str.lower()
    
    if movie_name_lower not in movies['title_clean'].values:
        return pd.DataFrame(columns=["movieId","title"]), 0.0
    
    movie_id = movies[movies['title_clean'] == movie_name_lower]['movieId'].values[0]
    
    # Content similarity
    if movie_id not in content_sim_df.index:
        return pd.DataFrame(columns=["movieId","title"]), 0.0
    
    content_scores = content_sim_df[movie_id].copy()
    content_scores.drop(movie_id, inplace=True)
    content_scores_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
    
    # Collaborative filtering: average ratings
    cf_scores = ratings.groupby('movieId')['rating'].mean()
    cf_scores_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
    
    # Keep only common movies
    common_movies = content_scores_norm.index.intersection(cf_scores_norm.index)
    
    hybrid_scores = alpha * cf_scores_norm[common_movies] + (1 - alpha) * content_scores_norm[common_movies]
    top_movies = hybrid_scores.sort_values(ascending=False).head(top_n)
    
    recommended_movies = movies[movies['movieId'].isin(top_movies.index)][['movieId','title']]
    avg_similarity = top_movies.mean()
    
    return recommended_movies, avg_similarity

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("""
#### Combines **Collaborative Filtering** + **Content-Based Filtering**  
Powered by Python, Streamlit, and scikit-learn.
""")
st.markdown("---")

movie_input = st.text_input("Enter a Movie Name (ignore year):", placeholder="e.g., Toy Story", max_chars=100)
top_n = st.slider("Number of Recommendations:", min_value=5, max_value=20, value=10)
alpha = st.slider("Weight for Collaborative Filtering (alpha):", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        recommendations, avg_similarity = hybrid_recommendation(movie_input, top_n=top_n, alpha=alpha)
    
    st.markdown("---")
    if recommendations.empty:
        st.warning("⚠️ Movie not found in database. Try another title.")
    else:
        st.subheader(f"Top {top_n} Recommendations for '{movie_input}':")
        for idx, row in enumerate(recommendations.itertuples(), 1):
            st.write(f"**{idx}. {row.title}**")
        st.markdown(f"**Average Similarity Score: {avg_similarity*100:.2f}%**")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: gray; font-size:12px;'>Built professionally by Mahalakshmi ❤️</p>",
    unsafe_allow_html=True
)
