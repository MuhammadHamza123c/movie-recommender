import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import requests

# TMDb API details
api_key = 'fcd82d6941bdac53e63708ffcfbb955a'
base_url = "https://api.themoviedb.org/3"
image_base_url = "https://image.tmdb.org/t/p/w500"

# Load movie data
data = pd.read_csv("D:/Code/movie_tags.csv")
data = data.head(5000)

# Clean 'tags' column by removing numbers
data['tags'] = data['tags'].apply(lambda x: re.sub(r'\d+(\.\d+)?', '', x))

# Create TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=10000)
change_ = vectorizer.fit_transform(data['tags'])
matrix_array = change_.toarray()

# Compute cosine similarity
similar = cosine_similarity(matrix_array)

# Streamlit interface
st.title("Movie Recommender System")

# Get user input
user_input = st.text_input("Write down a movie name here: ").strip()

if user_input:
    # Find the movie based on user input (case-insensitive)
    new = data[data['title'].str.lower() == user_input.lower()]

    # Ensure that the movie exists
    if not new.empty:
        new_index = new.index[0]

        # Get sorted list of similar movies
        sorted_list = sorted(enumerate(similar[new_index]), key=lambda x: x[1], reverse=True)[1:4]
        indices = [index for index, _ in sorted_list]

        st.write(f"Movies similar to **{user_input}**:")

        # Display movies in a single row
        cols = st.columns(3)  # Create three columns for 3 movies

        # Fetch movie posters and titles
        i = 0
        for movie_index in indices:
            movie_name = data['title'].iloc[movie_index]

            # Ensure the movie isn't the same as the user input
            if movie_name.lower() != user_input.lower():
                # Fetch movie poster from TMDb API
                params = {'api_key': api_key, 'query': movie_name}
                response = requests.get(f"{base_url}/search/movie", params=params)

                poster_url = None
                if response.status_code == 200:
                    tmdb_data = response.json()
                    if tmdb_data['results']:
                        movie = tmdb_data['results'][0]
                        poster_path = movie.get('poster_path', None)
                        if poster_path:
                            poster_url = f"{image_base_url}{poster_path}"

                # Display the movie poster and title in the appropriate column
                with cols[i]:
                    if poster_url:
                        st.image(poster_url, width=150)
                    st.write(f"**{movie_name}**")

                i += 1  # Move to the next column
    else:
        st.write("Movie not found in the database.")
