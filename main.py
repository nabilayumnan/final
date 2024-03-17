import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define functions
def pre_processor_clean(titles):
    # Your pre_processor_clean function code here...
    movies_title = titles.copy()
    for feature in features:
        movies_title[feature] = movies_title[feature].fillna('')
    return movies_title

def combine_features_string(current):
    rows = ""
    for feature in ['title', 'type', 'genres', 'description',
                    'imdb_score', 'tmdb_popularity', 'tmdb_score']:
        rows += f'{current[feature]}\n '
    return rows   # Your combine_features_string function code here...

def get_index_using_title(title, movies):
    return movies[movies["title"] == title].index[0]    # Your get_index_using_title function code here...

def select_movie(movies, movie_title, cosine_similarity_rm, number_of_recommendations):
    similar_movies = list(enumerate(cosine_similarity_rm[get_index_using_title(movie_title, movies)]))
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]
    sorted_similar_movies = sorted_similar_movies[:number_of_recommendations]

    df_recommender = {"_id": [], "title": [], "description": [], "confidence": []}
    for i, similarity_movie in enumerate(sorted_similar_movies):
        index_movie, confidence = similarity_movie[0], similarity_movie[1]
        filter_movie = movies.iloc[index_movie]
        df_recommender["_id"].append(index_movie)
        df_recommender["title"].append(filter_movie['title'])
        df_recommender["description"].append(filter_movie['description'])
        df_recommender["confidence"].append(confidence)

    return pd.DataFrame(df_recommender)
   # Your select_movie function code here...

# Load data
# Load your dataset here, e.g., titles = pd.read_csv('your_dataset.csv')
titles = pd.read_csv('titles.csv')

features = ['title', 'type', 'genres', 'description', 'imdb_score', 'tmdb_popularity', 'tmdb_score']

# Preprocess data
movies_title = pre_processor_clean(titles)
movies_title['features'] = movies_title.apply(combine_features_string, axis=1)

# Vectorize features
vectorizer = CountVectorizer()
matrix_transform = vectorizer.fit_transform(movies_title["features"])
cosine_similarity_rm = cosine_similarity(matrix_transform)

# Streamlit app
st.title("Movie Recommender")

movie_input = st.text_input("Enter a movie title", "Fight For My Way")
number_of_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)

if st.button("Get Recommendations"):
    movies_similiraty = select_movie(movies_title, movie_input, cosine_similarity_rm, number_of_recommendations)
    st.dataframe(movies_similiraty)

