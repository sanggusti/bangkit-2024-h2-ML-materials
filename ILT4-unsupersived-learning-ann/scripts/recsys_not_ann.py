import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem.snowball import SnowballStemmer
import numpy as np

def load_data():
    movies = pd.read_csv('https://drive.google.com/uc?id=1xs4RldQMooikknISna4Hcp2uYzOvq1JX')
    credits = pd.read_csv('https://drive.google.com/uc?id=1G8Z1X9x5z8X8z8X8z8X8z8X8z8X8z8X')
    return movies, credits

def preprocess_data(movies, credits):
    # Merge movies and credits data
    movies = movies.merge(credits, on='id')
    
    # Fill NaN values with empty string
    movies['overview'] = movies['overview'].fillna('')
    
    # Stemming process
    stemmer = SnowballStemmer("english")
    movies['overview'] = movies['overview'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    
    return movies

def compute_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    return tfidf_matrix

def compute_cosine_similarity(tfidf_matrix):
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim, movies):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

def main():
    movies, credits = load_data()
    movies = preprocess_data(movies, credits)
    tfidf_matrix = compute_tfidf_matrix(movies)
    cosine_sim = compute_cosine_similarity(tfidf_matrix)
    
    # Example usage
    title = 'The Dark Knight'
    recommendations = get_recommendations(title, cosine_sim, movies)
    print(f"Recommendations for '{title}':")
    print(recommendations)

if __name__ == "__main__":
    main()