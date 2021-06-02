import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle as pkl
import numpy as np
import gzip

@st.cache
def load_metadata():
    return pd.read_pickle('metadata.pkl')

@st.cache
def load_similarity():
    with gzip.open('similarity.pkl.gz', 'rb') as f:
        p = pkl.load(f)
    return p



def get_recommendations(title, cosine_sim, indices, metadata, n = 10):
    if type(title) == list:
        ids = [indices[t] for t in title]
        
        sim_scores = list(enumerate(cosine_sim[ids].sum(axis = 0)))
        
    else:
        # Get the index of the movie that matches the title
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:n + 1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata.iloc[movie_indices]

def main():

    st.title('IMBD Movie Sugestion')
    metadata = load_metadata()
    similarity = load_similarity()

    indices = pd.Series(np.arange(metadata.shape[0]), index = metadata['title'])

    movie_name = st.selectbox('Select Movie', metadata['title'].values)

    m = metadata.iloc[indices[movie_name]]

    st.subheader(m['title'])

    # st.text(m['tagline'])

    st.write(m['overview'])

    similar_movies = get_recommendations(
            movie_name, 
            similarity, 
            indices, 
            metadata, 
            n = 20
        )


    st.write(similar_movies[['title', 'vote_average', 'vote_count', 'year']])




    # st.dataframe(
    #     metadata[:100][['original_title', 'vote_average', 'vote_count', 'wr', 'release_date']]
    #     )



if __name__ == '__main__':

    main()