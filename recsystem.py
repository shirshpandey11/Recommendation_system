import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = {
    'Title': ['Extraction', 'Superbad', 'Inception', 'The Dark Knight', 'Rush Hour'],
    'Genres': ['Action', 'Comedy', 'Drama', 'Action, Drama', 'Comedy, Drama'],
    'Description': ['Action-packed movie with thrilling scenes.',
                    'Hilarious comedy with funny characters.',
                    'Emotional drama that tugs at the heartstrings.',
                    'A mix of action and drama in this gripping film.',
                    'Comedy-drama with a heartwarming story.'],
}

df = pd.DataFrame(data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_preference, cosine_sim=cosine_sim):
    idx = df[df['Title'] == user_preference].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]

    # Returning the top 5 recommended movie titles
    return df['Title'].iloc[movie_indices]

user_preference = input("Enter your preferred movie: ")
recommendations = get_recommendations(user_preference)

if not recommendations.empty:
    print("Recommended Movies for '{}' are:".format(user_preference))
    for movie_title in recommendations:
        print(movie_title)
else:
    print("Movie not found in the dataset.")
