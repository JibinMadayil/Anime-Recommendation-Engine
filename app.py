from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the data and perform necessary preprocessing
df = pd.read_csv('MAL-all-from-winter1917-to-fall2023.csv')
df['genres'] = df['genres'].str.strip("[]").str.replace("'", "")
df['studio'] = df['studio'].str.strip("[]").str.replace("'", "")
df.rename(columns={'score': 'rating', 'genres': 'genre'}, inplace=True)
df.replace('', np.nan, inplace=True)
df = df.drop(['release-date', 'source-material', 'demographics', 'themes'], axis=1)
df.dropna(inplace=True)
df.drop_duplicates(subset='title', keep='first', inplace=True)
df.reset_index(drop=True, inplace=True)

# Create a CountVectorizer and fit_transform on the 'genre' column
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df['genre'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get anime recommendations based on similarity
def get_anime_recommendations(anime_title):
    anime_title_lower = anime_title.lower()
    try:
        idx = df[df['title'].str.lower() == anime_title_lower].index[0]
    except IndexError:
        raise ValueError(f"Anime '{anime_title}' not found in the dataset. Please check the spelling or try another anime.")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: (x[1], df['rating'].iloc[x[0]]), reverse=True)
    sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    recommendations_df = df.iloc[anime_indices][['title', 'rating', 'type', 'genre']]
    recommendations_df = recommendations_df.sort_values(by='rating', ascending=False)
    return recommendations_df

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# ...

# Define the route for the recommendation page
@app.route('/recommendation', methods=['POST'])
def recommendation():
    if request.method == 'POST':
        anime_title = request.form['anime_title']
        try:
            recommendations = get_anime_recommendations(anime_title)
            return render_template('recommendation.html', anime_title=anime_title, recommendations=recommendations)
        except ValueError as e:
            error_message = str(e)
            return render_template('recommendation.html', anime_title=anime_title, recommendations=pd.DataFrame(), error_message=error_message)

# ...

if __name__ == '__main__':
    app.run(debug=True)
