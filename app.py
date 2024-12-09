from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

package_tourism = pd.read_csv('package_tourism.csv')
user_data = pd.read_csv('user.csv')
tourism_rating = pd.read_csv('tourism_rating.csv')
tourism_with_id = pd.read_csv('tourism_with_id.csv')

def preprocess_data():
    # Merge data for collaborative filtering
    merged_data = tourism_rating.merge(tourism_with_id, on='Place_Id')
    return merged_data

merged_data = preprocess_data()



@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user_id = request.form['user_id']
        if user_id.isdigit() and int(user_id) in user_data['User_Id'].values:
            return redirect(url_for('index', user_id=user_id))
        else:
            error = 'Invalid User ID. Please try again.'
    return render_template('login.html', error=error)

@app.route('/index/<user_id>', methods=['GET', 'POST'])
def index(user_id):
    recommendations = []
    if request.method == 'POST':
        place_name = request.form['place_name']
        recommendations = content_based_recommendations(place_name)
    else:
        recommendations = collaborative_recommendations(user_id)
    return render_template('index.html', recommendations=recommendations)

@app.route('/content_recommendations', methods=['POST'])
def content_recommendations():
    place_name = request.form['place_name']
    searched_place, recommendations = content_based_recommendations(place_name, return_searched=True)
    return render_template('content_recommendations.html', searched_place=searched_place, recommendations=recommendations)

def collaborative_recommendations(user_id):
    # Collaborative Filtering 
    user_ratings = merged_data.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
    user_similarity = cosine_similarity(user_ratings.fillna(0))
    user_idx = user_data[user_data['User_Id'] == int(user_id)].index[0]

    similar_users = user_similarity[user_idx]
    similar_user_indices = np.argsort(-similar_users)
    
    similar_places = []
    for similar_idx in similar_user_indices:
        similar_user_id = user_data.iloc[similar_idx]['User_Id']
        rated_places = tourism_rating[tourism_rating['User_Id'] == similar_user_id]
        for _, place in rated_places.iterrows():
            if place['Place_Id'] not in similar_places:
                similar_places.append(place['Place_Id'])
                if len(similar_places) >= 5:
                    break
        if len(similar_places) >= 10:
            break

    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(similar_places)].to_dict(orient='records')
    return recommendations

def content_based_recommendations(place_name, return_searched=False):
    # Content-Based 
    tfidf = TfidfVectorizer(stop_words='english')
    tourism_with_id['Description'] = tourism_with_id['Description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(tourism_with_id['Description'])

    if place_name not in tourism_with_id['Place_Name'].values:
        return [] if not return_searched else (None, [])

    place_idx = tourism_with_id[tourism_with_id['Place_Name'] == place_name].index[0]
    cosine_similarities = cosine_similarity(tfidf_matrix[place_idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]

    recommendations = tourism_with_id.iloc[similar_indices].to_dict(orient='records')
    searched_place = tourism_with_id.iloc[place_idx].to_dict() if return_searched else None
    return recommendations if not return_searched else (searched_place, recommendations)

if __name__ == '__main__':
    app.run(debug=True)