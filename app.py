import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load datasets
package = pd.read_csv('package_tourism.csv')
tourism = pd.read_csv('tourism_with_id.csv')
rating = pd.read_csv('tourism_rating.csv')
user = pd.read_csv('user.csv')

# Preprocess the tourism data
def preprocess_data(tourism, rating):
    tourism = tourism.drop(['Description', 'City', 'Price', 'Rating', 'Time_Minutes', 
                            'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'], axis=1)
    all_tourism = pd.merge(rating, tourism, on="Place_Id", how="left")
    all_tourism = all_tourism.drop_duplicates('Place_Id').fillna(method='ffill')
    return all_tourism

all_tourism = preprocess_data(tourism, rating)

# Content-based Filtering using Cosine Similarity
def content_based_recommendations(place_name, top_n=5):
    data = all_tourism[['Place_Id', 'Place_Name', 'Category']]
    data['combined'] = data['Place_Name'] + ' ' + data['Category']
    tf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tf.fit_transform(data['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=data['Place_Name'], columns=data['Place_Name'])
    
    if place_name not in cosine_sim_df.columns:
        return pd.DataFrame(columns=['Place_Id', 'Place_Name', 'Category'])
    
    similar_places = cosine_sim_df[place_name].sort_values(ascending=False)[1:top_n+1]
    recommendations = data[data['Place_Name'].isin(similar_places.index)]
    return recommendations

# Collaborative Filtering using SVD (Singular Value Decomposition)
def collaborative_filtering_svd(rating):
    # Prepare the data for the SVD model
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
    
    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Build the SVD model
    model = SVD()
    model.fit(trainset)
    
    return model

# Get recommendations for a user using SVD
def get_collaborative_recommendations(model, user_id, top_n=5):
    # Get all places the user has not rated
    all_places = tourism['Place_Id'].unique()
    rated_places = rating[rating['User_Id'] == user_id]['Place_Id'].values
    unrated_places = [place for place in all_places if place not in rated_places]
    
    # Predict ratings for unrated places
    predictions = [model.predict(user_id, place_id) for place_id in unrated_places]
    
    # Sort by predicted ratings and return top N recommendations
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_predictions = predictions[:top_n]
    
    recommended_places = [pred.iid for pred in top_predictions]
    recommended_places = tourism[tourism['Place_Id'].isin(recommended_places)]
    
    return recommended_places

# Prepare city-specific recommendations
def get_city_packages(user_id):
    user_city = user[user['User_Id'] == user_id]['Location'].iloc[0].split(',')[0]
    city_packages = package[package['City'].str.contains(user_city, case=False, na=False)]
    return city_packages

# Initialize Collaborative Filtering Model
model_cf = collaborative_filtering_svd(rating)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    try:
        user_id = int(request.form['user_id'])
        if user_id not in user['User_Id'].values:
            return render_template('index.html', error="User ID tidak ditemukan. Silakan coba lagi.")
        
        # Collaborative Filtering recommendation
        user_city_packages = get_city_packages(user_id)
        
        # Render recommendations in the HTML page
        return render_template('recommendations.html', city_packages=user_city_packages.to_dict('records'))
    
    except ValueError:
        return render_template('index.html', error="User ID harus berupa angka. Silakan coba lagi.")

@app.route('/recommendation_content', methods=['POST'])
def recommendation_content():
    place_name = request.form['place_name']
    
    # Content-based filtering recommendation
    content_recommendations = content_based_recommendations(place_name, top_n=5)

    return render_template('recommendations.html', content_recommendations=content_recommendations.to_dict('records'))

@app.route('/recommendation_collaborative', methods=['POST'])
def recommendation_collaborative():
    try:
        user_id = int(request.form['user_id'])
        if user_id not in user['User_Id'].values:
            return render_template('index.html', error="User ID tidak ditemukan. Silakan coba lagi.")
        
        # Collaborative filtering recommendation using SVD
        collaborative_recommendations = get_collaborative_recommendations(model_cf, user_id, top_n=5)
        
        return render_template('recommendations.html', collaborative_recommendations=collaborative_recommendations.to_dict('records'))
    
    except ValueError:
        return render_template('index.html', error="User ID harus berupa angka. Silakan coba lagi.")

if __name__ == '__main__':
    app.run(debug=True)
