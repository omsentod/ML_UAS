import pandas as pd
import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import os

app = Flask(__name__)

# Set secret_key untuk session
app.secret_key = os.urandom(24)  # Menggunakan key acak yang lebih aman

# Load datasets
package = pd.read_csv('package_tourism.csv')
tourism = pd.read_csv('tourism_with_id.csv')
rating = pd.read_csv('tourism_rating.csv')
user = pd.read_csv('user.csv')

# Preprocess the tourism data
def preprocess_data(tourism, rating):
    tourism = tourism.drop(['Description', 'City', 'Price', 'Time_Minutes', 'Coordinate'], axis=1)
    return tourism

# Collaborative Filtering using Surprise library
def collaborative_filtering(user_id):
    # Load ratings and apply collaborative filtering
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(rating[['User_Id', 'Place_Id', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train the model
    model = SVD()
    model.fit(trainset)

    # Get recommendations for the given user
    recommendations = []
    for place_id in tourism['Place_Id']:
        pred = model.predict(user_id, place_id)
        recommendations.append((place_id, pred.est))
    
    # Sort by predicted rating (highest first)
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 recommendations
    top_recommendations = recommendations[:5]

    # Get the corresponding places from the tourism dataset
    recommended_places = []
    for place_id, _ in top_recommendations:
        place = tourism[tourism['Place_Id'] == place_id].iloc[0]
        recommended_places.append({
            'Place_Name': place['Place_Name'],
            'City': place['City'],
            'Place_Id': place['Place_Id'],
            'Place_Ratings': place['Rating']  # Assuming there's a 'Rating' column in tourism
        })

    return recommended_places

# Route untuk halaman login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']
        session['user_id'] = user_id  # Simpan ID pengguna di session
        return redirect(url_for('index'))  # Arahkan ke halaman index setelah login
    return render_template('login.html')  # Halaman login

# Route untuk halaman index
@app.route('/')
def index():
    # Ambil user_id dari session
    user_id = session.get('user_id')
    if user_id:
        # Dapatkan rekomendasi berdasarkan collaborative filtering
        collaborative_recommendations = collaborative_filtering(user_id)
        return render_template('index.html', collaborative_recommendations=collaborative_recommendations)
    else:
        return redirect(url_for('login'))  # Jika user belum login, arahkan ke halaman login

# Route untuk halaman hasil pencarian berdasarkan Content-Based Filtering
@app.route('/content_recomendations', methods=['GET'])
def content_recomendations():
    return render_template('content_recommendations.html')

print(app.url_map)  # Ini akan mencetak semua rute yang terdaftar

if __name__ == '__main__':
    app.run(debug=True)
