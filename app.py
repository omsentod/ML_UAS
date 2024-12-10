from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)  # Perbaikan __name__
app.secret_key = 'vavi'  # Ganti dengan string yang unik dan aman

# Load data
package_tourism = pd.read_csv('package_tourism.csv')
user_data = pd.read_csv('user.csv')
tourism_rating = pd.read_csv('tourism_rating.csv')
tourism_with_id = pd.read_csv('tourism_with_id.csv')

def preprocess_data():
    # Merge data for collaborative filtering
    merged_data = tourism_rating.merge(tourism_with_id, on='Place_Id')
    return merged_data

merged_data = preprocess_data()


# Load dataset
package_tourism_data = pd.read_csv('package_tourism.csv')
tourism_rating_data = pd.read_csv('tourism_rating.csv')
tourism_with_id_data = pd.read_csv('tourism_with_id.csv')
user_data_data = pd.read_csv('user.csv')

# Fungsi untuk memeriksa nilai yang hilang di setiap kolom
def check_missing_values(datasets, dataset_names):
    """
    Menampilkan jumlah nilai yang hilang di setiap kolom untuk dataset yang diberikan.

    Parameters:
    - datasets: list dari DataFrame
    - dataset_names: list dari nama dataset (string)
    """
    for df, name in zip(datasets, dataset_names):
        print(f"Missing values in {name}:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])  # Tampilkan hanya kolom dengan missing values
        print("-" * 40)


# Panggil fungsi untuk memeriksa missing value di setiap dataset
datasets = [package_tourism_data, tourism_rating_data, tourism_with_id_data, user_data_data]
dataset_names = ['package_tourism', 'tourism_rating', 'tourism_with_id', 'user_data']

check_missing_values(datasets, dataset_names)


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user_id = request.form['user_id']
        if user_id.isdigit() and int(user_id) in user_data['User_Id'].values:
            session['user_id'] = user_id  # Simpan user_id ke session
            return redirect(url_for('index', user_id=user_id))  # Arahkan ke halaman index dengan user_id
        else:
            error = 'Invalid User ID. Please try again.'
    return render_template('login.html', error=error)


@app.route('/index', methods=['GET', 'POST'])
def index():
    user_id = session.get('user_id')  # Ambil user_id dari session
    if not user_id:  # Jika tidak ada user_id, arahkan ke login
        return redirect(url_for('login'))

    recommendations = []
    if request.method == 'POST':
        # Tangani pencarian tempat
        place_name = request.form['place_name']
        recommendations = hybrid_recommendations(user_id, place_name)
    else:
        # Rekomendasi default berdasarkan pengguna
        recommendations = hybrid_recommendations(user_id, None)

    return render_template('index.html', recommendations=recommendations, user_id=user_id)

@app.route('/content_recommendations', methods=['GET', 'POST'])
def content_recommendations():
    user_id = session.get('user_id')  # Ambil user_id dari session
    if not user_id:  # Jika tidak ada user_id, arahkan ke login
        return redirect(url_for('login'))

    if request.method == 'POST':
        place_name = request.form['place_name']
    else:
        place_name = request.args.get('place_name', '')

    if not place_name:
        return redirect(url_for('index', user_id=user_id))

    searched_place, recommendations = hybrid_recommendations(user_id, place_name, return_searched=True)

    if searched_place is None:
        return render_template('error.html', message=f"Place '{place_name}' not found.")

    return render_template('content_recommendations.html', searched_place=searched_place, recommendations=recommendations, user_id=user_id)

# Route untuk log out
@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Hapus user_id dari session
    return redirect(url_for('login')) 

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


def hybrid_recommendations(user_id, place_name=None, return_searched=False):
    # Collaborative Filtering
    user_ratings = merged_data.pivot_table(index='User_Id', columns='Place_Id', values='Place_Ratings')
    user_similarity = cosine_similarity(user_ratings.fillna(0))
    user_idx = user_data[user_data['User_Id'] == int(user_id)].index[0]

    similar_users = user_similarity[user_idx]
    collaborative_scores = {}
    for similar_idx, score in enumerate(similar_users):
        similar_user_id = user_data.iloc[similar_idx]['User_Id']
        rated_places = tourism_rating[tourism_rating['User_Id'] == similar_user_id]
        for _, place in rated_places.iterrows():
            collaborative_scores[place['Place_Id']] = collaborative_scores.get(place['Place_Id'], 0) + score

    # Content-Based Filtering (Deskripsi dan Kategori)
    tfidf = TfidfVectorizer(stop_words='english')
    tourism_with_id['Description'] = tourism_with_id['Description'].fillna('')  # Mengisi missing description
    tfidf_matrix = tfidf.fit_transform(tourism_with_id['Description'])

    content_scores = {}
    searched_place = None

    # Jika user mencari tempat tertentu, ambil data tempat tersebut
    if place_name and place_name in tourism_with_id['Place_Name'].values:
        place_idx = tourism_with_id[tourism_with_id['Place_Name'] == place_name].index[0]
        searched_place = tourism_with_id.iloc[place_idx].to_dict()

        # Menghitung cosine similarity untuk tempat yang dicari dengan semua tempat lainnya
        cosine_similarities = cosine_similarity(tfidf_matrix[place_idx], tfidf_matrix).flatten()
        
        # Mengisi dictionary content_scores dengan nilai cosine similarity
        content_scores = {tourism_with_id.iloc[i]['Place_Id']: score for i, score in enumerate(cosine_similarities)}

    # Menambahkan kemiripan berdasarkan kategori
    category_scores = {}
    if place_name:
        category_of_searched_place = searched_place['Category'] if searched_place else None
        if category_of_searched_place:
            for idx, row in tourism_with_id.iterrows():
                if row['Category'] == category_of_searched_place:
                    category_scores[row['Place_Id']] = 1  # Bobot tinggi untuk tempat dengan kategori yang sama

    # Hybrid Scoring (gabungkan hasil content, category, dan collaborative)
    hybrid_scores = {}
    all_place_ids = set(collaborative_scores.keys()).union(content_scores.keys(), category_scores.keys())
    
    for place_id in all_place_ids:
        collaborative_score = collaborative_scores.get(place_id, 0)
        content_score = content_scores.get(place_id, 0)
        category_score = category_scores.get(place_id, 0)
        
        # Bobot lebih tinggi pada content-based filtering
        hybrid_scores[place_id] = 0.4 * content_score + 0.4 * category_score + 0.2 * collaborative_score 

    # Urutkan berdasarkan hybrid score
    sorted_place_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)
    recommendations = tourism_with_id[tourism_with_id['Place_Id'].isin(sorted_place_ids)].copy()
    recommendations['Hybrid_Score'] = recommendations['Place_Id'].apply(lambda x: hybrid_scores[x])
    
    # Filter tempat yang sama dengan yang dicari (hindari tempat yang sama muncul)
    if place_name:
        recommendations = recommendations[recommendations['Place_Name'] != place_name]

    # Urutkan kembali hasil rekomendasi berdasarkan hybrid score dan ambil 10 teratas
    recommendations = recommendations.sort_values(by='Hybrid_Score', ascending=False).head(10)

    if return_searched:
        return searched_place, recommendations.to_dict(orient='records')

    return recommendations.to_dict(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
