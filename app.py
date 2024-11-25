import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae

# Load datasets
package = pd.read_csv('package_tourism.csv')
tourism = pd.read_csv('tourism_with_id.csv')
rating = pd.read_csv('tourism_rating.csv')
user = pd.read_csv('user.csv')

# **1. Data Preprocessing**
def preprocess_data(tourism, rating):
    # Drop unused columns
    tourism = tourism.drop(['Description', 'City', 'Price', 'Rating', 'Time_Minutes', 
                            'Coordinate', 'Lat', 'Long', 'Unnamed: 11', 'Unnamed: 12'], axis=1)
    
    # Merge rating with tourism data for enrichment
    all_tourism = pd.merge(rating, tourism, on="Place_Id", how="left")
    
    # Drop duplicates and handle missing values
    all_tourism = all_tourism.drop_duplicates('Place_Id').dropna()
    
    return all_tourism

all_tourism = preprocess_data(tourism, rating)

# **2. Normalization**
def normalize_ratings(df):
    df['Place_Ratings'] = df['Place_Ratings'].astype(np.float32)
    min_rating = df['Place_Ratings'].min()
    max_rating = df['Place_Ratings'].max()
    df['Normalized_Rating'] = (df['Place_Ratings'] - min_rating) / (max_rating - min_rating)
    return df

all_tourism = normalize_ratings(all_tourism)

# **3. Exploratory Data Analysis**
def eda_visualization(df):
    # Distribusi Rating
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Place_Ratings', data=df, palette='gist_rainbow', order=df['Place_Ratings'].value_counts().index)
    plt.title("Distribusi Rating Tempat Wisata")
    plt.xlabel("Rating")
    plt.ylabel("Jumlah")
    plt.show()
    
    # Distribusi Kategori
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Category', data=df, palette='gist_rainbow', order=df['Category'].value_counts().index)
    plt.title("Distribusi Kategori Tempat Wisata")
    plt.xlabel("Kategori")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=45)
    plt.show()

eda_visualization(all_tourism)

# **4. Content-Based Filtering**
def content_based_recommendations(place_name, top_n=5):
    data = all_tourism[['Place_Id', 'Place_Name', 'Category']]
    tf = TfidfVectorizer()
    tfidf_matrix = tf.fit_transform(data['Place_Name'])
    
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=data['Place_Name'], columns=data['Place_Name'])
    
    # Get recommendations
    if place_name not in cosine_sim_df.columns:
        print(f"Error: Place '{place_name}' not found.")
        return pd.DataFrame(columns=['Place_Id', 'Place_Name', 'Category'])
    
    similar_places = cosine_sim_df[place_name].sort_values(ascending=False)[1:top_n+1]
    recommendations = data[data['Place_Name'].isin(similar_places.index)]
    return recommendations

# **5. Collaborative Filtering with KNNBasic**
def collaborative_filtering_knn(all_tourism):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(all_tourism[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
    
    # Train-test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Train KNN model
    algo = KNNBasic(k=20, sim_options={'name': 'cosine', 'user_based': True})
    algo.fit(trainset)
    
    # Evaluate model
    predictions = algo.test(testset)
    rmse_value = rmse(predictions)
    mae_value = mae(predictions)
    
    return algo, testset, {"RMSE": rmse_value, "MAE": mae_value}

# **6. Testing Collaborative Filtering for New Users**
def test_new_user_recommendation(model, all_tourism, user_id, top_n=10):
    all_places = all_tourism['Place_Id'].unique()
    places_visited = all_tourism[all_tourism['User_Id'] == user_id]['Place_Id'].unique()
    places_not_visited = [pid for pid in all_places if pid not in places_visited]
    
    # Predict ratings for all places not visited
    predictions = [model.predict(user_id, place_id).est for place_id in places_not_visited]
    recommendations = pd.DataFrame({
        'Place_Id': places_not_visited,
        'Predicted_Rating': predictions
    }).sort_values(by='Predicted_Rating', ascending=False).head(top_n)
    
    recommendations_df = recommendations.merge(
        all_tourism[['Place_Id', 'Place_Name', 'Category']].drop_duplicates(),
        on='Place_Id'
    )
    return recommendations_df

# **7. Testing Collaborative Filtering per Category**
def test_recommendation_per_category(model, all_tourism, user_id, categories, top_n=5):
    recommendations_by_category = {}
    
    for category in categories:
        category_places = all_tourism[all_tourism['Category'] == category]['Place_Id'].unique()
        predictions = [
            model.predict(user_id, place_id).est for place_id in category_places
        ]
        recommendations = pd.DataFrame({
            'Place_Id': category_places,
            'Predicted_Rating': predictions
        }).sort_values(by='Predicted_Rating', ascending=False).head(top_n)
        recommendations_df = recommendations.merge(
            all_tourism[['Place_Id', 'Place_Name', 'Category']].drop_duplicates(),
            on='Place_Id'
        )
        recommendations_by_category[category] = recommendations_df
    return recommendations_by_category

# **8. Execution**
# Train Collaborative Filtering Model
model_knn, testset, metrics = collaborative_filtering_knn(all_tourism)
print(f"Collaborative Filtering Metrics - RMSE: {metrics['RMSE']:.4f}")
print(f"Collaborative Filtering Metrics - MAE: {metrics['MAE']:.4f}")

# Testing for a New User
new_user_id = 9999  # Example user ID
top_recommendations = test_new_user_recommendation(model_knn, all_tourism, new_user_id, top_n=10)
print("\nTop 10 Recommendations for a New User:")
print(top_recommendations)

# Recommendations per Category
categories = all_tourism['Category'].unique()
recommendations_per_category = test_recommendation_per_category(model_knn, all_tourism, new_user_id, categories, top_n=5)

print("\nRecommendations per Category:")
for category, recommendations in recommendations_per_category.items():
    print(f"\nCategory: {category}")
    print(recommendations)
