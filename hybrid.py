import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae


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
    all_tourism = all_tourism.drop_duplicates('Place_Id').fillna(method='ffill')
    
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

def explain_content_based_recommendations(input_place, all_tourism, recommendations):
    """
    Provide an explanation for the recommendations based on input place and its category.

    Parameters:
        input_place (str): Name of the input place.
        all_tourism (pd.DataFrame): DataFrame containing tourism data.
        recommendations (pd.DataFrame): DataFrame with recommended places.

    Returns:
        explanation (str): Explanation of the recommendations.
    """
    # Get input place details
    input_place_details = all_tourism[all_tourism['Place_Name'] == input_place].iloc[0]
    input_category = input_place_details['Category']
    
    # Generate explanation
    explanation = f"Anda menginputkan '{input_place}' yang merupakan kategori '{input_category}'.\n"
    explanation += "Tempat yang direkomendasikan berdasarkan kesamaan kategori atau nama adalah:\n"
    
    for index, row in recommendations.iterrows():
        reason = "Kategori yang sama" if row['Category'] == input_category else "Kemiripan dengan nama tempat"
        explanation += f"- {row['Place_Name']} ({row['Category']}): {reason}\n"

    return explanation

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

model_cf, testset, metrics = collaborative_filtering_knn(all_tourism)
print(f"Collaborative Filtering Metrics - RMSE: {metrics['RMSE']:.4f}")
print(f"Collaborative Filtering Metrics - MAE: {metrics['MAE']:.4f}")

# **6. Top Recommendations**
def collaborative_filtering_top_recommendations(user_id, model_cf, all_tourism, top_n=10):
    all_places = all_tourism['Place_Id'].unique()
    predictions = [model_cf.predict(user_id, place_id).est for place_id in all_places]
    
    recommendations = pd.DataFrame({
        'Place_Id': all_places,
        'Predicted_Rating': predictions
    }).sort_values(by='Predicted_Rating', ascending=False).head(top_n)
    
    recommendations = recommendations.merge(
        all_tourism[['Place_Id', 'Place_Name', 'Category']].drop_duplicates(),
        on='Place_Id'
    )
    return recommendations

# **7. Recommendations per Category**
def collaborative_filtering_recommendations_per_category(user_id, model_cf, all_tourism, categories, top_n=5):
    recommendations_by_category = {}

    for category in categories:
        category_places = all_tourism[all_tourism['Category'] == category]['Place_Id'].unique()
        predictions = [model_cf.predict(user_id, place_id).est for place_id in category_places]
        
        recommendations = pd.DataFrame({
            'Place_Id': category_places,
            'Predicted_Rating': predictions
        }).sort_values(by='Predicted_Rating', ascending=False).head(top_n)
        
        recommendations = recommendations.merge(
            all_tourism[['Place_Id', 'Place_Name', 'Category']].drop_duplicates(),
            on='Place_Id'
        )
        recommendations_by_category[category] = recommendations

    return recommendations_by_category

# **8. Execution**
new_user_id = 9999  # Example user ID

# Top 10 Recommendations
top_recommendations = collaborative_filtering_top_recommendations(new_user_id, model_cf, all_tourism, top_n=10)
print("\nCollaborative Filtering Recommendations:")
print(top_recommendations)

# Recommendations per Category
categories = all_tourism['Category'].unique()
recommendations_per_category = collaborative_filtering_recommendations_per_category(new_user_id, model_cf, all_tourism, categories, top_n=5)

print("\nRecommendations per Category:")
for category, recommendations in recommendations_per_category.items():
    print(f"\nCategory: {category}")
    print(recommendations)

# **9. Content-Based Recommendations**
input_place = 'Pelabuhan Marina'  # Example input
content_recommendations = content_based_recommendations(input_place, top_n=10)
print(f"\nContent-Based Recommendations for '{input_place}':")
print(content_recommendations)

# **9. Content-Based Recommendations with Explanation**
input_place = 'Pelabuhan Marina'  # Example input
content_recommendations = content_based_recommendations(input_place, top_n=10)

if not content_recommendations.empty:
    explanation = explain_content_based_recommendations(input_place, all_tourism, content_recommendations)
    print(f"\nContent-Based Recommendations for '{input_place}':")
    print(content_recommendations)
    print("\nExplanation:")
    print(explanation)
else:
    print(f"\nNo recommendations found for '{input_place}'.")




