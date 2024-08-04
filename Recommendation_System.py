import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
    'item_id': [101, 102, 103, 101, 104, 102, 103, 101, 103, 104],
    'rating': [5, 4, 3, 5, 2, 4, 5, 3, 4, 1]
}

df = pd.DataFrame(data)

def create_pivot_table(df):
    return df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

def compute_user_similarity(pivot_table):
    return cosine_similarity(pivot_table)

def get_recommendations(user_id, pivot_table, user_similarity, num_recommendations=3):
    if user_id not in pivot_table.index:
        print(f"User {user_id} not found in the dataset.")
        return []
    
    user_index = pivot_table.index.get_loc(user_id)
    similar_users = np.argsort(-user_similarity[user_index])
    similar_users = similar_users[similar_users != user_index]

    user_ratings = pivot_table.loc[user_id]
    recommendations = {}
    
    for sim_user in similar_users:
        sim_user_id = pivot_table.index[sim_user]
        sim_user_ratings = pivot_table.iloc[sim_user]
        
        for item_id in sim_user_ratings.index:
            if user_ratings[item_id] == 0 and sim_user_ratings[item_id] > 0:
                if item_id not in recommendations:
                    recommendations[item_id] = sim_user_ratings[item_id]
                else:
                    recommendations[item_id] += sim_user_ratings[item_id]
    
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in recommendations[:num_recommendations]]

def main():
    pivot_table = create_pivot_table(df)
    user_similarity = compute_user_similarity(pivot_table)
    
    print("\nOptions:")
    print("1. Get recommendations")
    print("2. Exit")
    
    while True:
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            try:
                user_id = int(input("Enter user ID to get recommendations for: ").strip())
                num_recommendations = int(input("Enter number of recommendations: ").strip())
                
                recommended_items = get_recommendations(user_id, pivot_table, user_similarity, num_recommendations)
                if recommended_items:
                    print(f"Recommended items for user {user_id}: {recommended_items}")
                else:
                    print(f"No recommendations available for user {user_id}.")
            except ValueError:
                print("Invalid input. Please enter integers for user ID and number of recommendations.")
        
        elif choice == '2':
            break
        
        else:
            print("Invalid choice. Please select a valid option.")

if __name__ == "__main__":
    main()