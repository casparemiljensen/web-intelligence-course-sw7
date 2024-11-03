import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import gzip


# def setup_dataset():
#     # Create a dataset
#     data = {
#         'Product ID': ['M1', 'M2', 'M3', 'M4'],
#         'Title': ['The Shawshank Redemption', 'The Dark Knight Rises', 'A Night on Elm Street', 'Donnie Darko'],
#         'Genre': ['Drama, Crime', 'Action, Crime', 'Horror', 'Fantasy, Drama'],
#         'Price': [100, 200, 50, 75],
#         'Keywords': ['Freedom', 'Solving Crimes', '', 'Psychological']
#     }
#
#     df = pd.DataFrame(data)
#     return df

def preprocess_data(df, no_of_items):
    selected_columns = ['asin', 'category', 'description', 'price']  # Adjust as needed
    new_df = df[selected_columns].copy()
    # Select only the first 10000 rows
    new_df = new_df.iloc[:no_of_items]

    # Step 2.1: One-Hot Encode Category
    mlb = MultiLabelBinarizer()
    category_encoded = mlb.fit_transform(new_df['category'])
    # Convert encoded categories to a DataFrame and concatenate with original DataFrame
    category_encoded = pd.DataFrame(category_encoded, columns=mlb.classes_)

    # Debug: Check for duplicates after encoding
    if category_encoded.columns.duplicated().any():
        print("Duplicate columns after category encoding:", category_encoded.columns[category_encoded.columns.duplicated()])

    # Step 2.3: Convert Keywords to TF-IDF Vector
    tfidf = TfidfVectorizer()
    new_df['description'] = new_df['description'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    tfidf_matrix = tfidf.fit_transform(new_df['description']).toarray()

    # # Step 2.2: Normalize Price
    new_df['price'] = new_df['price'].apply(try_parse_float)
    scaler = MinMaxScaler()
    new_df['price'] = scaler.fit_transform(new_df[['price']])
    price = new_df['price']

    # Keep 'asin' as a regular integer column
    asin_df = pd.DataFrame(new_df['asin'].values, columns=['asin'])

    category_df = pd.DataFrame(category_encoded, columns=mlb.classes_).astype(pd.SparseDtype("string", fill_value=None))
    price_df = pd.DataFrame(price.values, columns=['price']).astype(pd.SparseDtype("float", fill_value=0))
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out()).astype(
        pd.SparseDtype("float", fill_value=0))

    # Concatenate into a sparse DataFrame
    item_profiles = pd.concat([asin_df, category_df, price_df, tfidf_df], axis=1)

    # Drop rows where any column has None (NaN)
    # new_df.dropna(inplace=True)

    return item_profiles


def try_parse_float(value):
    if isinstance(value, str) and value.startswith('$'):
        value = value.replace('$', '')  # Remove the dollar sign
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# def create_user_profile(df):
#     # User ratings: (user, item, rating)
#     ratings = [
#         ('UserX', 'M1', 1),
#         ('UserX', 'M2', 1),
#         ('UserX', 'M3', 5),
#         ('UserX', 'M4', 3)
#     ]
#
#     # Step 3.1: Create a User Profile by Aggregating Weighted Item Profiles
#     user_profile = np.zeros(item_profiles.shape[1])
#     for _, item_id, rating in ratings:
#         item_index = df.index[df['Product ID'] == item_id][0]  # Get item index
#         user_profile += item_profiles[item_index] * rating  # Weighted sum
#
#     # Normalize the user profile to avoid scaling issues
#     user_profile /= np.sum([rating for _, _, rating in ratings])
#
#     return user_profile

# def create_single_user_profile(reviewer_id, df_ratings, item_profiles):
#     user_ratings = df_ratings[df_ratings['reviewerID'] == reviewer_id]
#
#     user_ratings = user_ratings.groupby(['reviewerID', 'asin']).agg({'overall': 'mean'}).reset_index()
#
#     print("Final columns in item_profiles:", item_profiles.columns)
#     print("Final columns in user_ratings:", user_ratings.columns)
#
#     # Assuming 'asin' is the column in both DataFrames to match on
#     user_rated_items = item_profiles.merge(user_ratings[['asin']], on='asin', how='inner')
#
#     num_features = item_profiles.shape[1] - 1  # Exclude 'asin' column
#     user_profile = np.zeros(num_features)
#     # Initialize user profile
#
#     # Aggregate weighted profiles
#     total_rating = 0
#     for _, row in user_rated_items.iterrows():
#         rating = row['overall']
#         item_vector = row.drop(['asin', 'overall']).values.astype(float)  # Only feature columns
#         user_profile += item_vector * rating
#         total_rating += rating
#
#     # Normalize the user profile by the sum of ratings
#     if total_rating > 0:
#         user_profile /= total_rating
#     user_profiles.append(user_profile)
#     return user_profile

# Helper function to create user profiles in chunks
# def create_user_profiles_chunk(reviewer_ids_chunk, df_ratings, item_profiles):
#     chunk_user_profiles = []
#     print(f'Starting new thread for {len(reviewer_ids_chunk)}')
#     for reviewer_id in reviewer_ids_chunk:
#         user_ratings = df_ratings[df_ratings['reviewerID'] == reviewer_id]
#         user_ratings = user_ratings.groupby(['asin']).agg({'overall': 'mean'}).reset_index()
#
#         # print("Final columns in item_profiles:", item_profiles.columns)
#         # print("Final columns in user_ratings:", user_ratings.columns)
#         # Debug: Check and clean columns before merging
#         if 'asin' in item_profiles.columns.duplicated():
#             item_profiles = item_profiles.loc[:, ~item_profiles.columns.duplicated()]
#         if 'asin' in user_ratings.columns.duplicated():
#             user_ratings = user_ratings.loc[:, ~user_ratings.columns.duplicated()]
#
#         # Merge user ratings with item profiles
#         user_rated_items = item_profiles.merge(user_ratings, on='asin', how='inner')
#
#         num_features = item_profiles.shape[1] - 1  # Exclude 'asin' column
#         user_profile = np.zeros(num_features)
#
#         # Aggregate weighted profiles
#         total_rating = 0
#         for _, row in user_rated_items.iterrows():
#             rating = row['overall']
#             item_vector = row.drop(['asin', 'overall']).values.astype(float)
#             user_profile += item_vector * rating
#             total_rating += rating
#
#         # Normalize the user profile by the sum of ratings
#         if total_rating > 0:
#             user_profile /= total_rating
#
#         chunk_user_profiles.append(user_profile)
#     print('Stopping thread')
#     return chunk_user_profiles

# def create_user_profiles(df_ratings):
#     reviewer_ids = df_ratings['reviewerID'].unique()
#     user_profiles = []
#
#     # Use ThreadPoolExecutor to parallelize profile creation
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(create_single_user_profile, reviewer_id, df_ratings, item_profiles)
#                    for reviewer_id in reviewer_ids]
#
#         # Collect results as they complete
#         user_profiles = [future.result() for future in futures]
#
#     return user_profiles
#     reviewer_ids = df_ratings['reviewerID'].unique()


# THIS ONE
# Main function to create user profiles using processes with a simple counter
# def create_user_profiles(df_ratings, chunk_size=50):
#     reviewer_ids = df_ratings['reviewerID'].unique()
#     user_profiles = []
#
#     # Split reviewer_ids into chunks for chunk processing
#     reviewer_id_chunks = [reviewer_ids[i:i + chunk_size] for i in range(0, len(reviewer_ids), chunk_size)]
#
#     # Initialize a counter
#     counter = 0
#     total_chunks = len(reviewer_id_chunks)
#     print(f'Total number of chunks: {total_chunks} - Total number user profiles: {total_chunks*chunk_size}')
#
#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor() as executor:
#         # Create a list of futures
#         futures = [executor.submit(create_user_profiles_chunk, chunk, df_ratings, item_profiles) for chunk in
#                    reviewer_id_chunks]
#
#         # Collect results and update the counter
#         for future in as_completed(futures):
#             try:
#                 user_profiles.extend(future.result())  # Combine profiles from each chunk
#                 counter += 1
#                 print(f"Processed {counter}/{total_chunks} chunks", flush=True)
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#
#     return user_profiles


def create_user_profiles(df_ratings):
    user_profiles = []

    # Loop through reviewer IDs to create user profiles
    reviewer_ids = df_ratings['reviewerID'].unique()

    for reviewer_id in reviewer_ids:
        user_ratings = df_ratings[df_ratings['reviewerID'] == reviewer_id]
        user_ratings = user_ratings.groupby(['reviewerID', 'asin']).agg({'overall': 'mean'}).reset_index()

        # Retrieve items rated by the user based on 'asin' values
        asin_list = user_ratings['asin'].unique()
        user_rated_items = ITEM_PROFILES[ITEM_PROFILES['asin'].isin(asin_list)]

        # Calculate the user profile based on rated items
        num_features = ITEM_PROFILES.shape[1] - 1  # Exclude 'asin' column
        user_profile = np.zeros(num_features)
        total_rating = 0
        for _, row in user_rated_items.iterrows():
            rating = row['overall']
            item_vector = row.drop(['asin', 'overall']).values.astype(float)  # Only feature columns
            user_profile += item_vector * rating
            total_rating += rating

        # Normalize the user profile by the sum of ratings
        if total_rating > 0:
            user_profile /= total_rating
        user_profiles.append(user_profile)
    return user_profiles


def calculate_similarity_recommend_items(user_profile, df):
    # Calculate cosine similarity between user profile and each item profile
    similarity_scores = cosine_similarity([user_profile], ITEM_PROFILES).flatten()

    # Recommend top 2 items based on similarity
    top_indices = similarity_scores.argsort()[-2:][::-1]  # Get top indices (2 most similar items)
    recommended_items = df.iloc[top_indices]
    print("Recommended Items:")
    print(recommended_items[['Title', 'Genre', 'Price', 'Keywords']])


def remove_duplicate_columns(df):
    """
    Removes duplicate columns from a DataFrame, keeping only the first occurrence of each column label.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which to remove duplicate columns.

    Returns:
    - pd.DataFrame: A DataFrame with unique column names.
    """
    # Remove duplicated columns by keeping only the first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


if __name__ == "__main__":
    # df = setup_dataset()
    df_content = getDF('data/meta_Software.json.gz')
    df_rating_and_reviews = getDF('data/Software.json.gz')
    no_of_items = 100
    no_of_user_ratings = 300
    ITEM_PROFILES = preprocess_data(df_content, 10000)
    ITEM_PROFILES = remove_duplicate_columns(ITEM_PROFILES)

    subset_ratings = df_rating_and_reviews[:no_of_user_ratings]
    user_profiles = create_user_profiles(subset_ratings)
    # calculate_similarity_recommend_items(user_profile, df_content)
    print(user_profiles)

    # print("DF Content Start")
    # print("----------")
    # print(df_content)
    # print(df_content.columns)
    # print(df_content["title"])
    # print(df_content["feature"])
    # print("DF Content End")
    # print("----------")
    #
    # print("DF Rating Start")
    # print("----------")
    # df_rating_and_reviews = getDF('data/Software.json.gz')
    # print(df_rating_and_reviews)
    # print("DF Rating End")
    # print("----------")
