import json
import gzip
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Function to parse gzipped JSON files
def parse_gz_file(path):
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

# Function to load the dataset into a DataFrame
def load_reviews(path, num_rows=None):
    data = parse_gz_file(path)
    df = pd.DataFrame.from_dict(data, orient='columns')
    if num_rows:
        return df.head(num_rows)
    return df

# Load the reviews dataset
reviews_path = 'data/Software.json.gz'  # Replace with the actual path to your file
reviews_df = load_reviews(reviews_path, num_rows=10000)  # Load the first 10,000 rows

# Extract relevant fields: reviewerID, asin, and overall rating
reviews_df = reviews_df[['reviewerID', 'asin', 'overall']]
reviews_df.dropna(inplace=True)

# Map reviewerID and asin to numeric indices for the rating matrix
user_map = {user: i for i, user in enumerate(reviews_df['reviewerID'].unique())}
item_map = {item: i for i, item in enumerate(reviews_df['asin'].unique())}

reviews_df['user_id'] = reviews_df['reviewerID'].map(user_map)
reviews_df['item_id'] = reviews_df['asin'].map(item_map)

num_users = len(user_map)
num_items = len(item_map)

# Create the rating matrix
R = np.zeros((num_users, num_items))
for _, row in reviews_df.iterrows():
    R[int(row['user_id']), int(row['item_id'])] = row['overall']

# Split data into training and testing sets (80% train, 20% test)
train_indices, test_indices = train_test_split(np.array(reviews_df.index), test_size=0.2, random_state=42)
train_mask = np.zeros_like(R, dtype=bool)
test_mask = np.zeros_like(R, dtype=bool)

for idx in train_indices:
    user, item = int(reviews_df.loc[idx, 'user_id']), int(reviews_df.loc[idx, 'item_id'])
    train_mask[user, item] = True

for idx in test_indices:
    user, item = int(reviews_df.loc[idx, 'user_id']), int(reviews_df.loc[idx, 'item_id'])
    test_mask[user, item] = True

R_train = np.where(train_mask, R, 0)
R_test = np.where(test_mask, R, 0)

# Parameters
K = 10  # Latent factors
learning_rate = 0.001  # Reduced learning rate for stability
lambda_reg = 0.1  # Regularization term
num_epochs = 50

# Initialize user and item matrices with smaller random values
A = np.random.rand(num_users, K) * 0.01
B = np.random.rand(K, num_items) * 0.01

# Helper function to calculate RMSE and MAE
def calculate_metrics(R_true, R_pred, mask):
    true_ratings = R_true[mask]
    pred_ratings = R_pred[mask]
    mae = mean_absolute_error(true_ratings, pred_ratings)
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    return mae, rmse

# Training with gradient clipping and NaN handling
for epoch in range(num_epochs):
    for i in range(num_users):
        for j in range(num_items):
            if train_mask[i, j]:
                # Compute error
                prediction = A[i, :] @ B[:, j]
                error = R_train[i, j] - prediction

                # Update gradients with clipping
                A_gradient = np.clip(-2 * error * B[:, j] + lambda_reg * A[i, :], -1e3, 1e3)
                B_gradient = np.clip(-2 * error * A[i, :] + lambda_reg * B[:, j], -1e3, 1e3)

                A[i, :] -= learning_rate * A_gradient
                B[:, j] -= learning_rate * B_gradient

    # Check for NaN or overflow
    if np.isnan(A).any() or np.isnan(B).any():
        print("NaN detected. Reinitializing matrices.")
        A = np.random.rand(num_users, K) * 0.01
        B = np.random.rand(K, num_items) * 0.01

    # Evaluate the model every 10 epochs
    if epoch % 10 == 0:
        predictions = A @ B
        train_mae, train_rmse = calculate_metrics(R_train, predictions, train_mask)
        test_mae, test_rmse = calculate_metrics(R_test, predictions, test_mask)
        print(f"Epoch {epoch} - Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# Final evaluation
predictions = A @ B
train_mae, train_rmse = calculate_metrics(R_train, predictions, train_mask)
test_mae, test_rmse = calculate_metrics(R_test, predictions, test_mask)
print(f"Final Results - Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
