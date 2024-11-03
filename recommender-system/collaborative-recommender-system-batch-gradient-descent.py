import numpy as np

# Define the rating matrix R with NaN for missing values
R = np.array([
    [5, 4, np.nan, np.nan],
    [np.nan, 5, 3, np.nan],
    [4, np.nan, np.nan, 1]
])

# Parameters
num_users, num_items = R.shape
K = 2  # Number of latent factors
learning_rate = 0.01
lambda_reg = 0.01  # Regularization strength for weight decay
num_iterations = 10000

# Initialize user and item matrices with small random values
A = np.random.rand(num_users, K) * 0.1
B = np.random.rand(K, num_items) * 0.1

# Create a mask to ignore NaN values in R during calculations
mask = ~np.isnan(R)  # True where ratings exist


# Helper function to compute the error (considering only observed entries)
def compute_error(R, A, B, mask):
    prediction = A @ B
    # Extract only the observed elements for error calculation
    observed_ratings = R[mask]
    predicted_ratings = prediction[mask]
    error = observed_ratings - predicted_ratings
    return np.sum(error ** 2)


# Gradient descent with weight decay (regularization)
for iteration in range(num_iterations):
    # Compute predictions and error only for observed ratings
    prediction = A @ B
    error = np.zeros_like(R)
    error[mask] = R[mask] - prediction[mask]  # Only calculate error for observed entries

    # Compute the gradients with regularization term
    A_gradient = -2 * (error @ B.T) + lambda_reg * A
    B_gradient = -2 * (A.T @ error) + lambda_reg * B

    # Update A and B
    A -= learning_rate * A_gradient
    B -= learning_rate * B_gradient

    # Compute and print the error for tracking convergence
    if iteration % 100 == 0:
        print(f"Iteration {iteration}: Error = {compute_error(R, A, B, mask)}")

# Final predicted matrix
predicted_ratings = A @ B
print("Predicted Ratings:\n", predicted_ratings)
