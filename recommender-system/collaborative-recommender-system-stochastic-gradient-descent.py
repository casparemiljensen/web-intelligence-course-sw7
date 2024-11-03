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
num_epochs = 50  # Number of passes through the data

# Initialize user and item matrices with small random values
A = np.random.rand(num_users, K) * 0.1
B = np.random.rand(K, num_items) * 0.1

# Stochastic Gradient Descent
for epoch in range(num_epochs):
    # Loop through each user-item pair with a rating
    for i in range(num_users):
        for j in range(num_items):
            if not np.isnan(R[i, j]):  # Only consider observed entries
                # Calculate prediction error for the single entry
                prediction = A[i, :] @ B[:, j]
                error = R[i, j] - prediction

                # Compute the gradients for A[i, :] and B[:, j]
                A_gradient = -2 * error * B[:, j] + lambda_reg * A[i, :]
                B_gradient = -2 * error * A[i, :] + lambda_reg * B[:, j]

                # Update A[i, :] and B[:, j] with SGD step
                A[i, :] -= learning_rate * A_gradient
                B[:, j] -= learning_rate * B_gradient

    # Compute and print the total error after each epoch
    if epoch % 10 == 0:
        # Compute the full error only for tracking progress (optional)
        total_error = np.nansum((R - A @ B) ** 2)
        print(f"Epoch {epoch}: Error = {total_error}")

# Final predicted matrix
predicted_ratings = A @ B
print("Predicted Ratings:\n", predicted_ratings)
