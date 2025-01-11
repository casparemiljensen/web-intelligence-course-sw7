import torch

# Define the function
def f(x1, x2):
    return torch.log(x1**2 + 2 * x2**2 - 2 * x1 * x2 - 2 * x2 + 2)

# Initialize x1 and x2 as torch tensors with requires_grad=True to enable gradient tracking
x1 = torch.tensor(-5.0, requires_grad=True)
x2 = torch.tensor(-5.0, requires_grad=True)



# Set the learning rate
learning_rate = 0.1

for step in range(2):
    # Calculate the gradient

    # Compute the function value
    y = f(x1, x2)

    # Perform backpropagation to compute gradients
    y.backward()

    # In PyTorch, we temporarily disable gradient tracking during the update step using torch.no_grad()
    # because we don’t want to record gradient computations during this part.

    # Update x1 and x2 using gradient descent
    with torch.no_grad():  # Temporarily disable gradient tracking for the update step
        x1 -= learning_rate * x1.grad
        x2 -= learning_rate * x2.grad

    # Clear the gradients for the next iteration
    x1.grad.zero_()
    x2.grad.zero_()

# Print the optimized values of x1 and x2
print(f"Optimized x1: {x1.item()}")
print(f"Optimized x2: {x2.item()}")