import torch

def is_cuda_available():
    """
    Checks if CUDA is available for PyTorch and prints the device name if available.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print("CUDA version:", torch.version.cuda)
        print("Device name:", torch.cuda.get_device_name(0))
        return True
    else:
        print("CUDA is not available.")
        return False

# Example usage:
if __name__ == "__main__":
    is_cuda_available()


# Use
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
