import torch

def get_device():
    """
    Get the appropriate device for PyTorch computations.
    Prioritizes MPS (M1 GPU) if available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def print_device_info(logger):
    """Print information about the available devices using the logger"""
    device = get_device()
    logger.info(f"Using device: {device}")