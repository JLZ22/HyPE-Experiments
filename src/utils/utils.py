import torch

def get_device(device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device)