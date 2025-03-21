import torch
import scipy.stats as stats

def get_device(device: torch.device = None) -> torch.device:
    if device is not None:
        return device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device)

def to_one_hot(action: int, action_space_size: int) -> torch.Tensor:
    '''Convert the given action to one-hot encoding.

    Args:
        action (int): The action to convert.
        action_space_size (int): The size of the action space.

    Returns:
        torch.Tensor: The one-hot encoded action.
    '''
    action_one_hot = torch.zeros(1, action_space_size)
    action_one_hot[0][action] = 1.0
    return action_one_hot

def init_truncnorm(min: float, max: float, mean: float, std: float) -> torch.distributions.Normal:
    a, b = (min - mean) / std, (max - mean) / std
    return stats.truncnorm(a, b, loc=mean, scale=std)