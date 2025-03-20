import torch

class HashedTensor():
    def __init__(self, tensor):
        self.tensor = tensor.clone().detach().cpu()

    def __hash__(self):
        return hash(self.tensor.numpy().tobytes())

    def __eq__(self, other):
        return torch.all(self.tensor == other.tensor)