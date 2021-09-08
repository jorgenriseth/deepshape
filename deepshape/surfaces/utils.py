import torch

def torch_square_grid(k=64):
    X, Y = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X = torch.stack((Y, X), dim=-1)
    return X