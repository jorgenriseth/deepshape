from matplotlib.pyplot import summer
import torch
import pickle

def torch_square_grid(k=64):
    Y, X = torch.meshgrid((torch.linspace(0, 1, k), torch.linspace(0, 1, k)))
    X = torch.stack((X, Y), dim=-1)
    return X


def single_component_mse(inputs, targets, component : int):
    """ Stored here for now, will probably be moved elsewhere in the future"""
    return torch.sum((inputs[..., component] - targets[..., component])**2) / inputs[..., component].nelement()

# Need to create a new optimizer when given new parameters.
def optimizer_builder(optimizer, *args, **kwargs):
    return lambda network: optimizer(network.parameters(), *args, **kwargs)
    

def save_distance_matrix(filename, D, y=None):
    with open(filename, 'wb') as f:
        pickle.dump((D, None), f)

        
def load_distance_matrix(filename):
    with open('./newfile.pickle', 'rb') as f:
        return pickle.load(f)


def symmetric_part(matrix):
    return 0.5 * (matrix + matrix.transpose())


def antisymmetric_part(matrix):
    return 0.5 * (matrix - matrix.transpose())