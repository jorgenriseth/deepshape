import torch
import torch.nn as nn

class BumpLayer(nn.Module):
    def forward(self, x):
        return nn.ReLU(x) * nn.ReLU(1-x)