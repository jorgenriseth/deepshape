import time
import torch
import torch.nn as nn
import numpy as np

# from .layers import FourierLayer
from ..common import central_differences, numpy_nans
from .layers import DeepShapeLayer

class ReparametrizationNetwork(nn.Module):
    def __init__(self, layerlist):
        super().__init__()
        self.layerlist = layerlist
        for layer in layerlist:
            assert isinstance(layer, DeepShapeLayer), "Layers must inherit DeepShapeLayer"
        
    def forward(self, x):
        for layer in self.layerlist:
            x = layer(x)
        return x
    
    def derivative(self, x, h=1e-4):
        if h is not None:
            return central_differences(self, x, h)
        
        dc = 1.
        for layer in self.layerlist:
            dc *= layer.derivative(x)
            x = layer(x)
            
        return dc

    def project(self, **kwargs):
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, DeepShapeLayer):
                    module.project(**kwargs)


def reparametrize(q, r, network, loss, optimizer, iterations, logger, scheduler=None):
    if isinstance(optimizer, torch.optim.LBFGS):
        return reparametrize_lbfgs(q, r, network, loss, optimizer, logger, scheduler)
        
    # Evaluate initial error
    logger.start()
    error = numpy_nans(iterations+1)
    error[0] = loss(network)
    
    for i in range(iterations):
        # Zero gradient buffers
        optimizer.zero_grad()
        
        # Compute current loss and gradients
        l = loss(network)
        l.backward()
        
        # Update optimizer if using scheduler
        if scheduler is not None:
            scheduler.step(l)
            
        # Update parameters
        optimizer.step()
        network.project()

        error[i+1] = loss.get_last()
        logger.log(it=i, value=error[i+1])
    
    logger.stop()
    return error


def reparametrize_lbfgs(q, r, network, loss, optimizer, logger, scheduler=None):
    # Get max iterations from optimizer
    iterations = max(optimizer.defaults["max_iter"], optimizer.defaults["max_eval"])
    print("Iterations", iterations)
    
    # Evaluate initial error
    logger.start()
    error = numpy_nans(iterations+1)
    error[0] = loss(network)
    it = [0]
    
    def closure():
        # Set gradient buffers to zero.
        optimizer.zero_grad()
        network.project()

        
        # Compute loss, and perform a backward pass and gradient step
        l = loss(network)
        l.backward()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(l)

        # Save Loss Function
        it[0] += 1

        # Log error
        try: 
            error[it[0]] = l.item()

        except IndexError:  # Increase length of error-array if necessary... 
            np.pad(error, (0, iterations), constant_values=np.nan)
            error[it[0]] = l.item()  # and retry.
            
        logger.log(it=it[0], value=loss.get_last())

        
        return l

    optimizer.step(closure)
    logger.stop()

    return error[~np.isnan(error)]
