import torch
import numpy as np
from .utils import numpy_nans

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
    iterations = optimizer.defaults["max_eval"]
    print(iterations)
    
    # Evaluate initial error
    logger.start()
    error = numpy_nans(iterations+1)
    error[0] = loss(network)
    it = [0]
    logger.log(it=0, value=loss.get_last())

    def closure():
        it[0] += 1

        # Set gradient buffers to zero.
        optimizer.zero_grad()
        network.project()

        
        # Compute loss, and perform a backward pass and gradient step
        l = loss(network)
        l.backward()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(l)

        # Log error
        try: 
            error[it[0]] = l.item()

        except IndexError:  # Increase length of error-array if necessary... 
            np.pad(error, (0, iterations), constant_values=np.nan)
            error[it[0]] = l.item()  # and retry.
            
        logger.log(it=it[0], value=loss.get_last())
        
        return l

    optimizer.step(closure)
    network.project()
    logger.log(it=it[0], value=loss.get_last())
    logger.stop()

    return error[~np.isnan(error)]