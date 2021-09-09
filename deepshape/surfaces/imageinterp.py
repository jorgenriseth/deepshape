import torch
import numpy as np
import torch.nn as nn
from .surfaces import Surface

class ImageInterpolator(nn.Module):
    def __init__(self, img, mode="bilinear", **kwargs):
        """ Assumes img of shape (C, H, W)"""
        super().__init__(**kwargs)
        if img.dim() == 2:
            self.img = img.view(1, 1, *img.shape).clone().detach()
            self.H, self.W = img.shape
            self.C = 1
        elif img.dim() == 3:
            self.img = img.view(1, *img.shape).clone().detach()
            self.C, self.H, self.W = img.shape
        else:
            raise ValueError(f"img should be of shape (C, H, W) or (H, W), got {img.shape}")

        self.mode = mode
    
    def forward(self, x):
        if x.dim() == 2:
            return self.eval_point_list(x)
        elif x.dim() == 3:
            return self.eval_grid(x)
        raise ValueError(f"Got x with shape {x.shape}, should be (N, 2) or (H, W, 2)")
    
    def eval_point_list(self, x):
        """ Assumes the input comes on the form (N, 2), i.e. a list of points  
        points (x, y) placed within the domain D = [0, 1]^2"""
        npoints = x.shape[0]
        H = int(np.sqrt(npoints))
        
         #  Map input from [0, 1] -> [-1, 1] (required by grid_sample)
        X = x.view(1, H, H , 2)
        X = X * 2. - 1.
        
        out = nn.functional.grid_sample(self.img, X, mode=self.mode,
                                        align_corners=True, padding_mode="border")
        # Reshape output (C, H, H) -> (H * H, C)
        return out.view(self.C, npoints).transpose(1, 0).squeeze()
    
    def eval_grid(self, x):
        """ Assumes the input comes on the form (H, W, 2), i.e. a grid of points  
        points (x, y) placed within the domain D = [0, 1]^2"""
        
        #  Map input from [0, 1] -> [-1, 1] (required by grid_sample)
        X = x.view(1, *x.shape)
        X = X * 2. - 1. 
        
        # Interpolate, and reshape output to input-form
        out = nn.functional.grid_sample(self.img, X, mode=self.mode,
                                        align_corners=True, padding_mode="border")

        return out.view(self.C, *x[..., 0].shape).permute(1, 2, 0).squeeze()


class SingleChannelImageSurface(Surface):
    def __init__(self, img, **kwargs):
        super().__init__((lambda x: x[..., 0], lambda x: x[..., 1], lambda x: self.img(x)))
        self.img = ImageInterpolator(img)
