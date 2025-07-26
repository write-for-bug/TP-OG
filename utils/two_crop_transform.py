import torch
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.cat([self.transform(x)[None,:], self.transform(x)[None,:]],dim=0 )