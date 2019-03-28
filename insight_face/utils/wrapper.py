import torch

def no_grad(func):

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            ret = func(*args, **kwargs)
        return ret

    return wrapper