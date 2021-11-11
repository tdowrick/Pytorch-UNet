import torch
import os
from unet.unet_model import UNet

def model_fn(model_dir):
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

