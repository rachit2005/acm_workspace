import kagglehub

# Download latest version
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")

print("Path to dataset files:", path)

'''
Cases include a representative collection of all important diagnostic categories in the realm of 
pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), 
basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses 
and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) 
and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
'''

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        pass