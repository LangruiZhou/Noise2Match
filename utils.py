import torch
import numpy as np
import torch.nn.functional as F


# denormalize to [-100, 200]HU for display
def trans_to_display_window(img, MIN_B=-1024, MAX_B=3072, cut_min=-100, cut_max=200):
    img = img.squeeze().detach().cpu().numpy()
    img = img * (MAX_B - MIN_B) + MIN_B
    img[img < cut_min] = cut_min
    img[img > cut_max] = cut_max
    img = 255 * (img - cut_min) / (cut_max - cut_min)
    return img




