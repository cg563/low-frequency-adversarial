"""Contains defense functions for adversarial image recognition. All inputs are of shape (n, c, w, h)"""

from scipy import ndimage
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image

def identity(x):
    return x

def bit_reduction(bits=3):
    """Generates bit reduction function"""
    bits_removed = 8-bits
    return lambda im: ((2**bits_removed)/255)*(im*255//(2**bits_removed))

def bit_reduction_torch(bits=3):
    """Generates bit reduction function"""
    bits_removed = 8-bits
    return lambda im: ((2**bits_removed)/255)*torch.floor(im*255/(2**bits_removed))

def jpeg_numpy(quality=75):
    def jpeg_sub(imgs):
        """Input image of shape (n, 3, 224, 224)"""
        n, c, w, h = imgs.shape
        transformed = imgs.copy()
        for i in range(n):
            im = torch.Tensor(imgs[i])
            im = ToPILImage()(im)
            savepath = BytesIO()
            im.save(savepath, 'JPEG', quality=quality)
            im = Image.open(savepath)
            im = ToTensor()(im).numpy()
            transformed[i] = im
        return transformed
    return jpeg_sub

def jpeg_torch(quality=75):
    def jpeg_sub(imgs):
        """Input image of shape (n, 3, 224, 224)"""
        n, c, w, h = imgs.shape
        transformed = imgs.clone().cpu()
        for i in range(n):
            im = ToPILImage()(transformed[i])
            savepath = BytesIO()
            im.save(savepath, 'JPEG', quality=quality)
            im = Image.open(savepath)
            im = ToTensor()(im)
            transformed[i] = im
        return transformed.cuda()
    return jpeg_sub
