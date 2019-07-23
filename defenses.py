"""Contains defense functions for adversarial image recognition. All inputs are of shape (n, c, w, h)"""

from scipy.misc import imresize
from scipy import ndimage
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import torch

from torchvision.transforms import ToPILImage, ToTensor
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
from PIL import Image

from scipy.optimize import minimize

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

def median_filter_torch(size=3):
    def median_filter_sub(imgs):
        transformed = ndimage.filters.median_filter(np.rollaxis(imgs.cpu().numpy(), 1, 4), size)
        return torch.from_numpy(np.rollaxis(transformed, 3, 1)).cuda()
    return median_filter_sub


def _resize(img, new_width):
    """Input must be 3 x w x h"""
    resized = imresize(np.transpose(img, (1, 2, 0)), (new_width, new_width, 3))/255.0
   
    return np.transpose(resized, (2, 0, 1))

def _crop(img, seed = None, ratio = .6):
    """Input must be 3 x w x h"""
    c, h, w = img.shape
    np.random.seed(seed)
    x_start = np.random.randint(w*(1-ratio)) if ratio < 1 else 0
    x_end = x_start + ratio*w
    y_start = np.random.randint(h*(1-ratio)) if ratio < 1 else 0
    y_end = y_start + ratio*h
    return img[:, int(y_start):int(y_end), int(x_start):int(x_end)]
    
def crop_and_resize(ratio = .65):
    def crop_and_resize_sub(img, seed = None):
        """Input must be (n x 3 x w x h)"""
        n, c, w, h = img.shape
        transformed = img.copy()
        for i in range(n):
            cropped = _crop(transformed[i], seed, ratio)
            resized = _resize(cropped, w)
            transformed[i] = resized.astype(img.dtype)
        return transformed
    return crop_and_resize_sub

def crop_and_resize_torch(ratio = .65):
    def crop_and_resize_sub(img, seed = None):
        """Input must be (n x 3 x w x h)"""
        n, w = img.size(0), img.size(2)
        transformed = img.clone()
        for i in range(n):
            cropped = _crop(transformed[i].cpu().numpy(), seed, ratio)
            resized = _resize(cropped, w)
            transformed[i] = torch.from_numpy(resized).cuda()
        return transformed
    return crop_and_resize_sub
    
def identity(x):
    return x
    

def _tv(x, p):
    f = np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1).sum()
    f += np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0).sum()
    return f

def _tv_dx(x, p):
    if p == 1:
        x_diff0 = np.sign(x[1:, :] - x[:-1, :])
        x_diff1 = np.sign(x[:, 1:] - x[:, :-1])
    elif p > 1:
        x_diff0_norm = np.power(np.linalg.norm(x[1:, :] - x[:-1, :], p, axis=1), p - 1)
        x_diff1_norm = np.power(np.linalg.norm(x[:, 1:] - x[:, :-1], p, axis=0), p - 1)
        x_diff0_norm[x_diff0_norm < 1e-3] = 1e-3
        x_diff1_norm[x_diff1_norm < 1e-3] = 1e-3
        x_diff0_norm = np.repeat(x_diff0_norm[:, np.newaxis], x.shape[1], axis=1)
        x_diff1_norm = np.repeat(x_diff1_norm[np.newaxis, :], x.shape[0], axis=0)
        x_diff0 = p * np.power(x[1:, :] - x[:-1, :], p - 1) / x_diff0_norm
        x_diff1 = p * np.power(x[:, 1:] - x[:, :-1], p - 1) / x_diff1_norm
    df = np.zeros(x.shape)
    df[:-1, :] = -x_diff0
    df[1:, :] += x_diff0
    df[:, :-1] -= x_diff1
    df[:, 1:] += x_diff1
    return df

def _tv_l2(x, y, w, lam, p):
    f = 0.5 * np.power(x - y.flatten(), 2).dot(w.flatten())
    x = np.reshape(x, y.shape)
    return f + lam * _tv(x, p)

def _tv_l2_dx(x, y, w, lam, p):
    x = np.reshape(x, y.shape)
    df = (x - y) * w
    return df.flatten() + lam * _tv_dx(x, p).flatten()

def _minimize_tv(img, w, lam=0.1, p=2, solver='L-BFGS-B', maxiter=100, verbose=False):
    x_opt = np.copy(img)
    for i in range(img.shape[2]):
        options = {'disp': verbose, 'maxiter': maxiter}
        res = minimize(
            _tv_l2, x_opt[:, :, i], (img[:, :, i], w[:, :, i], lam, p),
            method=solver, jac=_tv_l2_dx, options=options).x
        x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
    return x_opt


def tv_torch_old(lam = .03):
    def tv_wrapper_sub(imgs):
        "Input image must be n x 3 x 224 x 224"
        n, c, w, h = imgs.shape
        transformed = imgs.clone()
        for i in range(n):
            img = np.transpose(transformed[i].cpu().numpy(), (1, 2, 0))
            next_transformed = _minimize_tv(img, w=np.ones_like(img), lam=lam)
            transformed[i] = torch.from_numpy(np.transpose(next_transformed, (2, 0, 1))).cuda()
        return transformed
    return tv_wrapper_sub



def tv_wrapper(lam = .03):
    def tv_wrapper_sub(imgs):
        "Input image must be n x 3 x 224 x 224"
        n, c, w, h = imgs.shape
        transformed = imgs.copy()
        for i in range(n):
            img = np.transpose(imgs[i].copy(), (1, 2, 0))
            next_transformed = _minimize_tv(img, w=np.ones_like(img), lam=lam)
            transformed[i] = np.transpose(next_transformed, (2, 0, 1))
        return transformed
    return tv_wrapper_sub

def tv_torch(lam=0.1):
    def tv_torch_sub(imgs):
        n, c, w, h = imgs.shape
        transformed = np.rollaxis(imgs.clone().cpu().numpy(), 1, 4)
        for i in range(n):
            transformed[i] = denoise_tv_chambolle(transformed[i], weight=lam, multichannel=True)
        transformed = torch.from_numpy(np.rollaxis(transformed, 3, 1)).cuda()
        return transformed
    return tv_torch_sub
