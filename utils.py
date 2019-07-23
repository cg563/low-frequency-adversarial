import torch
import numpy as np
import torchvision.transforms as trans
import math
from scipy.fftpack import dct, idct


IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose([
    trans.Scale(256),
    trans.CenterCrop(224),
    trans.ToTensor()])

CIFAR_SIZE = 32
# CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
# CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR_MEAN = [0.0, 0.0, 0.0]
CIFAR_STD = [1.0, 1.0, 1.0]
CIFAR_TRANSFORM = trans.Compose([
    trans.ToTensor()])

MNIST_SIZE = 28
MNIST_MEAN = [0.5]
MNIST_STD = [1.0]
MNIST_TRANSFORM = trans.Compose([
    trans.ToTensor()])

# reverses the normalization transformation
def invert_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    imgs_trans = imgs.clone()
    if len(imgs.size()) == 3:
        for i in range(imgs.size(0)):
            imgs_trans[i, :, :] = imgs_trans[i, :, :] * std[i] + mean[i]
    else:
        for i in range(imgs.size(1)):
            imgs_trans[:, i, :, :] = imgs_trans[:, i, :, :] * std[i] + mean[i]
    return imgs_trans

# applies the normalization transformations
def apply_normalization(imgs, dataset):
    if dataset == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    elif dataset == 'cifar':
        mean = CIFAR_MEAN
        std = CIFAR_STD
    elif dataset == 'mnist':
        mean = MNIST_MEAN
        std = MNIST_STD
    imgs_tensor = imgs.clone()
    if dataset == 'mnist':
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor

def get_preds(model, inputs, dataset_name, correct_class=None, batch_size=50, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size):upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs

def sample_gaussian_torch(image_size, dct_ratio=1.0):
    x = torch.zeros(image_size)
    fill_size = int(image_size[-1] * dct_ratio)
    x[:, :, :fill_size, :fill_size] = torch.randn(x.size(0), x.size(1), fill_size, fill_size)
    if dct_ratio < 1.0:
        x = torch.from_numpy(idct(idct(x.numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
    return x

def sample_gaussian_tf(image_size, dct_ratio=1.0):
    if dct_ratio < 1.0:
        dct_shape = (image_size[0], int(image_size[1] * dct_ratio), int(image_size[2] * dct_ratio), image_size[3])
        padding = [[0, 0], [0, image_size[1] - dct_shape[1]], [0, image_size[2] - dct_shape[2]], [0, 0]]
        x = tf.random_normal(dct_shape)
        x = tf.pad(x, padding)
        x = tf.transpose(tf.spectral.idct(tf.transpose(tf.spectral.idct(tf.transpose(x, [0, 3, 1, 2]), norm='ortho'), [0, 1, 3, 2]), norm='ortho'), [0, 3, 2, 1])
    else:
        x = tf.random_normal(image_size)
    return x
