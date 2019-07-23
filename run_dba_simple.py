import os
import numpy as np
import random
import argparse
import utils
import defenses
import math
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as trans
from boundary_attack_simple import boundary_attack

parser = argparse.ArgumentParser(description='Runs decision-based attack in either RGB space or LF-DCT space')
parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
parser.add_argument('--result_dir', type=str, default='save', help='directory for saving results')
parser.add_argument('--model', type=str, default='resnet50', help='type of base model to use')
parser.add_argument('--num_runs', type=int, default=1000, help='number of repeated runs')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for parallel runs')
parser.add_argument('--num_steps', type=int, default=30000, help='maximum number of iterations')
parser.add_argument('--log_every', type=int, default=1, help='log every n iterations')
parser.add_argument('--defense', type=str, default='none', help='type of transformation defense')
parser.add_argument('--perturb_mode', type=str, default='gaussian', help='type of spherical perturbation sample (gaussian/dct)')
parser.add_argument('--spherical_step', type=float, default=0.01, help='spherical step size')
parser.add_argument('--source_step', type=float, default=0.01, help='source step size')
parser.add_argument('--repeat_images', type=int, default=1, help='number of repetitions for successive halving when using Hyperband')
parser.add_argument('--halve_every', type=int, default=250, help='number of iterations before successive halving when using Hyperband')
parser.add_argument('--dct_ratio', type=float, default=0.03125, help='ratio of nonzero frequencies for dct')
parser.add_argument('--blended_noise', action='store_true', help='interpolate between initial noise and target image to form starting point')
parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
args = parser.parse_args()

random.seed(123)
if args.model == 'resnet101':
    model = models.resnet101(pretrained=True).cuda()
    model.eval()
elif args.model == 'resnet50':
    model = models.resnet50(pretrained=True).cuda()
    model.eval()
else:
    raise NotImplementedError

if args.defense == 'none':
    trans = lambda x: x
elif args.defense == 'jpeg':
    trans = lambda x: defenses.jpeg_torch()(x)
elif args.defense == 'bit':
    trans = lambda x: defenses.bit_reduction_torch()(x)
elif args.defense == 'tv':
    trans = lambda x: defenses.tv_torch()(x)
dct_mode = (args.perturb_mode == 'dct')

# load previously sampled set of images
batchfile = 'save/batch_%s_%s_%d.pth' % (args.model, args.defense, args.num_runs)
if os.path.isfile(batchfile):
    checkpoint = torch.load(batchfile)
    images = checkpoint['images']
    labels = checkpoint['labels']
else:
    # sample a new set of images that are correctly classified
    testset = datasets.ImageFolder(args.data_root + '/val', utils.IMAGENET_TRANSFORM)
    images = torch.zeros(args.num_runs, 3, 224, 224)
    labels = torch.zeros(args.num_runs).long()
    preds = labels + 1
    while preds.ne(labels).sum() > 0:
        idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
        for i in list(idx):
            images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
        preds[idx], _ = utils.get_preds(model, trans(images[idx]), 'imagenet', batch_size=args.batch_size)
    torch.save({'images': images, 'labels': labels}, batchfile)
    
N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
# some statistics for analysis
all_perturbed = None
all_mse_stats = None
all_distance_stats = None
all_spherical_step_stats = None
all_source_step_stats = None
for i in range(N):
    upper = min((i + 1) * args.batch_size, args.num_runs)
    images_batch = images[(i * args.batch_size):upper]
    labels_batch = labels[(i * args.batch_size):upper]
    perturbed, mse_stats, distance_stats, spherical_step_stats, source_step_stats = boundary_attack(
        model, images_batch, labels_batch, max_iters=args.num_steps, spherical_step=args.spherical_step,
        source_step=args.source_step, blended_noise=args.blended_noise, transformation=trans, dct_mode=dct_mode,
        dct_ratio=args.dct_ratio, repeat_images=args.repeat_images, halve_every=args.halve_every)
    if all_perturbed is None:
        all_perturbed = perturbed
        all_mse_stats = mse_stats
        all_distance_stats = distance_stats
        all_spherical_step_stats = spherical_step_stats
        all_source_step_stats = source_step_stats
    else:
        all_perturbed = torch.cat([all_perturbed, perturbed], dim=0)
        all_mse_stats = torch.cat([all_mse_stats, mse_stats], dim=0)
        all_distance_stats = torch.cat([all_distance_stats, distance_stats], dim=0)
        all_spherical_step_stats = torch.cat([all_spherical_step_stats, spherical_step_stats], dim=0)
        all_source_step_stats = torch.cat([all_source_step_stats, source_step_stats], dim=0)
    if args.perturb_mode == 'dct':
        save_suffix = '_%.4f%s' % (args.dct_ratio, args.save_suffix)
    savefile = '%s/%s_%s_%d_%d_%d_%.4f_%.4f_%d_%s%s.pth' % (args.result_dir, args.model, args.defense, args.num_runs, args.num_steps, int(args.blended_noise), args.spherical_step, args.source_step, args.repeat_images, args.perturb_mode, args.save_suffix)
    torch.save({'original': images, 'perturbed': all_perturbed, 'mse_stats': all_mse_stats, 'distance_stats': all_distance_stats, 'spherical_step_stats': all_spherical_step_stats, 'source_step_stats': all_source_step_stats}, savefile)
