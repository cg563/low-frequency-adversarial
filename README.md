This repository contains code for the UAI 2019 paper:

Chuan Guo, Jared S. Frank,Kilian Q. Weinberger. Low Frequency Adversarial Perturbation.
https://arxiv.org/abs/1809.08758

Our code uses PyTorch (pytorch >= 0.4.1, torchvision >= 0.2.1) with CUDA 9.0 and Python 3.5.

Both RGB-BA (original boundary attack) and LF-BA (low frequency boundary attack) are implemented. Before running the code, make sure that the output directory exists (default ./save).

Notable options:
--defense: Type of transformation defense to evaluate against [none/jpeg/bit]
--dct_ratio: Frequency ratio r. Recommend using 1/32 with Hyperband to select from {1/4, 1/8, 1/16, 1/32}

To run RGB-BA:
```
python run_dba_simple.py --data_root <imagenet_root> --num_runs 1000 --num_steps 30000 --defense <defense> --perturb_mode gaussian --dct_ratio 1.0 --blended_noise
```
To run LF-BA with fixed frequency ratio r:
```
python run_dba_simple.py --data_root <imagenet_root> --num_runs 1000 --num_steps 30000 --defense <defense> --perturb_mode dct --dct_ratio <dct_ratio> --blended_noise
```
To run LF-BA using Hyperband to tune the frequency ratio r:
```
python run_dba_simple.py --data_root <imagenet_root> --num_runs 1000 --num_steps 28000 --defense <defense> --perturb_mode dct --dct_ratio <dct_ratio> --blended_noise --repeat_images 4 --halve_every 500
```
which starts with 4 parallel runs using frequency ratios r, 2r, 4r, 8r, halving every 500 iterations. With 28000 steps, this amounts to a total of 30000 model queries.

Our code also contains two useful utility functions for sampling low frequency random perturbations: utils.sample_gaussian_torch and utils.sample_gaussian_tf, for use with PyTorch and Tensorflow. Any black-box attack that relies on sampling Gaussian noise can be modified to do low frequency perturbation using the two sampling functions.
