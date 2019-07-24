import torch
import utils
import math
import random
import numpy as np
from scipy.fftpack import dct, idct
import defenses


def boundary_attack(
    model, images, labels, targeted=False, init=None, max_iters=1000, spherical_step=0.01,
    source_step=0.01, step_adaptation=1.5, reset_step_every=50, transformation=None,
    dataset_name='imagenet', blended_noise=False, dct_mode='none', dct_ratio=1.0,
    repeat_images=1, halve_every=250):
    
    if transformation is None:
        transformation = lambda x: x
    
    images = images.cuda()
    labels = labels.cuda()
    batch_size = images.size(0)
    base_preds, _ = utils.get_preds(model, transformation(images.cuda()), dataset_name, batch_size=batch_size, return_cpu=False)
    images = images.repeat(repeat_images, 1, 1, 1)
    labels = labels.repeat(repeat_images)
    if repeat_images > 1:
        multipliers = (torch.ones(repeat_images) * 2).pow(torch.arange(0, repeat_images).float())
        dct_ratio = torch.ones(batch_size) * dct_ratio
        dct_ratio = (dct_ratio.unsqueeze(0).repeat(repeat_images, 1) * multipliers.unsqueeze(1).repeat(1, batch_size)).view(-1, 1).squeeze()
    images_vec = images.view(batch_size, -1)
    spherical_step_stats = torch.zeros(batch_size, max_iters)
    source_step_stats = torch.zeros(batch_size, max_iters)
    mse_stats = torch.zeros(batch_size, max_iters)
    distance_stats = torch.zeros(batch_size, max_iters)
    
    # sample random noise as initialization
    init = torch.zeros(images.size()).cuda()
    preds = labels.clone()
    while preds.eq(labels).sum() > 0:
        print("trying again")
        idx = torch.arange(0, batch_size).long().cuda()[preds.eq(labels)]
        noise = torch.rand(images[idx].size())
        init[idx] = noise.cuda()
        preds, _ = utils.get_preds(model, transformation(init), dataset_name, batch_size=batch_size, return_cpu=False)
    
    if blended_noise:
        min_alpha = torch.zeros(batch_size).cuda()
        max_alpha = torch.ones(batch_size).cuda()
        # binary search up to precision 2^(-10)
        for _ in range(10):
            alpha = (min_alpha + max_alpha) / 2
            alpha_expanded = alpha.view(batch_size, 1, 1, 1).expand_as(init)
            interp = alpha_expanded * init + (1 - alpha_expanded) * images
            preds, _ = utils.get_preds(model, transformation(interp), dataset_name, batch_size=batch_size, return_cpu=False)
            if targeted:
                min_alpha[preds.ne(labels)] = alpha[preds.ne(labels)]
                max_alpha[preds.eq(labels)] = alpha[preds.eq(labels)]
            else:
                min_alpha[preds.eq(labels)] = alpha[preds.eq(labels)]
                max_alpha[preds.ne(labels)] = alpha[preds.ne(labels)]
        alpha = max_alpha.view(batch_size, 1, 1, 1).expand_as(init)
        perturbed = alpha * init + (1 - alpha) * images
    else:
        perturbed = init
        
    # recording success rate of previous moves for adjusting step size
    spherical_succ = torch.zeros(batch_size, reset_step_every).cuda()
    source_succ = torch.zeros(batch_size, reset_step_every).cuda()
    spherical_steps = (torch.ones(batch_size) * spherical_step).cuda()
    source_steps = (torch.ones(batch_size) * source_step).cuda()
    
    for i in range(max_iters):
        candidates, spherical_candidates = generate_candidate(
            images, perturbed, spherical_steps, source_steps, dct_mode=dct_mode, dct_ratio=dct_ratio)
        # additional query on spherical candidate for RGB-BA
        if dct_mode:
            spherical_preds = labels + 1
        else:
            spherical_preds, _ = utils.get_preds(model, transformation(spherical_candidates), dataset_name, batch_size=batch_size, return_cpu=False)
        source_preds, _ = utils.get_preds(model, transformation(candidates), dataset_name, batch_size=batch_size, return_cpu=False)
        spherical_succ[:, i % reset_step_every][spherical_preds.ne(labels)] = 1
        source_succ[:, i % reset_step_every][source_preds.ne(labels)] = 1
        # reject moves if they result in correctly classified images
        if source_preds.eq(labels).sum() > 0:
            idx = torch.arange(0, batch_size).long().cuda()[source_preds.eq(labels)]
            candidates[idx] = perturbed[idx]
        # record some stats
        perturbed_vec = perturbed.view(batch_size, -1)
        candidates_vec = candidates.view(batch_size, -1)
        mse_prev = (images_vec - perturbed_vec).pow(2).mean(1)
        mse = (images_vec - candidates_vec).pow(2).mean(1)
        reduction = 100 * (mse_prev.mean() - mse.mean()) / mse_prev.mean()
        norms = (images_vec - candidates_vec).norm(2, 1)
        print('Iteration %d:  MSE = %.6f (reduced by %.4f%%), L2 norm = %.4f' % (i + 1, mse.mean(), reduction, norms.mean()))
        
        if (i + 1) % reset_step_every == 0:
            # adjust step size
            spherical_steps, source_steps, p_spherical, p_source = adjust_step(spherical_succ, source_succ, spherical_steps, source_steps, step_adaptation, dct_mode=dct_mode)
            spherical_succ.fill_(0)
            source_succ.fill_(0)
            print('Spherical success rate = %.4f, new spherical step = %.4f' % (p_spherical.mean(), spherical_steps.mean()))
            print('Source success rate = %.4f, new source step = %.4f' % (p_source.mean(), source_steps.mean()))
            
        mse_stats[:, i] = mse
        distance_stats[:, i] = norms
        spherical_step_stats[:, i] = spherical_steps
        source_step_stats[:, i] = source_steps
        perturbed = candidates
        
        if halve_every > 0 and perturbed.size(0) > batch_size and (i + 1) % halve_every == 0:
            # apply Hyperband to cut unsuccessful branches
            num_repeats = int(batch_size / batch_size)
            perturbed_vec = perturbed.view(batch_size, -1)
            mse = (images_vec - perturbed_vec).pow(2).mean(1).view(num_repeats, batch_size)
            _, indices = mse.sort(0)
            indices = indices[:int(num_repeats / 2)].cpu()
            idx = torch.arange(0.0, float(batch_size)).unsqueeze(0).repeat(int(num_repeats / 2), 1).long()
            idx += indices * batch_size
            idx = idx.view(-1, 1).squeeze()
            batch_size = idx.size(0)
            images = images[idx.cuda()]
            labels = labels[idx.cuda()]
            images_vec = images_vec[idx.cuda()]
            perturbed = perturbed[idx.cuda()]
            spherical_step_stats = spherical_step_stats[idx]
            source_step_stats = source_step_stats[idx]
            mse_stats = mse_stats[idx]
            distance_stats = distance_stats[idx]
            dct_ratio = dct_ratio[idx]
            spherical_steps = spherical_steps[idx.cuda()]
            source_steps = source_steps[idx.cuda()]
            spherical_succ = spherical_succ[idx.cuda()]
            source_succ = source_succ[idx.cuda()]
            
    return perturbed.cpu(), mse_stats, distance_stats, spherical_step_stats, source_step_stats
        
    
def generate_candidate(images, perturbed, spherical_steps, source_steps, dct_mode='none', dct_ratio=1.0):
    
    batch_size = images.size(0)
    unnormalized_source_direction = images - perturbed
    source_norm = unnormalized_source_direction.view(batch_size, -1).norm(2, 1)
    source_direction = unnormalized_source_direction.div(source_norm.view(batch_size, 1, 1, 1).expand_as(unnormalized_source_direction))
    
    perturbation = utils.sample_gaussian_torch(images.size(), dct_ratio=dct_ratio)
    perturbation = perturbation.cuda()
    
    if not dct_mode:
        dot = (images * perturbation).view(batch_size, -1).sum(1)
        perturbation -= source_direction.mul(dot.view(batch_size, 1, 1, 1).expand_as(source_direction))
    alpha = spherical_steps * source_norm / perturbation.view(batch_size, -1).norm(2, 1)
    perturbation = perturbation.mul(alpha.view(batch_size, 1, 1, 1).expand_as(perturbation))
    if not dct_mode:
        D = spherical_steps.pow(2).add(1).pow(-0.5)
        direction = perturbation - unnormalized_source_direction
        spherical_candidates = (images + direction.mul(D.view(batch_size, 1, 1, 1).expand_as(direction)))
    else:
        spherical_candidates = perturbed + perturbation
    spherical_candidates = spherical_candidates.clamp(0, 1)
    
    new_source_direction = images - spherical_candidates
    new_source_direction_norm = new_source_direction.view(batch_size, -1).norm(2, 1)
    length = source_steps * source_norm
    deviation = new_source_direction_norm - source_norm
    length += deviation
    length[length.le(0)] = 0
    length = length / new_source_direction_norm
    candidates = (spherical_candidates + new_source_direction.mul(length.view(batch_size, 1, 1, 1).expand_as(new_source_direction)))
    candidates = candidates.clamp(0, 1)
    
    return (candidates, spherical_candidates)


def adjust_step(spherical_succ, source_succ, spherical_steps, source_steps, step_adaptation, dct_mode='none'):
    p_spherical = spherical_succ.mean(1)
    num_spherical = spherical_succ.sum(1)
    p_source = torch.zeros(source_succ.size(0)).cuda()
    for i in range(source_succ.size(0)):
        if num_spherical[i] == 0:
            p_source[i] = 0
        else:
            p_source[i] = source_succ[i, :][spherical_succ[i].eq(1)].mean()
    if not dct_mode:
        # adjust spherical steps when using RGB-BA
        spherical_steps[p_spherical.lt(0.2)] = spherical_steps[p_spherical.lt(0.2)] / step_adaptation
        spherical_steps[p_spherical.gt(0.6)] = spherical_steps[p_spherical.gt(0.6)] * step_adaptation
    source_steps[num_spherical.ge(10) * p_source.lt(0.2)] = source_steps[num_spherical.ge(10) * p_source.lt(0.2)] / step_adaptation
    source_steps[num_spherical.ge(10) * p_source.gt(0.6)] = source_steps[num_spherical.ge(10) * p_source.gt(0.6)] * step_adaptation
    return (spherical_steps, source_steps, p_spherical, p_source)
