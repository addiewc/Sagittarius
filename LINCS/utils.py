"""
Utils file for LINCS experiments.
"""

import numpy as np
import torch
import sys
import os
import json

sys.path.append(os.path.join(sys.path[0], '../'))
from LINCS import LINCS_dataset


def load_all_joint_data(seed, device, use_onehot, taskname, use_cvae_version=False):
    """
    Load dataloader for LINCS.
    
    Parameter:
        seed (int): random seed to use in dataloader
        device (str): device to use for dataset
        use_onehot (bool): True iff we should use one-hot encoding
        taskname (str): `full_dataset` to use all available data; 
            `complete_generation`, `combination_and_dose`, `combination_and_time` for
            the three defined extrapolation tasks.
    """
    assert taskname in {
        'full_dataset', 'complete_generation', 'combination_and_dose',
        'combination_and_time'}, "Unknown task {}".format(taskname)
    
    kwargs = {
        'num_unseen_combos': 0 if taskname == 'full_dataset' else 5,
        'num_unseen_dosages': 0 if taskname in {'full_dataset', 'combination_and_time'} else 3,
        'num_unseen_times': 0 if taskname in {'full_dataset', 'combination_and_dose'} else 3,
        'generate_joint_unseen': taskname == 'complete_generation',
        'generate_combo_dosage_joint_unseen': taskname == 'combination_and_dose',
        'generate_combo_time_joint_unseen': taskname == 'combination_and_time'
    }
    
    if not use_cvae_version:
        if taskname == 'full_dataset':
            return LINCS_dataset.Lincs2dDataLoader(seed, device, use_onehot)
        dl = LINCS_dataset.Lincs2dDataLoaderWithUnseen(
            seed, device, use_onehot, **kwargs)
    else:
        # otherwise give the cvae version
        dl = LINCS_dataset.Lincs2dDataLoaderWithUnseenForCvae(
            seed, device, use_onehot, **kwargs)
    return dl


def reconstruct(dl):
    """
    Summarize measurements from all perturbations.
    
    Parameters:
        dl: LINCS dataloader object
    
    Returns:
        K x T Tensor of expression where K is the number of measurements in the
            training split of `dl`.
    """
    rec_expr = []
    full_mask = []
    for expr, _, _, _, _, mask in dl.get_data_loader('train'):
        N, T, M = expr.shape
        vis_ex = torch.masked_select(expr, mask.view(N, T, 1).bool()).view(-1, M)
        rec_expr.append(vis_ex)
        full_mask.append(mask)
    return torch.cat(rec_expr, dim=0).view(-1, M), torch.cat(
        full_mask, dim=0).view(-1, T)


def generate(dl):
    """
    Summarize measurements for generation task.
    
    Parameters:
        dl: LINCS dataloader object with a source and target listing.
    
    Return:
        K x T tensor of expression where K is the number of measurements in the target
            expressions of `dl`.
    """
    rec_expr = []
    full_mask = []
    for source, target in dl:
        expr, _, _, _, _, mask = target  # what we care about here!
        N, T, M = expr.shape
        vis_ex = torch.masked_select(expr, mask.view(N, T, 1).bool()).view(-1, M)
        rec_expr.append(vis_ex)
        full_mask.append(mask)
    return torch.cat(rec_expr, dim=0).view(-1, M), torch.cat(
        full_mask, dim=0).view(-1, T)
