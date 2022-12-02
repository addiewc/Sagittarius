"""
Utils file for EvoDevo experiments.
"""

import sys
import os

import warnings
import torch
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr
import scipy
import sklearn.metrics
import matplotlib.pyplot as plt

sys.path.append(os.path.join(sys.path[0], '../'))
from EvoDevo import EvoDevo_dataset


def shuffle_data(expr, spec, org, ts, mask):
    """
    Shuffle the dataset given by expr, spec, org, ts, mask.
    """
    N, _ = mask.shape
    ordering = np.random.choice(range(N), size=N, replace=False)
    return expr[ordering], spec[ordering], org[ordering], ts[ordering], mask[ordering]


def make_validation_split(original_tr_mask, seqs=None, val_proportion=0.2):
    """
    Defines a validation split.
    
    Parameters:
        original_tr_mask (Tensor): mask for train/val split
        seqs (List[int]): sequences that must be entirely training data, or None if no
            such sequences
        val_proportion (float): fraction of training measurements to use as validation data
        
    Returns:
        new_tr_mask (Tensor): mask to use as train split
        val_mask (Tensor): mask to use as val split
    """
    N, T = original_tr_mask.shape
    Nval = int(val_proportion * torch.count_nonzero(original_tr_mask))
    print('...making {} validation samples'.format(Nval))
    optX, optY = torch.where(original_tr_mask)  # times that we _could_ mask
    val_spots = np.random.choice(range(len(optX)), size=Nval)

    val_mask = torch.zeros(N, T).to(original_tr_mask.device)
    val_mask[optX[val_spots], optY[val_spots]] = 1
    
    if seqs is not None:
        for s in seqs:
            val_mask[s, np.arange(T)] = 0  # these must be training sequences!

    new_tr_mask = original_tr_mask.clone() * (1 - val_mask)
    return new_tr_mask, val_mask


def get_hidden_times_range_extrap(mask):
    """
    Return N x 4 np matrix of the indices of the last 4 measured time points for each time series.
    
    Parameters:
        mask (Tensor): measured time points mask, shape N x T
    """
    N, T = mask.shape

    # take last 4 _observed_ times
    obs_idxs = [torch.where(mask[i])[0] for i in range(N)]
    hidden_times = torch.stack([obs_idxs[i][-4:] for i in range(N)])

    for i in range(N):
        assert all([(mask[i, t] == 1).item() for t in hidden_times[i]]), \
            "Problem with ht={} for ot={} @ seq {}".format(hidden_times[i], torch.where(mask[i]))
    return hidden_times.detach().cpu().numpy()


def limit_data_to_species_organ_combo(specOI, orgOI, device):
    """
    Load reduce dataset with only a given species and organ.
    
    Parameters:
        specOI (str): species to keep
        orgOI (str): organ to keep
        device (str): device to put dataset on.
    """
    species = sorted(['Chicken', 'Rat', 'Mouse', 'Rabbit', 'Opossum', 'RhesusMacaque', 'Human'])
    organs = sorted(['Brain', 'Cerebellum', 'Liver', 'Heart', 'Kidney', 'Ovary', 'Testis'])

    full_spec, full_org, full_expr, full_ts, full_mask = load_all_data(device)
    specIdx = species.index(specOI)
    orgIdx = organs.index(orgOI)
    specOptions = set(np.where(full_spec == specIdx)[0])
    orgOptions = set(np.where(full_org == orgIdx)[0])
    finalOptions = sorted(list(specOptions.intersection(orgOptions)))
    return full_spec[finalOptions], full_org[finalOptions], full_expr[finalOptions], \
           full_ts[finalOptions], full_mask[finalOptions]


def load_all_data(device, verbose=False):
    """
    Load Evo-Devo dataset.
    
    Parameters:
        device (str): device to put dataset on
        verbose (bool): True iff more sequence details should be reported via std out
        
    Returns:
        spec_vec (Tensor): N x 1 tensor of species indices
        org_vec (Tensor): N x 1 tensor of organ indices
        expr_vec (Tensor): N x T x M tensor of gene expression measurements, or 0 if unmeasured
        ts_vec (Tensor): N x T tensor of measurement time points
        mask_vec (Tensor): N x T tensor where `mask_vec[i, t]` = 1 indicates that `expr_vec[i, t]` was
            measured
    """
    ds_train = EvoDevo_dataset.EvoDevoDataset(train=True, device=device)
    ds_test = EvoDevo_dataset.EvoDevoDataset(
        train=False, device=device, train_ordering=ds_train.ordering)
    species = sorted(['Chicken', 'Rat', 'Mouse', 'Rabbit', 'Opossum', 'RhesusMacaque', 'Human'])
    organs = sorted(['Brain', 'Cerebellum', 'Liver', 'Heart', 'Kidney', 'Ovary', 'Testis'])
    S = len(species)
    O = len(organs)

    if verbose:
        # print out the available <species, organ> combinations
        found = {}
        for s, o, t, ex, m in ds_train:
            sres = torch.argmax(s).item()
            if species[sres] not in found:
                found[species[sres]] = []
            ores = torch.argmax(o).item()
            found[species[sres]].append((organs[ores], m.count_nonzero()))
        for s, o, t, ex, m in ds_test:
            sres = torch.argmax(s).item()
            if species[sres] not in found:
                found[species[sres]] = []
            ores = torch.argmax(o).item()
            found[species[sres]].append((organs[ores], m.count_nonzero()))
        for s in found:
            print(s, ':')
            for o, t in found[s]:
                print('\t', o, '(', t, ')')

    M = 5037

    spec_list = []
    org_list = []
    ts_list = []
    expr_list = []
    mask_list = []
    for s, o, t, ex, m in ds_train:
        spec_list.append(torch.argmax(s))
        org_list.append(torch.argmax(o))
        ts_list.append(t)
        expr_list.append(ex)
        mask_list.append(m)

    for s, o, t, ex, m in ds_test:
        spec_list.append(torch.argmax(s))
        org_list.append(torch.argmax(o))
        ts_list.append(t)
        expr_list.append(ex)
        mask_list.append(m)

    spec_vec = torch.tensor(spec_list).view(-1, 1)  # N x 1
    org_vec = torch.tensor(org_list).view(-1, 1)  # N x 1
    expr_vec = torch.stack(expr_list, dim=0)  # N x T x M
    ts_vec = torch.stack(ts_list, dim=0)  # N x T
    mask_vec = torch.stack(mask_list, dim=0)  # N x T

    return spec_vec, org_vec, expr_vec, ts_vec, mask_vec


def restrict_to_nonstationary_genes(expr_vec):
    """
    Restrict expr dataset to genes that failed to reject null hypothesis in ADF test based on 0th
    time series (ADF p >= 0.05).
    
    Parameters:
        expr_vec (Tensor): N x T x M expression measurements
    
    Returns:
        expr_vec (Tensor): N x T x M' expression measurements for non-stationary genes
        non_stationary_mask (Tensor): M' mask where `non_stationary_mask[m] = 1` indicates that we
            retained gene m
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Restrict to genes of "interest"
        non_stationary_genes = []
        N, T, M = expr_vec.shape
        threshold = 0.05

        for g in range(M):
            g_ts = expr_vec[0, :, g].detach().numpy()
            adf_stat, adf_p, _, _, adf_crits, _ = adfuller(g_ts)
            confirmed_stationary = False
            if adf_p < threshold:
                confirmed_stationary = True
            if not confirmed_stationary:
                # if we reach here: can't reject null hypothesis -> gene time series is non-stationary
                non_stationary_genes.append(((adf_stat, adf_p), g))

        non_stationary_mask = torch.zeros((M))
        non_stationary_mask[[g for _, g in non_stationary_genes]] = 1

        expr_vec = torch.masked_select(expr_vec, torch.stack(
            [non_stationary_mask for _ in range(T)], dim=0).view(1, T, M).bool()).view(N, T, -1)
        return expr_vec, non_stationary_mask


def get_num_aging_genes(non_stationary_mask):
    """
    Count number of aging-related genes M'.
    
    Parameters:
        non_stationary_mask (Tensor): M tensor, where `non_stationary_mask[m]` = 1 indicates that `m`
            is included in filtered dataset; torch.count_nonzero(`non_stationary_mask`) = M'
    """
    agingdf = pd.read_csv(HAGR_AGING_GENES_LOC, names=['Idx', 'Gene'])
    indices = sorted(agingdf['Idx'].values)
    aging_mask = torch.zeros(len(non_stationary_mask))
    aging_mask[indices] = 1
    relevant_aging_mask = torch.masked_select(aging_mask, non_stationary_mask.bool())
    return torch.count_nonzero(relevant_aging_mask)
