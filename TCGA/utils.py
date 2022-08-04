import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import json
import sklearn
import sys
import os
import pandas as pd


sys.path.append(os.path.join(sys.path[0], '../'))
from config import TCGA_DATA_LOC

CANCER_TYPES = {'ACC', 'CHOL', 'COADREAD', 'LIHC', 'BLCA', 'PCPG',
                'UVM', 'GBM', 'KIRP', 'DLBC', 'UCS', 'CESC', 'KICH',
                'SARC', 'GBMLGG', 'KIPAN', 'LUAD', 'OV', 'PRAD', 'TGCT',
                'UCEC', 'BRCA', 'COAD', 'HNSC', 'KIRC', 'LGG', 'PAAD',
                'LUSC', 'THCA', 'READ', 'STES'}


def restrict_to_most_mutated_genes(mut_data_np, num_genes=1000):
    """
    Restrict the mutation matrix to the most-frequently-mutated genes across cancer types.
    
    Parameters:
        mut_data_np (np.ndarray): mutation matrix, N x M
        num_genes (int): number of genes to retain
        
    Returns:
        torch tensor, shape N x `num_genes`, of mutations in most-frequently-mutated genes;
        tensor mutation mask, shape M, where `mask[m]` = 1 indicates that `m` is among the most-
        frequently-mutated genes.
    """
    N, M = mut_data_np.shape
    mut_data = torch.tensor(mut_data_np)
    mutation_counts = [torch.count_nonzero(mut_data[:, g]) for g in range(M)]
    sorted_genes = sorted(list(range(M)), key=lambda x: mutation_counts[x])
    most_mutated_genes = sorted_genes[-num_genes:]
    mutation_mask = torch.zeros(M)
    mutation_mask[most_mutated_genes] = 1
    return torch.masked_select(
        mut_data, mutation_mask.view(1, M).bool()).view(N, num_genes).detach().cpu().numpy(), mutation_mask


def get_unique_times_mutation(remove_censored_data=False):
    times = np.loadtxt(TCGA_DATA_LOC + 'patient_survival_info.txt')
    if remove_censored_data:
        times = times[times[:, 1] == 1][:, 0]  # if column 1 is 1 -> observed patient
    else:
        times = times[:, 0]
    times = times[~np.isnan(times)]
    return np.unique(times)


def get_mutation_survival_time_series(mut, surv, all_times, remove_censored_data=False, cens=None):
    """
    Compute mutation time series for a given cancer type.
    
    Parameters:
        mut (np.ndarray): mutation matrix for cancer type
        surv (np.array): survival time array for cancer type
        all_times (np.array): all survival times across all cancer types
        remove_censored_data (bool): True iff we should automatically exclude all censored patients
        cens (np.array): censoring label for patients
    """
    # Now, construct our time series for this cancer type.
    N, M = mut.shape
    if N <= 11:
        # too short to use
        return None, None, None, None, None

    mut_ts = []
    mask_ts = []
    surv_ts = np.array(sorted(all_times))
    mask_total = []
    noisy_value = []
    for t in surv_ts:
        if t not in surv:
            mut_ts.append(np.zeros(M))
            mask_ts.append(0)
            noisy_value.append(0)
            mask_total.append(np.zeros(M))
        else:
            idxs = np.where(surv == t)[0]
            if len(idxs) > 1:  # need to take most common value!
                avg_mut = np.array([np.max(mut[idxs, i].astype(int)) for i in range(M)])
            else:
                avg_mut = mut[idxs[0]]
                
            if np.count_nonzero(avg_mut) == 0:  # no mutations -> exclude patient
                mut_ts.append(np.zeros(M))
                mask_ts.append(0)
                noisy_value.append(0)
                mask_total.append(np.zeros(M))
            else:
                mask = np.ones(M)
                mut_ts.append(avg_mut)
                mask_ts.append(1)
                noisy_value.append(np.max(cens[idxs].astype(int)))  # say it's noisy if any are noisy
                mask_total.append(mask)
    
    full_mut = np.stack(mut_ts)
    return torch.tensor(full_mut), torch.tensor(surv_ts), torch.tensor(np.stack(mask_ts)), \
           torch.tensor(np.stack(mask_total)), None if remove_censored_data else torch.tensor(np.stack(noisy_value))


def load_all_mutation_data(remove_censored_data, limit_to_ct=None, restrict_to_highly_variable=False, hv_number=1000):
    """
    Construct time series dataset from TCGA mutation data.
    
    Parameters:
        remove_censored_data (bool): True iff we should automatically remove all censored patients.
        limit_to_ct (List[str]): cancer types to restrict the dataset to, or None if we should include all available
            cancer types.
        restrict_to_highly_variable (bool): True iff we should restrict the genes in the dataset to the most-frequently-
            mutated genes.
        hv_number (int): number of genes to retain, if `restrict_to_highly_variable`.
    """
    C = len(CANCER_TYPES)
    
    uTimes = get_unique_times_mutation(remove_censored_data)

    # Now, create full lists
    ct_index = []
    exprs = []
    times = []
    masks = []
    gene_masks = []
    censored_mask = []  # use 1 -> censored; 0 -> observed

    tfile = np.loadtxt(TCGA_DATA_LOC + 'patient_survival_info.txt')
    exfile = np.loadtxt(TCGA_DATA_LOC + 'mutation_matrix.txt')
    with open(TCGA_DATA_LOC + 'patient_cancer_types.txt', 'r') as f:
        ctfile = np.asarray(json.load(f))
    if remove_censored_data:
        filtering = (tfile[:, 1] == 1)
        ctfile = ctfile[filtering]
        exfile = exfile[filtering]
        censfile = (1 - tfile[filtering][:, 1])
        tfile = tfile[filtering][:, 0]
        assert np.unique(censfile) == 0, "Nothing should be censored!"
    else:
        censfile = (1 - tfile[:, 1])
        tfile = tfile[:, 0]
    
    # remove any nan survival times
    ctfile = ctfile[~np.isnan(tfile)]
    exfile = exfile[~np.isnan(tfile)]
    censfile = censfile[~np.isnan(tfile)]
    tfile = tfile[~np.isnan(tfile)]
    
    if restrict_to_highly_variable:
        exfile, hv_mask = restrict_to_most_mutated_genes(exfile, num_genes=hv_number)
    
    ct_mapping = {}
    for i, ct in tqdm(enumerate(sorted(CANCER_TYPES))):
        ct_mapping[i] = ct
        
        if limit_to_ct is not None and ct not in limit_to_ct:
            continue
            
        # let's restrict our expression, times to those cancer types
        ct_filter = (ctfile == ct)
        ct_ex = exfile[ct_filter]
        ct_ts = tfile[ct_filter]
        ct_cens = censfile[ct_filter] if censfile is not None else None
        print('...{} had {} patients'.format(ct, len(ct_ex)))
        
        ex, ts, mas, mas_tot, noisy = get_mutation_survival_time_series(
            ct_ex, ct_ts, uTimes, remove_censored_data, cens=ct_cens)

        if ex is None:
            continue  # couldn't use this cancer type!
        elif torch.count_nonzero(mas) <= 11:  # too few observations actually have expression data
            continue

        ct_index.append(torch.tensor([i]))
        exprs.append(ex)
        times.append(ts)
        masks.append(mas)
        gene_masks.append(mas_tot)
        censored_mask.append(noisy)

    if remove_censored_data:
        return torch.stack(ct_index), torch.stack(exprs), torch.stack(times), torch.stack(masks), \
               torch.stack(gene_masks), ct_mapping, hv_mask if restrict_to_highly_variable else None
    else:
        return torch.stack(ct_index), torch.stack(exprs), torch.stack(times), torch.stack(masks), \
               torch.stack(gene_masks), ct_mapping, hv_mask if restrict_to_highly_variable else None, \
               torch.stack(censored_mask)


def shuffle_data(expr, cts, ts, mask, noise=None):
    N, _ = mask.shape
    ordering = np.random.choice(range(N), size=N, replace=False)
    if noise is not None:
        return expr[ordering], cts[ordering], ts[ordering], mask[ordering], noise[ordering]
    return expr[ordering], cts[ordering], ts[ordering], mask[ordering]


def make_validation_split(original_tr_mask, val_proportion=0.2):
    N, T = original_tr_mask.shape
    Nval = int(val_proportion * torch.count_nonzero(original_tr_mask))
    print('...making {} validation samples'.format(Nval))
    optX, optY = torch.where(original_tr_mask)  # times that we _could_ mask
    val_spots = np.random.choice(range(len(optX)), size=Nval)

    val_mask = torch.zeros(N, T).to(original_tr_mask.device)
    val_mask[optX[val_spots], optY[val_spots]] = 1

    new_tr_mask = original_tr_mask.clone() * (1 - val_mask)
    return new_tr_mask, val_mask
