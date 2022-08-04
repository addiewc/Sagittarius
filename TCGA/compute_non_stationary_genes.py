import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import torch
import sys
import os
from itertools import combinations

sys.path.append(os.path.join(sys.path[0], '../'))
from config import NON_STATIONARY_GENES_DIR


def compute_non_stationary(expr, gene_mask, threshold=0.05):
    """
    Determine which genes are non-stationary for at least one cancer type.
    
    Parameters:
        expr (Tensor): N x T x M tensor of mutations.
        gene_mask (Tensor): N x T x M tensor where `gene_mask[i, t, m] = 1` indicates that we are
            confident in the mutation/lack-thereof in gene `m` for patient with survival time indexed
            `t` in cancer type indexed `i`.
        threshold (float): significance threshold for ADF test.
    """
    non_stationary_genes = {}  # ct -> result
    N, T, M = expr.shape
    for i in range(N):
        non_stationary_genes[i] = []
        for g in range(M):
            g_ts = torch.masked_select(expr[i, :, g], gene_mask[i, :, g].bool()).detach().numpy()
            if len(g_ts) < 4:
                continue
            adf_stat, adf_p, _, _, adf_crits, _ = adfuller(g_ts)
            confirmed_stationary = False
            if adf_p < threshold:
                confirmed_stationary = True
            if not confirmed_stationary:
                # if we reach here: can't reject null hypothesis -> gene time series is non-stationary
                non_stationary_genes[i].append(((adf_stat, adf_p), g))
    return non_stationary_genes


def get_NchooseK_combos(k):
    N = len(cts)
    combos = combinations(range(N), k)
    return [c for c in combos]


def find_k_intersection_genes(k, ns_genes):
    combos = get_NchooseK_combos(k)
    set_options = [set.intersection(*[set(ns_genes[i]) for i in j]) for j in combos]
    full_res = set.union(*set_options)
    return full_res


def get_non_stationary_k_mask(k, reload=True, expr=None, gene_mask=None, threshold=0.05, save_result=False):
    """
    Get the non-stationary mask intersection from k cancer types.
    
    Parameters:
        reload (bool): True iff we should load the existing mask from file
        expr (Tensor): mutation matrix. Needed if not `reload`
        gene_mask (Tensor): mask indicating whether a gene was fully observed. Needed if not `reload`
        threshold (float): threshold for ADF test. Only used if not `reload`
        save_result (bool): True iff we should save the new result to a file. Only used if not `reload`
    """
    if reload:
        fname = NON_STATIONARY_GENES_DIR + 'k={}.txt'.format(k)
        if not os.path.exists(fname):
            print('Unable to preload for non-stationary k={}; no such file {}'.format(k, fname))
            assert False
        return np.loadtxt(fname)
    
    assert expr is not None, "Must provide mutation matrix if not reloading!"
    assert gene_mask is not None, "Must provide gene mask if not reloading!"
    
    non_stationary_genes = compute_non_stationary(expr, gene_mask, threshold=threshold)
    
    genes = sorted(list(find_k_intersection_genes(k=k, ns_genes=non_stationary_genes)))
    full_mask = np.zeros(M)
    full_mask[genes] = 1
    if save_result:
        np.savetxt(NON_STATIONARY_GENES_DIR + 'k={}.txt'.format(k), full_mask)
    return full_mask
