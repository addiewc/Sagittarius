import numpy as np
import torch
import sys
import os
from tqdm import tqdm
from scipy.stats import ttest_ind, ttest_rel, f_oneway


def load_imputed_early_development_data(specOI, minT, maxT):
    data_folder = 'results/EvoDevo/simulated_figure_trajectories/{}/'.format(specOI)
    organs = ['Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis']
    O = len(organs)
    
    examples = {}
    for org in organs:
        exs = []
        for n in range(10):
            exs.append(torch.load(data_folder + '{}/ex_{}.txt'.format(org, n)))
        examples[org] = torch.cat(exs, dim=0).detach().cpu().numpy()
    gen_ts = np.arange(minT, maxT, 0.1)
    
    return examples, organs, gen_ts


def load_imputed_aging_data(specOI, minT, maxT):
    data_folder = '../results/EvoDevo/simulated_figure_trajectories/{}/'.format(specOI)
    organs = ['Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis']
    O = len(organs)
    
    examples = {}
    for org in organs:
        exs = []
        for n in range(10):
            exs.append(torch.load(data_folder + '{}/aging_extrapolation_{}.txt'.format(org, n)))
        examples[org] = torch.cat(exs, dim=0).detach().cpu().numpy()
    gen_ts = np.arange(minT, maxT, 0.1)
    
    return examples, organs, gen_ts


def moving_average(data_ex, k=10):
    T, M = data_ex.shape
    smoothed = []
    for t in range(T):
        kmin = max(0, t-k)
        kmax = min(T, t+k)
        smoothed.append(np.mean(data_ex[kmin:kmax], axis=0))
    return np.stack(smoothed, axis=0)


def anova_test(*data):
    stat, pval = f_oneway(*data)
    return stat, pval


def find_similar_genes_across_all_organs(organ_listing, representations, M, threshold=0.05):
    similarity_mapping = {}
    for m in range(M):
        stat, pval = anova_test(*[representations[o][m].flatten() for o in organ_listing])
        similarity_mapping[m] = (stat, pval * M)
    return similarity_mapping


def find_different_genes_across_all_organs(organ_listing, representations, M, threshold=0.05):
    difference_mapping = {}
    for m in range(M):
        stat, pval = anova_test(*[representations[o][m].flatten() for o in organ_listing])
        difference_mapping[m] = (stat, pval * M)
    return difference_mapping
            
