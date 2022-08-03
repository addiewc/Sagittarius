import torch
import numpy as np
from scipy.stats import spearmanr
import scipy
import sklearn.metrics
import pandas as pd


def run_base_evaluation_rmse_spearmans(pred, gt, mask, get_per_sequence=True):
    """
    Compute basic evaluation metrics for EvoDevo and related tasks.
    
    Parameters:
        pred (Tensor): K x M tensor of simulated expression.
        gt (Tensor): K x M tensor of measured expression.
        mask (Tensor): N x T tensor indicating whether an expression was measured for each of N
            sequences; torch.count_nonzero(`mask`) = K
        get_per_sequence (bool): True if a metric value should be reported for every sequence N;
            otherwise, report summary statistic across dataset
    
    Returns:
        rmse: root mean squared error; list of length N if get_per_sequence, otherwise float
        sp_rank_by_genes: Spearman correlation (ranked by genes); list of length N if get_per_sequence,
            otherwise float
        sp_rank_by_time: Spearman correlation (ranked by time); list of length N if get_per_sequence,
            otherwise float
    """
    rmse = compute_rmse(pred, gt, get_per_sequence=get_per_sequence)
    sp_rank_by_genes = get_ranked_spearman_corr(pred, gt, get_per_sequence=get_per_sequence)
    if not get_per_sequence:
        print('rho (sp): {}; rho (pe): {}'.format(corr, corr_pear))
    if get_per_sequence:  # refactor rmse, rhos to have len(sequence)
        rmse_new = []
        corr_new = []
        for i in range(len(mask)):
            T_start, T_end = torch.count_nonzero(
                mask[:i]).item(), torch.count_nonzero(mask[:i + 1]).item()
            rmse_new.append(np.nanmean(rmse[T_start:T_end]))
            corr_new.append(np.nanmean(sp_rank_by_genes[T_start:T_end]))
        rmse = np.asarray(rmse_new)
        sp_rank_by_genes = np.asarray(corr_new)

    sp_rank_by_time = get_ranked_spearman_corr_over_time(
        pred, gt, mask, get_per_sequence=get_per_sequence)
    return rmse, sp_rank_by_genes, sp_rank_by_time


def compute_rmse(pred, gt, get_per_sequence=False):
    """
    Returns root mean squared error.
    
    Parameters:
        pred (Tensor): simulated expression
        gt (Tensor): measured expression
        get_per_sequence (bool): True iff we should return an rmse for each measurement; otherwise,
            return summary statistic
    """
    if get_per_sequence:
        return [torch.sqrt(torch.pow(pred[i] - gt[i], 2).mean()).item() for i in range(len(pred))]
    return torch.sqrt(torch.pow(pred - gt, 2).mean()).item()


def get_ranked_spearman_corr(pred, gt, get_per_sequence=False):
    """
    Return Spearman correltion (ranked by genes).
    
    Parameters:
        pred (Tensor): simulated expression
        gt (Tensor): measured expression
        get_per_sequence (bool): True iff we should return a correlation for each measurement;
            otherwise, return summary statistic
    """
    assert pred.shape == gt.shape

    if len(pred.shape) == 3:
        pred = pred.view(-1, pred.shape[-1])
        gt = gt.view(-1, gt.shape[-1])

    total_res = []
    for i in range(len(pred)):  # go through each snapshot (NT)
        ex1 = pred[i]
        ex2 = gt[i]
        M = len(ex1)

        ranked1 = sorted(range(M), key=lambda g: -ex1[g].item())
        ranked1 = [ranked1.index(g) for g in range(M)]
        ranked2 = sorted(range(M), key=lambda g: -ex2[g].item())
        ranked2 = [ranked2.index(g) for g in range(M)]

        corrt, pt = spearmanr(ranked1, ranked2)
        total_res.append(corrt)
    if get_per_sequence:
        return total_res
    return np.nanmean(total_res)


def get_ranked_spearman_corr_over_time(pred, gt, mask, get_per_sequence=False):
    """
    Returns the Spearman correlation (ranked by time).
    
    Parameters:
        pred (Tensor): simulated expression
        gt (Tensor): measured expression
        mask (Tensor): mask indicating number of measurements per sequence
        get_per_sequence (bool): True iff we should return a correlation for each sequence;
            otherwise, return summary statistic
    """
    assert pred.shape == gt.shape
    tot_spearmans = []
    M = pred.shape[-1]
    
    if len(pred.shape) == 3:  # currently N x T x M -> refactor!
        pred = pred.view(-1, pred.shape[-1])
        gt = gt.view(-1, pred.shape[-1])

    T_primes = torch.count_nonzero(mask, dim=1)
    assert torch.sum(T_primes) == len(pred), \
        "Incompatible {} and {}".format(torch.sum(T_primes), len(pred))
    for i in range(len(mask)):  # for each subject
        sp_output = pred[torch.sum(T_primes[:i]):torch.sum(T_primes[:i+1])]  # T' x M
        sp_gt = gt[torch.sum(T_primes[:i]):torch.sum(T_primes[:i+1])]  # T' x M
            
        if len(sp_output) < 2:
            if get_per_sequence:
                tot_spearmans.append([])
            continue  # can't proceed with this subject for t-rho

        rhos = []
        for m in range(M):
            rankedOut = sorted(range(T_primes[i]), key=lambda t: sp_output[t, m])
            rankedOut = [rankedOut.index(t) for t in range(T_primes[i])]
            rankedGt = sorted(range(T_primes[i]), key=lambda t: sp_gt[t, m])
            rankedGt = [rankedGt.index(t) for t in range(T_primes[i])]

            corrm, pm = spearmanr(rankedOut, rankedGt)
            rhos.append(corrm)
        tot_spearmans.append(np.nanmean(rhos))
    
    if get_per_sequence:
        return tot_spearmans
    return np.nanmean(tot_spearmans)


def construct_sequence_level_results_df(logging_file, sequences, results):
    """
    Construct dataframe with results. Saves to `logging_file`.
    
    Parameters:
        logging_file (str): filename to log results to
        sequences (List[str]): nested list (outside of length 1) with names for sequences
        results (dict[str, list[float]]): dictionary of metric results, where metric name is keyname
            and results[met][i] is the metric value for the ith sequence in `sequences`
    
    Returns:
        result dataframe
    """
    mets = sorted(results.keys())
    df_dict = {'sequence': [], **{met: [] for met in mets}}
    assert len(sequences) == 1, "Sequences should be a nested list!"
    
    df_dict['sequence'].extend(sequences[0])
    for res in results:
        df_dict[res].extend(results[res])
    
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(logging_file)
    return df
    

