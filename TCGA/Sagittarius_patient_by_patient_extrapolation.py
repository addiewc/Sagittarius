import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch
import argparse
from importlib import reload
import json

sys.path.append(os.path.join(sys.path[0], '../'))
from TCGA import utils, compute_non_stationary_genes, filter_censored_patients
from evaluation import initialize_experiment, evaluate_results
from models import manager_for_sagittarius


parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str, help='Directory to store model results')
parser.add_argument('--cancer-type', type=str, help='Acronym for cancer type to split into train/test')
parser.add_argument('--non-stationary-k', type=int, 
                    help='Number of cancer types k to compute non-stationary intersection;' +\
                    'to mimic paper, use 4 for SARC and 2 for THCA')
parser.add_argument('--gpu', type=int, default=None, help='GPU to use, or None for cpu')
parser.add_argument('--seed', type=int, default=0, help='Random seed for model script')
parser.add_argument('--reload', action='store_true', help='Reload existing model files')
parser.add_argument('--config-file', type=str, default='model_config_files/Sagittarius_config.json')
parser.add_argument('--logging-file', type=str, default=None, 
                    help='Where to log results as dataframe, or None to log to std out')

args = parser.parse_args()

device = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
seed = args.seed
initialize_experiment.initialize_random_seed(seed)


def load_config_file(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)
    
    
def restrict_genes(expr):
    gene_filter = torch.tensor(ns_mask).to(device)
    top_gene_mask = gmask.to(device)
    gene_filter = torch.masked_select(gene_filter, top_gene_mask.bool()).view(-1)
    new_expr = torch.masked_select(expr, gene_filter.view(1, -1).bool()).view(expr.shape[0], -1)
    return new_expr
    
    
def time_transfer(mask_to_use, num_ex, ts_to_hide=3):
    # start by picking _num_ex_ sequences
    seqs = sorted(np.random.choice([i for i in range(N) if torch.count_nonzero(mask_to_use[i]) > ts_to_hide],
                                   size=num_ex))
    starting_mask = []
    target_mask = []
    for s in seqs:
        seen_ts = [ts[s, t].detach().cpu().item() for t in range(T) if mask_to_use[s, t] == 1]
        hidden_ts = sorted(np.random.choice(seen_ts, size=ts_to_hide))
        s_mask = mask_to_use[s].clone()
        s_mask[hidden_ts] = 0
        t_mask = torch.zeros(T).to(device)
        t_mask[hidden_ts] = 1

        starting_mask.append(s_mask)
        target_mask.append(t_mask)
    return seqs, torch.stack(starting_mask, dim=0).to(device), torch.stack(target_mask, dim=0).to(device)


def cancer_transfer(mask_to_use, num_ex):
    # start by picking _num_ex_ source sequences
    candidate_seqs = sorted(np.random.choice([s for s in range(N) if torch.count_nonzero(mask_to_use[s]) > 0],
                                             size=num_ex))
    start_seqs = []
    target_seqs = []
    for s in candidate_seqs:
        options = [o for o in range(N) if o != s and torch.count_nonzero(mask_to_use[o]) > 0]
        start_seqs.append(s)
        target_seqs.append(np.random.choice(options))
    return start_seqs, target_seqs


def compute_transfer_tasks(mask_to_use, expr_to_use):
    # transfer tasks -> across time, across cancer types
    N_task = int(N / 2)
    seqs, smask, tmask = time_transfer(mask_to_use, N_task)

    # start getting input lists going
    starting_expr, target_expr = [expr_to_use[s] for s in seqs], [expr_to_use[s] for s in seqs]
    starting_ts, target_ts = [ts[s] for s in seqs], [ts[s] for s in seqs]
    starting_ct, target_ct = [ct_vec[s] for s in seqs], [ct_vec[s] for s in seqs]
    starting_mask, target_mask = [smask[i] for i in range(N_task)], [tmask[i] for i in range(N_task)]

    other_seqs = [cancer_transfer(mask_to_use, N_task)]
    for stS, taS in other_seqs:
        starting_expr.extend([muts[s] for s in stS])
        target_expr.extend([muts[s] for s in taS])
        starting_ts.extend([ts[s] for s in stS])
        target_ts.extend([ts[s] for s in taS])
        starting_ct.extend([ct_vec[s] for s in stS])
        target_ct.extend([ct_vec[s] for s in taS])
        starting_mask.extend([mask[s] for s in stS])
        target_mask.extend([mask[s] for s in taS])

    return [torch.stack(x, dim=0).to(device) for x in (
        starting_expr, starting_ts, starting_ct, starting_mask,
        target_expr, target_ts, target_ct, target_mask)]


def compute_mask_for_ct(pidx):
    held_idx = [i for i in range(N) if ct_mapping[ct_vec[i].item()] == args.cancer_type]
    assert len(held_idx) == 1

    # masks for this cancer type
    observed_patients = torch.where(mask[held_idx] * (1-censoring[held_idx]))[1]
    hmask = torch.tensor([
        (mask[held_idx] * (1-censoring[held_idx]))[0, i] for i in range(observed_patients[pidx].item())] + [
        0 for _ in range(observed_patients[pidx].item(), T)]).unsqueeze(0).to(device)
    rmask = torch.where(hmask.float() == 0, mask[held_idx] * (1-censoring[held_idx]).float(), torch.zeros(hmask.shape).to(device)).float()
    N_seen = torch.count_nonzero(hmask).item()
    N_hid = torch.count_nonzero(rmask).item()
    print('Saw {} patients; hid {} patients; patient was {}'.format(N_seen, N_hid, pidx))

    # now construct full mask
    full_hmask = torch.cat([mask[:held_idx[0]], hmask, mask[held_idx[0] + 1:]], dim=0)  # N x T
    full_rmask = torch.cat([torch.zeros(held_idx[0], T).to(device), rmask,
                            torch.zeros(N - held_idx[0] - 1, T).to(device)], dim=0)  # N x T
    assert full_hmask.shape == (N, T)
    assert full_rmask.shape == (N, T)

    return full_hmask, full_rmask


def restrict_to_ctOI(reslists, ct):
    for i in range(len(reslists)):
        assert len(reslists[i]) == N, "Expected list of {} elements but got {}".format(N, len(reslist))
    ct_idx = [i for i in range(N) if ct_mapping[ct_vec[i].item()] == ct][0]
    return [[rl[ct_idx]] for rl in reslists]
    

# Now, conduct complete generation experiment
initialize_experiment.initialize_random_seed(seed)

highly_mutated = True  # check the gene mask that we should use
ct_vec, muts, ts, mask, _, ct_mapping, gmask, censoring = utils.load_all_mutation_data(
    remove_censored_data=False, restrict_to_highly_variable=highly_mutated)
N, T, M = muts.shape
C = len(ct_mapping)

ct_vec = ct_vec[:, 0].to(device)
muts = muts.to(device)
ts = ts.to(device)
mask = mask.to(device)
gmask = gmask.to(device)
censoring = censoring.to(device)

cleaner = filter_censored_patients.filter_cancer_type_time_series(
    muts, ts, censoring, mask, ct_vec, ct_mapping, load_from_file=True).to(device)
mask = mask * cleaner  # filter censored patients
maxT = torch.max(torch.masked_select(ts, mask.bool())).item()

ns_mask = compute_non_stationary_genes.get_non_stationary_k_mask(args.non_stationary_k, reload=True)

ct_idx = [i for i in range(N) if ct_mapping[ct_vec[i].item()] == args.cancer_type][0]
N_models = (mask * (1-censoring))[ct_idx].count_nonzero()
"Have {} models for {}".format(N_models, args.cancer_type)

aurocs = []
for pidx in range(1, N_models):  # no ZSL
    print('\n...on {}/{}'.format(pidx, N_models-1))
    
    initialize_experiment.initialize_random_seed(seed)
    hidden_mask, rev_mask = compute_mask_for_ct(pidx)
    tr_mask, val_mask = utils.make_validation_split(hidden_mask)

    mfile = args.model_dir + 'model_{}.pth'.format(pidx)
    sagittairus_manager = manager_for_sagittarius.Sagittarius_Manager(
        M, 1, [C], **load_config_file(args.config_file), minT=0, maxT=maxT, device=device, train_transfer=True, 
        rec_loss='bce', batch_size=2)
    
    sEx, sT, sCt, sM, tEx, tT, tCt, tM = compute_transfer_tasks(tr_mask, muts)
    sagittairus_manager.train_model(
        muts, ts, [ct_vec], tr_mask, reload=args.reload, mfile=mfile,
        val_mask=val_mask, transfer_expr=(sEx, tEx), transfer_ts=(sT, tT),
        transfer_ys=([sCt], [tCt]), transfer_mask=(sM, tM))

    # Start evaluation
    initialize_experiment.initialize_random_seed(seed)

    recon = sagittairus_manager.reconstruct().view(-1, M)
    h_extrap = sagittairus_manager.generate(rev_mask).view(-1, M)
    gt_extrap = torch.masked_select(muts, rev_mask.unsqueeze(-1).bool()).view(-1, M)
    
    filt_sim = restrict_genes(h_extrap)
    filt_meas = restrict_genes(gt_extrap)
    
    aurocs.append(evaluate_results.compute_auroc(filt_sim, filt_meas))
    

seqs = ['split_{}'.format(x) for x in range(len(aurocs))]
avg_perf = [np.nanmean(aurocs[x]) for x in range(len(aurocs))]

if args.logging_file is not None:
    evaluate_results.construct_sequence_level_results_df(args.logging_file, seqs, avg_perf) 
else:
    print('Average AUROC: {:.3f}'.format(np.mean(avg_perf)))
    print('\n')
    print('Split \t AUROC')
    print('-------------------------')
    for i in range(len(seqs)):
        print(seqs[i], '\t{:.3f}'.format(avg_perf[i]))
