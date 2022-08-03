"""
Extrapolation for EvoDevo quantitative results.
"""

import torch
import numpy as np
import os
import sys
import argparse
import json

sys.path.append(os.path.join(sys.path[0], '../'))
from models import manager_for_sagittarius
from evaluation import initialize_experiment, evaluate_results
from EvoDevo import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str, help='File to store model results')
parser.add_argument('--gpu', type=int, help='GPU to use, or None for cpu')
parser.add_argument('--seed', type=int, default=0, help='Random seed for model script')
parser.add_argument('--reload', action='store_true', help='Reload existing model file')
parser.add_argument('--verbose', action='store_true', help='Print more dataset details')
parser.add_argument('--config-file', type=str,
                    default='model_config_files/Sagittarius_config.json')
parser.add_argument('--logging-file', type=str, default=None, 
                    help='Where to log results as dataframe, or None to log to std out')

args = parser.parse_args()

device = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
seed = args.seed
initialize_experiment.initialize_random_seed(seed)

species = sorted(['Chicken', 'Rat', 'Mouse', 'Rabbit', 'Opossum', 'RhesusMacaque', 'Human'])
organs = sorted(['Brain', 'Cerebellum', 'Liver', 'Heart', 'Kidney', 'Ovary', 'Testis'])
S = len(species)
O = len(organs)
spec_vec_long, org_vec_long, expr_vec, ts_vec, mask_vec = utils.load_all_data(
    device=device, verbose=args.verbose)

# shuffle the data!
expr_vec, spec_vec_long, org_vec_long, ts_vec, mask_vec = utils.shuffle_data(
    expr_vec, spec_vec_long, org_vec_long, ts_vec, mask_vec)

expr_vec, non_stationary_mask = utils.restrict_to_nonstationary_genes(expr_vec)
N, T, M_new = expr_vec.shape
spec_vec_long = spec_vec_long[:, 0].to(device)  # N
org_vec_long = org_vec_long[:, 0].to(device)  # N
expr_vec = expr_vec.to(device)
ts_vec = ts_vec.to(device)
mask_vec = mask_vec.to(device)


def load_config_file(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

    
def get_aging_only_version(matx):
    return torch.stack([utils.restrict_to_aging_genes(vect, non_stationary_mask, device)
                        for vect in matx])


def time_transfer(mask_to_use, num_ex, ts_to_hide=3):
    # start by picking _num_ex_ sequences
    seqs = sorted(np.random.choice([i for i in range(N) if torch.count_nonzero(mask_to_use[i]) > ts_to_hide],
                                   size=num_ex))
    starting_mask = []
    target_mask = []
    for s in seqs:
        seen_ts = [ts_vec[s, t].detach().cpu().item() for t in range(T) if mask_to_use[s, t] == 1]
        hidden_ts = sorted(np.random.choice(seen_ts, size=ts_to_hide))
        s_mask = mask_to_use[s].clone()
        s_mask[hidden_ts] = 0
        t_mask = torch.zeros(T).to(device)
        t_mask[hidden_ts] = 1

        starting_mask.append(s_mask)
        target_mask.append(t_mask)
    return seqs, torch.stack(starting_mask, dim=0).to(device), torch.stack(target_mask, dim=0).to(device)


def species_transfer(mask_to_use, num_ex):
    # start by picking _num_ex_ source sequences
    candidate_seqs = sorted(np.random.choice([s for s in range(N) if torch.count_nonzero(mask_to_use[s]) > 0], size=num_ex))
    start_seqs = []
    target_seqs = []
    for s in candidate_seqs:
        # find another sequence with same organ, different species
        spec, org = spec_vec_long[s], org_vec_long[s]
        options = [o for o in range(N) if torch.count_nonzero(mask_to_use[o]) > 0 and \
                   org_vec_long[o] == org and spec_vec_long[o] != spec]
        if len(options) > 0:
            start_seqs.append(s)
            target_seqs.append(np.random.choice(options))
    return start_seqs, target_seqs


def organ_transfer(mask_to_use, num_ex):
    # start by picking _num_ex_ source sequences
    candidate_seqs = sorted(np.random.choice([s for s in range(N) if torch.count_nonzero(mask_to_use[s]) > 0], size=num_ex))
    start_seqs = []
    target_seqs = []
    for s in candidate_seqs:
        # find another sequence with same species, different organ
        spec, org = spec_vec_long[s], org_vec_long[s]
        options = [o for o in range(N) if torch.count_nonzero(mask_to_use[o]) > 0 and \
                   org_vec_long[o] != org and spec_vec_long[o] == spec]
        if len(options) > 0:
            start_seqs.append(s)
            target_seqs.append(np.random.choice(options))
    return start_seqs, target_seqs


def species_organ_transfer(mask_to_use, num_ex):
    # start by picking _num_ex_ source sequences
    candidate_seqs = sorted(np.random.choice([s for s in range(N) if torch.count_nonzero(mask_to_use[s]) > 0], size=num_ex))
    start_seqs = []
    target_seqs = []
    for s in candidate_seqs:
        # find another sequence with different organ, different species
        spec, org = spec_vec_long[s], org_vec_long[s]
        options = [o for o in range(N) if torch.count_nonzero(mask_to_use[o]) > 0 and \
                   org_vec_long[o] != org and spec_vec_long[o] != spec]
        if len(options) > 0:
            start_seqs.append(s)
            target_seqs.append(np.random.choice(options))
    return start_seqs, target_seqs


def compute_transfer_tasks(mask_to_use):
    # transfer tasks -> across time, across species, across organ, across both
    N_task = int(N / 4)
    seqs, smask, tmask = time_transfer(mask_to_use, N_task)

    # start getting input lists going
    starting_expr, target_expr = [expr_vec[s] for s in seqs], [expr_vec[s] for s in seqs]
    starting_ts, target_ts = [ts_vec[s] for s in seqs], [ts_vec[s] for s in seqs]
    starting_spec, target_spec = [spec_vec_long[s] for s in seqs], [spec_vec_long[s] for s in seqs]
    starting_org, target_org = [org_vec_long[s] for s in seqs], [org_vec_long[s] for s in seqs]
    starting_mask, target_mask = [smask[i] for i in range(N_task)], [tmask[i] for i in range(N_task)]

    other_seqs = [species_transfer(mask_to_use, N_task), organ_transfer(mask_to_use, N_task),
                  species_organ_transfer(mask_to_use, N_task)]
    for stS, taS in other_seqs:
        starting_expr.extend([expr_vec[s] for s in stS])
        target_expr.extend([expr_vec[s] for s in taS])
        starting_ts.extend([ts_vec[s] for s in stS])
        target_ts.extend([ts_vec[s] for s in taS])
        starting_spec.extend([spec_vec_long[s] for s in stS])
        target_spec.extend([spec_vec_long[s] for   s in taS])
        starting_org.extend([org_vec_long[s] for s in stS])
        target_org.extend([org_vec_long[s] for s in taS])
        starting_mask.extend([mask_vec[s] for s in stS])
        target_mask.extend([mask_vec[s] for s in taS])

    return [torch.stack(x, dim=0).to(device) for x in (
        starting_expr, starting_ts, starting_spec, starting_org, starting_mask,
        target_expr, target_ts, target_spec, target_org, target_mask)]


# Now, conduct extrapolation experiment task
initialize_experiment.initialize_random_seed(seed)

# Define train/val/test splits
mask = torch.tensor(mask_vec)
hidden_times = utils.get_hidden_times_range_extrap(mask)
hidden_mask = np.ones((N, T))
hidden_mask[np.arange(N), hidden_times[:, 0]] = 0
hidden_mask[np.arange(N), hidden_times[:, 1]] = 0
hidden_mask[np.arange(N), hidden_times[:, 2]] = 0
hidden_mask[np.arange(N), hidden_times[:, 3]] = 0
hidden_mask = torch.tensor(hidden_mask).to(device)
rev_mask = ~hidden_mask.bool().clone()
hidden_mask = mask * hidden_mask
tr_mask, val_mask = utils.make_validation_split(hidden_mask)

# Train the model
sagittarius_manager = manager_for_sagittarius.Sagittarius_Manager(
    M_new, 2, [S, O], **load_config_file(args.config_file), minT=0, maxT=T,
    device=device, train_transfer=True)

sEx, sT, sSpec, sOrg, sM, tEx, tT, tSpec, tOrg, tM = compute_transfer_tasks(tr_mask)
sagittarius_manager.train_model(
    expr_vec, ts_vec, [spec_vec_long, org_vec_long], tr_mask,
    reload=args.reload, mfile=args.model_file, val_mask=val_mask,
    transfer_expr=(sEx, tEx), transfer_ts=(sT, tT), transfer_ys=([sSpec, sOrg], [tSpec, tOrg]),
    transfer_mask=(sM, tM))

# Evaluate the model
initialize_experiment.initialize_random_seed(seed)

recon = sagittarius_manager.reconstruct(maxN=N).view(-1, M_new)  # don't reconstruct transfer tasks
vres = sagittarius_manager.generate(val_mask).view(-1, M_new)

# Check extrapolation to hidden times
unseen_expr = torch.masked_select(
    expr_vec, torch.stack([rev_mask.bool() for _ in range(M_new)], dim=-1)
).to(device).view(-1, M_new)  # measured expression
h_extrap = sagittarius_manager.generate(rev_mask).view(-1, M_new)  # simulated expression
h_extrap = get_aging_only_version(h_extrap).view(h_extrap.shape[0], -1)  # restrict to HAGR genes
unseen_expr = get_aging_only_version(unseen_expr).view(h_extrap.shape[0], -1)
gen_res = evaluate_results.run_base_evaluation_rmse_spearmans(
    h_extrap, unseen_expr, rev_mask, get_per_sequence=True)

ordered_seqs = ['{}:{}'.format(species[spec_vec_long[i]], organs[org_vec_long[i]]) for i in range(N)]

if args.logging_file is not None:
    evaluation_results.construct_sequence_level_results_df(args.logging_file, ordered_seqs, gen_res) 
else:
    print('Average rmse: {:.3f}'.format(np.mean(gen_res[0])))
    print('Average Spearman (rank by genes): {:.3f}'.format(np.mean(gen_res[1])))
    print('Average Spearman (rank by times): {:.3f}'.format(np.mean(gen_res[2])))
    print('\n')
    print('Sequence \t RMSE \t Spearman (rank by genes) \t Spearman (rank by time)')
    print('----------------------------------------------------------------------------')
    for i in range(len(ordered_seqs)):
        print(ordered_seqs[i], '\t{:.3f}\t{:.3f}\t{:.3f}'.format(
            gen_res[0][i], gen_res[1][i], gen_res[2][i]))
