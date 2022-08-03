import os
import sys
import argparse
import torch
import json
import numpy as np

sys.path.append(os.path.join(sys.path[0], '../'))
from models import manager_for_sagittarius
from evaluation import evaluate_results, initialize_experiment
from LINCS import utils


parser = argparse.ArgumentParser()
parser.add_argument('--model-file', type=str, help='File to store model results')
parser.add_argument('--gpu', type=int, default=None, help='GPU to use, or None for cpu')
parser.add_argument('--seed', type=int, default=0, help='Random seed for model script')
parser.add_argument('--reload', action='store_true', help='Reload existing model file')
parser.add_argument('--config-file', type=str, default='model_config_files/Sagittarius_config.json')
parser.add_argument('--dated-run', action='store_true', help='Save this model with timestamp and exact configs')
parser.add_argument('--logging-file', type=str, default=None, 
                    help='Where to log results as dataframe, or None to log to std out')

args = parser.parse_args()

device = 'cpu' if args.gpu is None else 'cuda:{}'.format(args.gpu)
seed = args.seed
initialize_experiment.initialize_random_seed(seed)

transfer_kwargs = {
    'N_drug': 32,
    'N_cell': 32,
    'N_random': 16,
}


def load_config_file(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)
    

# Now, conduct complete generation experiment
initialize_experiment.initialize_random_seed(seed)

dl = utils.load_all_joint_data(
    seed, device, use_onehot=False, taskname='combination_and_time', use_cvae_version=False)

D = len(dl.get_drug_list())
C = len(dl.get_cell_list())
M = dl.get_feature_dim()
max_dsg = dl.get_max_dosage()
max_time = dl.get_max_time()

sagittarius_manager = manager_for_sagittarius.Sagittarius_Manager_DataLoader(
    M, 2, [D, C], **load_config_file(args.config_file), minT=0, maxT=max_dsg,
    device=device, batch_size=4, train_transfer=True, num_cont=2,
    other_minT=[0], other_maxT=[max_time])

tr_dl = dl.get_transfer_data_loader(**transfer_kwargs)
sagittarius_manager.train_model(dl, reload=args.reload, mfile=args.model_file,
                                transfer_dl=tr_dl)

# Start evaluation
initialize_experiment.initialize_random_seed(seed)

recon = sagittarius_manager.reconstruct().view(-1, M)

gen_dl = dl.get_gen_data_loader()
gen = sagittarius_manager.generate(gen_dl).view(-1, M)
expr, full_mask = utils.generate(gen_dl)
gen_res = evaluate_results.get_ranked_spearman_corr(
    gen, expr, get_per_sequence=True)

rev_dr_mapping = {idx: dr for dr, idx in dl.train_dataset.drug_id_to_idx_mapping.items()}
rev_ce_mapping = {idx: ce for ce, idx in dl.train_dataset.cell_id_to_idx_mapping.items()}
seqs = []
for source, targets in gen_dl:  # these are batches!!!
    for i in range(len(source[0])):
        _, sdr, sce, _, _, _ = [s[i] for s in source]
        _, tdr, tce, _, _, _ = [t[i] for t in targets]
        seqs.append('{}:{} -> {}:{}'.format(
            rev_dr_mapping[torch.argmax(sdr).item()], 
            rev_ce_mapping[torch.argmax(sce).item()],
            rev_dr_mapping[torch.argmax(tdr).item()], 
            rev_ce_mapping[torch.argmax(tce).item()]))

if args.logging_file is not None:
    evaluate_results.construct_sequence_level_results_df(args.logging_file, seqs, gen_res) 
else:
    print('Average Spearman: {:.3f}'.format(np.mean(gen_res)))
    print('\n')
    print('Sequence \t Spearman')
    print('-------------------------')
    for i in range(len(seqs)):
        print(seqs[i], '\t{:.3f}'.format(gen_res[i]))
