"""
Dataset handler for LINCS dataset.
"""

import pandas as pd
import torch
import numpy as np
import random
import json
from typing import List
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
from config import LINCS2D_DATA_LOC
from evaluation.initialize_experiment import initialize_random_seed


TOTAL_N = 2687  # this is the original length of the dataframe (before processing)
EXPR = np.loadtxt(LINCS2D_DATA_LOC + 'expr.txt')  # only do this once
EXPR = EXPR.reshape(TOTAL_N, -1, EXPR.shape[-1])

TASK_NAMING = {
    'SEEN': 'full_dataset/',
    'ALL_MISSING': 'complete_generation/',
    'COMBO_DOSE_MISSING': 'combination_and_dose/',
    'COMBO_TIME_MISSING': 'combination_and_time/',
}


FILE_NAMING = {
    'df': 'base_dataframe.csv',
    'unseen_df': 'unseen_dataframe.csv',
    'train_order': 'train_rowordering.txt',
    'val_order': 'val_rowordering.txt',
    'test_order': 'test_rowordering.txt',
    'ucombos': 'unseen_combos.json',
    'udosages': 'unseen_dosages.json',
    'utimes': 'unseen_times.json',
    'gen_targets': 'gen_target_idx',
    'gen_sources': 'gen_source_idx'
}


def load_values_for_task(taskname):
    """
    Load the pre-processed data for the given task.
    
    Parameters:
        taskname (str): SEEN (for full dataset), ALL_MISSING (for complete
            generation), COMBO_DOSE_MISSING (for combination & dose task), or
            COMBO_TIME_MISSING (for combination & time task)
    """
    assert taskname in TASK_NAMING, "Expected one of {}".format(TASK_NAMING.keys())
    dir = LINCS2D_DATA_LOC + '{}/'.format(TASK_NAMING[taskname])

    train_order = np.loadtxt(dir + FILE_NAMING['train_order']).astype(int)
    val_order = np.loadtxt(dir + FILE_NAMING['val_order']).astype(int)
    test_order = np.loadtxt(dir + FILE_NAMING['test_order']).astype(int)
    
    base_df = pd.read_csv(dir + FILE_NAMING['df'])
    base_df['dosage_list'] = base_df['dosage_list'].apply(
        lambda row: [float(r) for r in row[1:-1].split(', ')])
    base_df['treatment_time_list'] = base_df['treatment_time_list'].apply(
        lambda row: [float(r) for r in row[1:-1].split(', ')])

    if taskname == 'SEEN':  # nothing unseen to load!
        return base_df, train_order, val_order, test_order

    # otherwise, load the hidden details
    unseen_df = pd.read_csv(dir + FILE_NAMING['unseen_df'])
    unseen_df['dosage_list'] = unseen_df['dosage_list'].apply(
        lambda row: [float(r) for r in row[1:-1].split()])
    unseen_df['treatment_time_list'] = unseen_df['treatment_time_list'].apply(
        lambda row: [float(r) for r in row[1:-1].split()])

    with open(dir + FILE_NAMING['ucombos'], 'r') as f:
        unseen_combos = json.load(f)
    with open(dir + FILE_NAMING['udosages'], 'r') as f:
        unseen_dosages = json.load(f)
    with open(dir + FILE_NAMING['utimes'], 'r') as f:
        unseen_times = json.load(f)

    # load the sources/targets for gen dataset
    source_base = dir + FILE_NAMING['gen_sources']
    target_base = dir + FILE_NAMING['gen_targets']
    sources = [torch.load(source_base + '{}.txt'.format(i)) for i in range(6)]
    targets = [torch.load(target_base + '{}.txt'.format(i)) for i in range(6)]
    
    return base_df, unseen_df, train_order, val_order, test_order, unseen_combos, unseen_dosages, unseen_times, \
        sources, targets


class Lincs2dDataset(Dataset):
    """
    Dataset object for basic LINCS dataset.
    """
    def __init__(self, device: str, df: pd.DataFrame, ordering: np.array, 
                 use_onehot: bool = True):
        self.df = df
        self.expr = torch.tensor(EXPR)

        self.drugs = sorted(np.unique(self.df['drug_id']))
        self.cells = sorted(np.unique(self.df['cell_id']))

        self.drug_id_to_idx_mapping = {drug: i for i, drug in enumerate(self.drugs)}
        self.cell_id_to_idx_mapping = {cell: i for i, cell in enumerate(self.cells)}

        self.row_ord = ordering

        self.d_counts = {
            d: len(self.df[self.df['drug_id'] == d]) for d in 
            self.df['drug_id'][self.row_ord]}
        self.c_counts = {
            c: len(self.df[self.df['cell_id'] == c]) for c in 
            self.df['cell_id'][self.row_ord]}

        self.N = len(self.row_ord)
        self.device = device
        self.max_unique_cont = self.expr.shape[1]
        self.use_onehot = use_onehot

        self.max_dosage = 20
        self.max_time = 120

    def get_drugs(self):
        return self.drugs

    def get_cells(self):
        return self.cells

    def num_drugs(self):
        return len(self.drugs)

    def num_cells(self):
        return len(self.cells)

    def include_norm_dosage(self, normD):
        self.max_dosage /= normD
        self.df['dosage_list'] = self.df['dosage_list'].apply(
            lambda dsg_arr: [d / normD for d in dsg_arr])

    def include_norm_time(self, normT):
        self.max_time /= normT
        self.df['treatment_time_list'] = self.df['treatment_time_list'].apply(
            lambda time_arr: [t / normT for t in time_arr])

    def get_drug_encoding(self, drug_id):
        if self.use_onehot:
            enc = torch.zeros(len(self.drugs)).to(self.device)
            enc[self.drug_id_to_idx_mapping[drug_id]] = 1
        else:  # return the long tensor of the index only
            enc = torch.tensor([self.drug_id_to_idx_mapping[drug_id]]).to(
                self.device).long()
        return enc

    def get_cell_encoding(self, cell_id):
        if self.use_onehot:
            enc = torch.zeros(len(self.cells)).to(self.device)
            enc[self.cell_id_to_idx_mapping[cell_id]] = 1
        else:
            enc = torch.tensor([self.cell_id_to_idx_mapping[cell_id]]).to(
                self.device).long()
        return enc

    def compute_transfer_drugs(self, N_task):
        drug_options = sorted([d for d in self.d_counts if self.d_counts[d] > 1])
        choices = np.random.choice(drug_options, size=N_task)
        start_idx = []
        target_idx = []
        for dr in choices:
            options = self.df.loc[self.row_ord][self.df.loc[self.row_ord][
                'drug_id'] == dr]
            so, ta = np.random.choice(options.index, size=2)
            so_ord = self.df.loc[self.row_ord].index.get_loc(so)
            ta_ord = self.df.loc[self.row_ord].index.get_loc(ta)
            start_idx.append(so_ord)
            target_idx.append(ta_ord)
        return [self.__getitem__(sidx) for sidx in start_idx], [self.__getitem__(tidx) for tidx in target_idx]

    def compute_transfer_cells(self, N_task):
        cell_options = sorted([c for c in self.c_counts if self.c_counts[c] > 1])
        choices = np.random.choice(cell_options, size=N_task)
        start_idx = []
        target_idx = []
        for ce in choices:
            options = self.df.loc[self.row_ord][self.df.loc[self.row_ord][
                'cell_id'] == ce]
            so, ta = np.random.choice(options.index, size=2)
            so_ord = self.df.loc[self.row_ord].index.get_loc(so)
            ta_ord = self.df.loc[self.row_ord].index.get_loc(ta)
            start_idx.append(so_ord)
            target_idx.append(ta_ord)
        return [self.__getitem__(sidx) for sidx in start_idx], [self.__getitem__(tidx) for tidx in target_idx]

    def compute_generic_transfers(self, N_task):
        start_idx = []
        target_idx = []
        for i in range(N_task):
            so, ta = np.random.choice(self.row_ord, size=2)
            so_ord = self.df.loc[self.row_ord].index.get_loc(so)
            ta_ord = self.df.loc[self.row_ord].index.get_loc(ta)
            start_idx.append(so_ord)
            target_idx.append(ta_ord)
        return [self.__getitem__(sidx) for sidx in start_idx], \
            [self.__getitem__(tidx) for tidx in target_idx]

    def __getitem__(self, item):
        idx = self.row_ord[item]
        rowOI = self.df.loc[idx]

        dr_onehot = self.get_drug_encoding(rowOI['drug_id']).to(self.device)
        ce_onehot = self.get_cell_encoding(rowOI['cell_id']).to(self.device)
        dsgs = rowOI['dosage_list']
        ttimes = rowOI['treatment_time_list']

        gene_expr = self.expr[idx].to(self.device)

        dsgs = torch.tensor(dsgs).to(self.device)
        ttimes = torch.tensor(ttimes).to(self.device)

        assert len(dsgs) <= self.max_unique_cont
        assert len(ttimes) <= self.max_unique_cont

        mask = torch.zeros(self.max_unique_cont).to(self.device)
        mask[np.arange(0, len(ttimes))] = 1

        ttimes = torch.cat([ttimes, torch.zeros(
            self.max_unique_cont - len(ttimes)).to(self.device)])
        dsgs = torch.cat([dsgs, torch.zeros(
            self.max_unique_cont - len(dsgs)).to(self.device)])

        return gene_expr, dr_onehot, ce_onehot, dsgs, ttimes, mask

    def __len__(self):
        return self.N


class Lincs2dDatasetWithUnseen(Lincs2dDataset):
    """
    Dataset object if there are unseen examples.
    """
    def __init__(self, device: str, df: pd.DataFrame, unseen_df: pd.DataFrame,
                 ordering: np.array, unseen_combos: set, unseen_dosages: set,
                 unseen_times: set, use_onehot: bool = True):
        super().__init__(device, df, ordering, use_onehot)

        self.unseen_df = unseen_df
        self.comb_unseen = unseen_combos
        self.dos_unseen = unseen_dosages
        self.time_unseen = unseen_times

    def get_unseen_combos(self):
        return self.comb_unseen

    def get_unseen_dosages(self):
        return self.dos_unseen

    def get_unseen_times(self):
        return self.time_unseen


class Lincs2dDatasetWithUnseenForCvae(Lincs2dDatasetWithUnseen):
    """
    Dataset object if there are unseen examples with the cVAE (sample-by-sample)
    version.
    """
    def __init__(self, device: str, df: pd.DataFrame, unseen_df: pd.DataFrame,
                 ordering: np.array, unseen_combos: set, unseen_dosages: set,
                 unseen_times: set, use_onehot: bool = True):
        super().__init__(device, df, unseen_df, ordering, unseen_combos,
                         unseen_dosages, unseen_times, use_onehot)

    def __getitem__(self, item):
        expr, dr_onehot, ce_onehot, dosages, times, mask = super().__getitem__(item)
        T = expr.shape[0]
        rep_drug = torch.stack([dr_onehot for _ in range(T)])
        rep_cell = torch.stack([ce_onehot for _ in range(T)])

        return expr, rep_drug, rep_cell, dosages, times, mask


class Lincs2dGenerationDataset(Dataset):
    def __init__(self, sources: List[Tensor], targets: List[Tensor], device: str):
        super().__init__()

        self.device = device
        self.sources = sources
        self.targets = targets

    def __getitem__(self, item):
        return [s[item].to(self.device) for s in self.sources], \
            [t[item].to(self.device) for t in self.targets]

    def __len__(self):
        return len(self.sources[0])


class Lincs2dGenerationDatasetCvae(Lincs2dGenerationDataset):
    def __init__(self, sources: List[Tensor], targets: List[Tensor], device: str,
                 tr_dl: Lincs2dDatasetWithUnseenForCvae):
        super().__init__(sources, targets, device)
        self.tr_dl = tr_dl
        
    def make_onehot(self, dr_idx, ce_idx):
        dr_onehot = torch.zeros(self.tr_dl.num_drugs()).to(self.device)
        dr_onehot[dr_idx] = 1
        ce_onehot = torch.zeros(self.tr_dl.num_cells()).to(self.device)
        ce_onehot[ce_idx] = 1
        return dr_onehot, ce_onehot

    def __getitem__(self, item):
        sources, targets = super().__getitem__(item)
        # expand sources for drug, cell
        sexpr, sdr, sce, sdsg, stime, smask = sources
        texpr, tdr, tce, tdsg, ttime, tmask = targets
        
        # convert drugs and cells to onehot!
        s_dronehot, s_ceonehot = self.make_onehot(sdr, sce)
        t_dronehot, t_ceonehot = self.make_onehot(tdr, tce)
        
        T, M = sexpr.shape
        rep_sdr = torch.stack([s_dronehot for _ in range(T)])
        rep_sce = torch.stack([s_ceonehot for _ in range(T)])
        rep_tdr = torch.stack([t_dronehot for _ in range(T)])
        rep_tce = torch.stack([t_ceonehot for _ in range(T)])

        # make smask only have same number of elements as tmask
        while torch.count_nonzero(smask) < torch.count_nonzero(tmask):
            # double up some
            seen_versions = torch.where(smask == 1)
            Tseen = torch.count_nonzero(smask).item()
            selected_expr = torch.masked_select(sexpr, smask.bool().view(
                -1, 1)).view(-1, M)
            selected_drug = torch.masked_select(sdr, smask.bool())
            selected_cell = torch.masked_select(sce, smask.bool())
            selected_dsg = torch.masked_select(sdsg, smask.bool())
            selected_time = torch.masked_select(stime, smask.bool())
            sexpr = torch.cat(
                [selected_expr, selected_expr, torch.zeros(
                    T - Tseen * 2, M).to(self.device)])
            sdr = torch.cat([selected_drug, selected_drug, torch.zeros(
                T - Tseen * 2).to(self.device)])
            sce = torch.cat([selected_cell, selected_cell, torch.zeros(
                T - Tseen * 2).to(self.device)])
            sdsg = torch.cat([selected_dsg, selected_dsg, torch.zeros(
                T - Tseen * 2).to(self.device)])
            stime = torch.cat([selected_time, selected_time, torch.zeros(
                T - Tseen * 2).to(self.device)])
            smask = torch.cat([torch.ones(Tseen*2), torch.zeros(
                T - Tseen * 2)]).to(self.device)

        assert torch.count_nonzero(smask) >= torch.count_nonzero(tmask), \
            "Issue here with counts {} versus {}".format(
            torch.count_nonzero(smask), torch.count_nonzero(tmask))
        new_smask = torch.tensor([
            smask[j] if torch.count_nonzero(smask[:j + 1]) <= torch.count_nonzero(
                tmask) else 0 for j in range(T)]
        ).to(self.device)
        assert torch.count_nonzero(new_smask) == torch.count_nonzero(tmask), \
            "Had {} versus {}".format(torch.count_nonzero(new_smask),
                                      torch.count_nonzero(tmask))
        return ([sexpr, rep_sdr, rep_sce, sdsg, stime, new_smask],
                [texpr, rep_tdr, rep_tce, tdsg, ttime, tmask])


class Lincs2dDataLoader():
    """
    Data loader for basic LINCS dataset.
    """
    def __init__(self, seed: int, device: str, use_onehot: bool, 
                 create_datasets: bool = True, batch_size: int = 1024):
        """
        Parameters:
            seed (int): seed to initialize dataset
            device (str): device to place data on
            use_onehot (bool): True iff we should use a one-hot encoding for the
                drug/cell line
            create_datasets (bool): True iff we still need to load the datasets
            batch_size (int): batch size to use
        """
        if create_datasets:
            initialize_random_seed(seed)
            df, tr_order, val_order, test_order = load_values_for_task('SEEN')
            self.train_dataset = Lincs2dDataset(device, df.copy(), tr_order,
                                                use_onehot)
            self.val_dataset = Lincs2dDataset(device, df.copy(), val_order,
                                              use_onehot)
            self.test_dataset = Lincs2dDataset(device, df.copy(), test_order,
                                               use_onehot)

            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            self.sources = [torch.tensor([]) for _ in range(6)]
            self.targets = [torch.tensor([]) for _ in range(6)]
        self.batch_size = batch_size

    def get_drug_list(self):
        return self.drugs_to_use

    def get_cell_list(self):
        return self.cells_to_use

    def get_feature_dim(self):
        return self.train_dataset.expr.shape[-1]

    def get_max_dosage(self):
        return self.train_dataset.max_dosage

    def get_max_time(self):
        return self.train_dataset.max_time

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
        
    def get_unseen_combos(self):
        return set()

    def get_unseen_dosages(self):
        return set()

    def get_unseen_times(self):
        return set()

    def get_train_dataset(self):
        return self.train_dataset

    def get_validation_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_data_loader(self, dataset_split):
        if dataset_split == 'train':
            ds = self.get_train_dataset()
        elif dataset_split == 'val':
            ds = self.get_validation_dataset()
        else:
            assert dataset_split == 'test', \
                "Unrecognized dataset split {}".format(dataset_split)
            ds = self.get_test_dataset()

        # for dataloader determinism:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(ds, batch_size=int(self.batch_size / 2), shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)

    def get_transfer_data_loader(self, N_drug, N_cell, N_random):

        def stack_example(tens_list):
            out = []
            for i in range(len(tens_list[0])):  # for each type of output
                stacked = torch.stack([tens_list[j][i] for j in 
                                       range(len(tens_list))])
                out.append(stacked)
            return out

        tr_source_drug, tr_target_drug = self.train_dataset.compute_transfer_drugs(N_drug)
        tr_source_cell, tr_target_cell = self.train_dataset.compute_transfer_cells(N_cell)
        tr_source_rand, tr_target_rand = self.train_dataset.compute_generic_transfers(N_random)

        tr_source_drug, tr_target_drug, tr_source_cell, tr_target_cell, tr_source_rand, tr_target_rand = [
            stack_example(s) for s in
            [tr_source_drug, tr_target_drug, tr_source_cell, tr_target_cell, tr_source_rand, 
             tr_target_rand]]

        sources = [torch.cat([tr_source_drug[i], tr_source_cell[i], tr_source_rand[i]]) for i in
                   range(len(tr_source_drug))]
        targets = [torch.cat([tr_target_drug[i], tr_target_cell[i], tr_target_rand[i]]) for i in
                   range(len(tr_target_drug))]

        # shuffle the ordering
        order = np.random.choice(len(sources[0]), size=len(sources[0]), replace=False)
        sources = [s[order] for s in sources]
        targets = [t[order] for t in targets]

        class TransferDataset(torch.utils.data.Dataset):
            def __init__(self):
                super().__init__()

            def __len__(self):
                return len(sources[0])

            def __getitem__(self, item):
                return [s[item] for s in sources], [t[item] for t in targets]

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(TransferDataset(), batch_size=self.batch_size, shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)

    def get_gen_data_loader(self):
        gen_ds = Lincs2dGenerationDataset(self.sources, self.targets, device=self.train_dataset.device)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(gen_ds, batch_size=10, shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)


class Lincs2dDataLoaderWithUnseen(Lincs2dDataLoader):
    """
    DataLoader for LINCS dataset with hidden perturbations.
    """
    def __init__(self, seed: int, device: str, use_onehot: bool,
                 num_unseen_combos: int, num_unseen_times: int, num_unseen_dosages: int,
                 generate_joint_unseen: bool,
                 generate_combo_dosage_joint_unseen: bool, generate_combo_time_joint_unseen: bool,
                 create_datasets: bool = True, batch_size: int = 1024):
        super().__init__(seed, device, use_onehot, False, batch_size)
        if create_datasets:
            if num_unseen_combos == 0 and num_unseen_dosages == 0 and num_unseen_combos == 0 \
                    and num_unseen_times == 0:
                taskname = 'SEEN'
            elif num_unseen_combos > 0 and generate_joint_unseen:
                taskname = 'ALL_MISSING'
            elif num_unseen_combos > 0 and generate_combo_dosage_joint_unseen:
                taskname = 'COMBO_DOSE_MISSING'
            elif num_unseen_combos > 0 and generate_combo_time_joint_unseen:
                taskname = 'COMBO_TIME_MISSING'
            else:
                assert False, "Unexpected setting."

            initialize_random_seed(seed)
            df, unseen_df, tr_order, val_order, test_order, ucombos, udosages, utimes, srcs, tgts = \
                load_values_for_task(taskname)

            self.train_dataset = Lincs2dDatasetWithUnseen(
                device, df, unseen_df, tr_order, ucombos, udosages, utimes, use_onehot)
            self.val_dataset = Lincs2dDatasetWithUnseen(
                device, df, unseen_df, val_order, ucombos, udosages, utimes, use_onehot)
            self.test_dataset = Lincs2dDatasetWithUnseen(
                device, df, unseen_df, test_order, ucombos, udosages, utimes, use_onehot)

            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            self.unseen_combos = self.train_dataset.get_unseen_combos()
            self.unseen_dosages = self.train_dataset.get_unseen_dosages()
            self.unseen_times = self.train_dataset.get_unseen_times()

            self.sources = srcs
            self.targets = tgts

    def get_unseen_combos(self):
        return self.unseen_combos

    def get_unseen_dosages(self):
        return self.unseen_dosages

    def get_unseen_times(self):
        return self.unseen_times


class Lincs2dDataLoaderWithUnseenForCvae(Lincs2dDataLoaderWithUnseen):
    """
    DataLoader for cVAE version of LINCS dataset with hidden perturbations.
    """
    def __init__(self, seed: int, device: str, use_onehot: bool,
                 num_unseen_combos: int, num_unseen_times: int, num_unseen_dosages: int,
                 generate_joint_unseen: bool,
                 generate_combo_dosage_joint_unseen: bool, generate_combo_time_joint_unseen: bool,
                 create_datasets: bool = True, batch_size: int = 1024):
        super().__init__(seed, device, use_onehot, num_unseen_combos, num_unseen_times,
                         num_unseen_dosages, generate_joint_unseen, generate_dosage_time_joint_unseen,
                         generate_combo_dosage_joint_unseen, generate_combo_time_joint_unseen, False,
                         batch_size)
        if create_datasets:
            if num_unseen_combos == 0 and num_unseen_dosages == 0 and num_unseen_combos == 0 \
                    and num_unseen_times == 0:
                taskname = 'SEEN'
            elif num_unseen_combos > 0 and generate_joint_unseen:
                taskname = 'ALL_MISSING'
            elif num_unseen_combos > 0 and generate_combo_dosage_joint_unseen:
                taskname = 'COMBO_DOSE_MISSING'
            elif num_unseen_combos > 0 and generate_combo_time_joint_unseen:
                taskname = 'COMBO_TIME_MISSING'
            else:
                assert False, 'Unexpected setting.'

            initialize_random_seed(seed)
            df, unseen_df, tr_order, val_order, test_order, ucombos, udosages, utimes, srcs, tgts = \
                load_values_for_task(taskname)

            self.train_dataset = Lincs2dDatasetWithUnseenForCvae(
                device, df, unseen_df, tr_order, ucombos, udosages, utimes, use_onehot)
            self.val_dataset = Lincs2dDatasetWithUnseenForCvae(
                device, df, unseen_df, val_order, ucombos, udosages, utimes, use_onehot)
            self.test_dataset = Lincs2dDatasetWithUnseenForCvae(
                device, df, unseen_df, test_order, ucombos, udosages, utimes, use_onehot)
            self.sources = srcs
            self.targets = tgts

            self.drugs_to_use = self.train_dataset.get_drugs()
            self.cells_to_use = self.train_dataset.get_cells()
            self.unseen_combos = self.train_dataset.get_unseen_combos()
            self.unseen_dosages = self.train_dataset.get_unseen_dosages()
            self.unseen_times = self.train_dataset.get_unseen_times()

    def get_gen_data_loader(self):
        gen_ds = Lincs2dGenerationDatasetCvae(self.sources, self.targets, 
                                              device=self.train_dataset.device,
                                              tr_dl=self.train_dataset)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        return DataLoader(gen_ds, batch_size=10, shuffle=False, num_workers=0,
                          worker_init_fn=seed_worker, generator=g)

