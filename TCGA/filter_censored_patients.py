import torch
from torch import nn
import numpy as np
import pandas as pd
import sys
import os
from scipy.stats import beta
from lifelines.utils import concordance_index

sys.path.append(os.path.join(sys.path[0], '../'))
from evaluation.initialize_experiment import initialize_random_seed
from config import FILTERING_MASK_LOCATION


class RegressionNetwork(nn.Module):
    def __init__(self, hds):
        super().__init__()
        modules = []
        prev_dim = M
        final_dim = 1
        for hd in hds:
            modules.append(nn.Sequential(nn.Linear(prev_dim, hd), nn.ReLU()))
            prev_dim = hd
        modules.append(nn.Sequential(nn.Linear(prev_dim, final_dim), nn.ReLU()))
        self.layers = nn.Sequential(*modules)
    
    def forward(self, inp):
        return self.layers(inp)
    
    def loss_fn(self, gt, pred, observed, alpha_weight=1.0, reg_lambda=0.0, get_individual_losses=False):
        error = pred - gt
        ae = torch.abs(error)
        abs_error = torch.where(observed == 1.0, ae.float(), alpha_weight*nn.functional.relu(-error))
        
        if get_individual_losses:
            return abs_error

        L2_reg = torch.tensor(0., requires_grad=True)
        for name, param in self.named_parameters():
            if 'weight' in name:
                L2_reg = L2_reg + torch.norm(param)
        
        return torch.mean(abs_error) + reg_lambda * torch.pow(L2_reg, 2)
    
    
def check_concordance(gt, pred, obs):
    return concordance_index(gt.detach().cpu().numpy(), pred.detach().cpu().numpy(), obs.detach().cpu().numpy())


def check_obs_concordance(gt, pred, obs):
    obs_pred = torch.masked_select(pred, obs.bool()).view(-1, 1)
    obs_gt = torch.masked_select(gt, obs.bool()).view(-1, 1)
    return concordance_index(obs_gt.detach().cpu().numpy(), obs_pred.detach().cpu().numpy(), None)


def get_model_for_ct(ct_idx, muts, ts_vec, censoring, mask_vec):
    initialize_random_seed(42)
    
    inp = torch.masked_select(muts[ct_idx], mask_vec[ct_idx].view(-1, 1).bool()).view(-1, 1000).float().to(device)
    gt = torch.masked_select(ts_vec[ct_idx], mask_vec[ct_idx].bool()).unsqueeze(-1).float().to(device)
    obs = torch.masked_select((1 - censoring[ct_idx]), mask_vec[ct_idx].bool()).long().unsqueeze(-1).to(device)
    
    net = RegressionNetwork(hds=[32]).to(device)
    n_epochs = 2500

    sgd = torch.optim.SGD(net.parameters(), lr=1e-1)

    losses = []
    concordances = []
    obs_cc = []
    preds_by_epoch = []
    for ep in range(n_epochs):
        pred = net(inp)
        loss = net.loss_fn(gt, pred, obs, alpha_weight=0.3, reg_lambda=0)
        
        losses.append(loss.item())
        concordances.append(check_concordance(gt, pred, obs))
        obs_cc.append(check_obs_concordance(gt, pred, obs))
        preds_by_epoch.append(pred.detach().cpu().numpy())

        sgd.zero_grad()
        loss.backward()
        sgd.step()
    
    return concordances, obs_cc, preds_by_epoch


def fit_BMMs(obs_ae, cens_ae):
    obs_params = beta.fit(obs_ae)
    pr_obs_obs = beta.pdf(obs_ae, *obs_params)
    pr_cens_obs = beta.pdf(cens_ae, *obs_params)

    cens_params = beta.fit(cens_ae)
    pr_obs_cens = beta.pdf(obs_ae, *cens_params)
    pr_cens_cens = beta.pdf(cens_ae, *cens_params)
    
    return (pr_obs_obs, pr_obs_cens), (pr_cens_obs, pr_cens_cens)


def filter_patients(obs_dists, cens_dists, obs_mask):
    retain = []
    obs_idx = 0
    cens_idx = 0
    for i in range(len(obs_mask)):  # go through each patient
        if obs_mask[i] == 1:
            retain.append(obs_idx)  # include all observed patients
            obs_idx += 1
        else:
            if cens_dists[i][0] > cens_dists[i][1]:  # more likely to be generated from observed-patient beta
                retain.append(cens_idx)
            elif cens_dists[i][0] <= min(obs_dists[i][0]):  # there is a harder observed patient
                retain.append(cens_idx)
            cens_idx += 1
    return retain


def filter_cancer_type_time_series(muts, ts_vec, censoring, mask_vec, ct_vec, ct_mapping,
                                   load_from_file=False, save_to_file=False):
    """
    Filter the cancer patients based on similarity to others wrt survival time.
    
    Parameters:
        muts (Tensor): mutation matrix
        ts_vec (Tensor): survival times
        censoring (Tensor): mask where 1 => patient has censored survival time
        mask_vec (Tensor): measurement mask where 1 => observation is measured
        ct_vec (Tensor): cancer type experimental variable
        ct_mapping (dict[int, str]): mapping from int cancer type variable to cancer type acronym
        load_from_file (bool): True iff we should load the mask from the file
        save_to_file (bool): True iff we should save the mask to file; only used if not `load_from_file`.
    """
    complete_filtering_mask = []
    N, T, M = muts.shape
    
    if load_from_file:
        for i in range(N):
            ct = ct_mapping[ct_vec[i].item()]
            complete_filtering_mask.append(torch.load(FILTERING_MASK_LOCATION + '{}.txt'.format(ct), map_location='cpu'))
        return torch.stack(complete_filtering_mask)
        
    for i in range(N):
        ct = ct_mapping[ct_vec[i].item()]
        
        ccs, obs_ccs, preds_by_ep = get_model_for_ct(i, muts, ts_vec, censoring, mask_vec)
        best_ep = obs_ccs.index(max(obs_ccs))
        
        preds = preds_by_ep[ep]  # best model-predicted survival times
        actual_listed = torch.masked_select(  # given survival/final follow-up times
            ts_vec[idx], mask_vec[idx].bool()).view(-1, 1).detach().cpu().numpy()
        obs = torch.masked_select(  # mask where 1 => observed patient, 0 => censored patient
            (1 - censoring[idx]), mask_vec[idx].bool()).view(-1, 1).detach().cpu().numpy()

        obs_error = [preds[i] - actual_listed[i] for i in range(len(preds)) if obs[i] == 1]
        obs_ae = [abs(err)[0] for err in obs_error]
        
        cens_error = [preds[i] - actual_listed[i] for i in range(len(preds)) if obs[i] == 0]
        cens_ae = [abs(err)[0] for err in cens_error]
        
        obs_dists, cens_dists = fit_BMMs(obs_ae, cens_ae)
        retain = filter_patients(obs_dists, cens_dists, obs)
        
        mask = np.zeros(T)
        mask[retain] = 1
        
        complete_filtering_mask.append(mask)
        
        if save_to_file:
            torch.save(torch.tensor(mask), FILTERING_MASK_LOCATION + '{}.txt'.format(ct))
        
    filtering_mask = np.stack(complete_filtering_mask)
    return torch.tensor(filtering_mask)
