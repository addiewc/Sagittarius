"""
Sagittarius model class.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from typing import *


class Sagittarius(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, class_sizes: List[int],
                 latent_dim: int, cvae_yemb_dims: List[int], cvae_hidden_dims: List[int], 
                 temporal_dim: int, cat_dims: List[int], num_heads: int, num_ref_points: int, 
                 minT: int, maxT: int, device: str, tr_dim: int=None, num_cont: int = None,
                 other_temporal_dims: List[int] = None, 
                 other_minT: List[int] = None, other_maxT: List[int] = None,
                 rec_loss: str='mse') -> None:
        """
        Constructor for Sagittarius model object.
        
        Parameters:
            input_dim (int): dimension M of input measurement
            num_classes (int): number of experimental variables C
            class_sizes (List[int]): number of possible experimental variables per
                class; must have length `num_classes`
            latent_dim (int): dimension d of the encoder's latent space
            cvae_yemb_dims (List[int]): size of experimental variable embedding in 
                the encoder for each of the C experimental variables; must have
                length `num_classes`
            cvae_hidden_dims (List[int]): hidden layer dimensions of encoder MLP
            temporal_dim (int): dimension V of transformer's time embedding for the
                0th temporal variable
            cat_dims (List[int]): size of experimental variable embedding in the
                transformer encoder/decoder for each of the C experimental 
                variables; must have length `num_classes`
            num_heads (int): number of heads H for transformer's multi-head
                attention
            num_ref_points (int): number of regularly-spaced reference points S+1 to
                use in the transformer's reference space
            minT (int): minimum time theta_0^0 to anchor the reference space for the
                0th temporal variable
            maxT (int): maximum time theta_0^1 to anchor the reference space for the
                0th temporal variable
            tr_dim (int): latent dimension to use in the transformer's reference
                space; if None, use latent_dim
            num_cont (int): number of continuous "temporal" variables B to use; if
                None, use 1 temporal variable
            other_temporal_dims (List[int]): dimension V_b of transformer's time
                embedding for the remaining temporal variables; must have length
                `num_cont` - 1; if None, must have `num_cont` = 0
            other_minT (List[int]): minimum time theta_b^0 to anchor the reference
                space for the b>0 th temporal variable; must have length 
                `num_cont` - 1; if None, must have `num_cont` = 0
            other_maxT (List[int]): maximum time theta_b^0 to anchor the reference
                space for the b>0 th temporal variable; must have length 
                `num_cont` - 1; if None, must have `num_cont` = 0
            rec_loss (str): type of reconstruction loss to use; by default, use
                'mse' for mean-squared error; 'bce' for binary cross entropy is 
                alternative option
        """
        super().__init__()
        self.M = input_dim
        self.K = num_classes
        self.ld = latent_dim
        self.tr_dim = tr_dim if tr_dim is not None else self.ld
        self.rec_loss = rec_loss
        assert len(class_sizes) == self.K, \
            'Expected {} class sizes, but got {}'.format(self.K, len(class_sizes))
        assert len(cvae_yemb_dims) == self.K, \
            'Expected {} class sizes, but got {}'.format(
                self.K, len(cvae_yemb_dims))
        assert len(cat_dims) == self.K, \
            'Expected {} categorical embedding sizes, but got {}'.format(
                self.K, len(cat_dims))
        self.class_sizes = class_sizes
        self.device = device
        self.num_cont = num_cont
        if num_cont is not None:
            assert len(other_temporal_dims) == num_cont - 1
            assert len(other_minT) == num_cont - 1
            assert len(other_maxT) == num_cont - 1

        self.kq_dim = temporal_dim + np.sum(cat_dims)
        
        self.cvae_emb = nn.ModuleList([
            nn.Embedding(self.class_sizes[k], cvae_yemb_dims[k]).to(self.device)
            for k in range(self.K)])
        self.cvae_enc = cVAE(
            input_size=self.M, latent_dim=self.ld, num_classes=self.K,
            class_sizes=cvae_yemb_dims, hidden_dims=cvae_hidden_dims,
            device=self.device).to(self.device)
        self.cvae_adv = Adversary(  # kept for legacy random seed consistency
            self.ld, self.K, self.class_sizes, [16] if rec_loss == 'mse' else [8], self.device).to(self.device)
        self.latent_adv = Adversary(  # kept for legacy random seed consistency
            self.tr_dim, self.K, self.class_sizes, [16] if rec_loss == 'mse' else [8],
            self.device).to(self.device)
        self.transformer_enc = SagittariusTransformerEncoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims,
            num_heads, num_ref_points, minT, maxT, device, num_cont=num_cont,
            other_temporal_dims=other_temporal_dims, other_minT=other_minT,
            other_maxT=other_maxT)
        self.transformer_dec = SagittariusTransformerDecoder(
            num_classes, class_sizes, latent_dim, tr_dim, temporal_dim, cat_dims,
            num_heads, num_ref_points, minT, maxT, device, num_cont=num_cont,
            other_temporal_dims=other_temporal_dims, other_minT=other_minT,
            other_maxT=other_maxT)

    def forward(self, xs: Tensor,  # N x T x M
                ts: Tensor,  # N x T
                ys: List[Tensor],  # ys[i] has shape N x T x 1
                mask: Tensor,  # N x T
                other_ts: List[Tensor] = None,
                ) -> Tuple[Tensor]:
        """
        Forward-pass of Sagittarius.
        
        Parameters:
            xs (Tensor): input time series measurements, of dimension N x T x M
            ts (Tensor): input time points for measurements, of dimension N x T
            ys (List[Tensor]): input experimental variables per time series; should
                have length `num_classes` with `y[i]` of dimension N x T x 1
            mask (Tensor): input mask for measurements of shape N x T; 
                `mask[i, t]` = 1 indicates that `xs[i, t]` was measured
            other_ts (List[Tensor]): input time points for non-0 temporal variable;
                should have length `num_cont` - 1, with `other_ts[t]` of dimension
                N x T
                
        Returns:
            xhat (Tensor): Reconstructed xs
            mu (Tensor): Mean of gaussian encoder
            logvar (Tensor): Log variance of gaussian encoder
        """
        N, T, M = xs.shape

        if self.num_cont is not None and (self.num_cont > 1 or other_ts is not None):
            assert len(other_ts) == self.num_cont - 1

        # first, get value and generalized embeddings
        cvae_ys = [self.cvae_emb[k](ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(torch.cat([xs, *cvae_ys], dim=-1).view(N*T, -1))
        z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(
            N, T, self.ld)  # N x T x ld
        self.cvae_adv(z_cvae)  # throwaway computation
        tr_embs = self.transformer_enc(
            z_cvae, ys, ts, mask, other_ts)  # N x T x tr_dim
        self.latent_adv(tr_embs)  # throwaway computation
        
        # decode at ts, ys
        z_dec = self.transformer_dec(tr_embs, ys, ts, other_ts)  # N x T x ld
        xhat = self.cvae_enc.decode(
            torch.cat([z_dec, *cvae_ys], dim=-1).view(N*T, -1)).view(N, T, M)
        return xhat, mu, logvar

    def loss_fn(self, x, xhat, mu, logvar, beta=1.0) -> Dict[str, Tensor]:
        """
        Compute loss for Sagittarius.
        
        Parameters:
            x (Tensor): measured expression time series, of dimension N x T
            xhat (Tensor): simulated expression time series, of dimension N x T
            mu (Tensor): mean of gaussian representation in latent space
            logvar (Tensor): log variance of gaussian representation in latent space
            beta (float): weighting parameter for KL divergence loss regularization 
                term
            
        Returns:
            dictionary with tensor values for:
                "loss" (overall model loss)
                "MSE" (mean-squared error loss, or binary cross entropy if
                    alternative loss used)
                "KLD" (KL divergence loss)
        """
        cvae_losses = self.cvae_enc.loss_fn(xhat, x, mu, logvar, beta=beta, rec_loss=self.rec_loss)

        total_loss = cvae_losses['loss']
        loss_dict = {
            'loss': total_loss,
            'MSE': cvae_losses['MSE'],
            'KLD': cvae_losses['KLD']}
        return loss_dict

    def generate(self, xs, old_ts, new_ts, old_ys, new_ys, old_mask,
                 old_other_ts=None, new_other_ts=None, k=1, get_zdec=False):
        """
        Simulate measurements in generation setting.
        
        Parameters:
            xs (Tensor): input measured time series, of shape N x T x M
            old_ts (Tensor): input time measurements, of shape N x T
            new_ts (Tensor): time points to simulate at, of shape N x T
            old_ys (List[Tensor]): input measured experimental variables; should
                have length `num_classes` with `old_ys[i]` of shape N x T x 1
            new_ys (List[Tensor]): experimental variables to simulate at; should
                have length `num_classes` with `old_ys[i]` of shape N x T x 1
            old_mask (Tensor): mask for measurements, of shape N x T, where
                `old_mask[i, t] = 1` indicates that `xs[i, t]` was measured
            old_other_ts (List[Tensor]): measured input time points for non-0
                temporal variable; should have length `num_cont` - 1, with
                `other_ts[t]` of dimension N x T; None if `num_cont` = 1
            new_other_ts (List[Tensor]): time points for non-0 temporal variable to
                simulate at; should have length `num_cont` - 1, with `other_ts[t]`
                of dimension N x T; None if `num_cont` = 1
            k (int): number of samples to take from cVAE encoder's latent space
            get_zdec (bool): True iff Sagittarius should also return the decoded
                representation from the transformer's regular reference space
        
        Returns:
            xgen (Tensor): simulated measurements for
                (`new_ts`, `new_ys`, `new_other_ts`) combination
            mu (Tensor): mean of gaussian distribution
            logvar (Tensor): log variance of gaussian distribution
            z_dec (Tensor) if `get_zdec`: decoded representation from the
                transformer's regular reference space
        """
        N, T, M = xs.shape
        T_new = new_ts.shape[-1]

        # again, embed values; no longer need to consider the adversary!
        old_cvae_ys = [self.cvae_emb[k](old_ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(
            torch.cat([xs, *old_cvae_ys], dim=-1).view(N * T, -1))
        z_cvaes = []
        for _ in range(k):
            z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(
                N, T, self.ld)  # N x T x ld
            z_cvaes.append(z_cvae)
        z_cvae = torch.mean(torch.stack(z_cvaes, dim=0), dim=0)  # N x T x ld
        
        self.cvae_adv(z_cvae)  # throw-away computation
        
        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(
            z_cvae, old_ys, old_ts, old_mask, additional_ts=old_other_ts
        )  # N x T x tr_dim
        
        self.latent_adv(tr_embs) # throw-away computation

        # decode at new ts, new ys
        z_dec = self.transformer_dec(
            tr_embs, new_ys, new_ts, additional_ts=new_other_ts)  # N x T x ld
        new_cvae_ys = [self.cvae_emb[k](new_ys[k]) for k in range(self.K)]
        xgen = self.cvae_enc.decode(
            torch.cat([z_dec, *new_cvae_ys], dim=-1).view(N * T_new, -1)
        ).view(N, T_new, M)
        
        if get_zdec:
            return xgen, mu, logvar, z_dec
        return xgen, mu, logvar
    
    def predict(self, xs: Tensor,  # N x T x M
                ts: Tensor,  # N x T
                ys: List[Tensor],  # ys[i] has shape N x T x 1
                mask: Tensor,  # N x T
                other_ts: List[Tensor] = None,
                k: int=10,
                ) -> None:
        """
        Predict reconstruction given multiple samples from cVAE latent space.
        
        Parameters:
            xs (Tensor): input measured time series, shape N x T x M
            ts (Tensor): input measured time points, shape N x T
            ys (List[Tensor]): list of input measured experimental variables;
                should have length `num_classes` with `ys[i]` shape N x T x 1
            mask (Tensor): input measurement mask, shape N x T, with 
                `mask[i, t] = 1` means `xs[i, t]` was measured 
            other_ts (List[Tensor]): measured time points for non-0 temporal
                variable to simulate at; should have length `num_cont` - 1, with
                `other_ts[t]` of dimension N x T; None if `num_cont` = 1
            k (int): number of samples to take from cVAE encoder's latent space
            
        Returns:
            xhat (Tensor): reconstructed xs
            mu (Tensor): mean of gaussian latent distribution
            logvar (Tensor): log variance of gaussian latent distribution
            
        """
        N, T, M = xs.shape

        if self.num_cont is not None and (
            self.num_cont > 1 or other_ts is not None):
            assert len(other_ts) == self.num_cont - 1

        # first, get value embeddings
        cvae_ys = [self.cvae_emb[k](ys[k]) for k in range(self.K)]
        mu, logvar = self.cvae_enc.encode(torch.cat([xs, *cvae_ys], dim=-1).view(N*T, -1))  # NT x M+sum(cvae_yemb_dims)
        z_cvaes = []
        for _ in range(k):
            z_cvae = self.cvae_enc.reparameterize(mu, logvar).view(N, T, self.ld)  # N x T x ld
            z_cvaes.append(z_cvae)
        z_cvae = torch.mean(torch.stack(z_cvaes, dim=0), dim=0)  # N x T x ld
        # next, use transformer to produce generalized embedding
        tr_embs = self.transformer_enc(z_cvae, ys, ts, mask, other_ts)  # N x T x tr_dim
        # decode at ts, ys
        z_dec = self.transformer_dec(tr_embs, ys, ts, other_ts)  # N x T x ld
        xhat = self.cvae_enc.decode(torch.cat([z_dec, *cvae_ys], dim=-1).view(N*T, -1)).view(N, T, M)

        return xhat, mu, logvar


class Adversary(nn.Module):
    """
    Adversarial class.
    
    Maintained for random-seed reproducibility with initial results (due to model
    random weight initialization) but not used for model training.
    """
    def __init__(self, input_dim: int, num_classes: int, class_sizes: List[int],
                 hidden_dims: List[int], device: str) -> None:
        super().__init__()
        self.inp_dim = input_dim
        self.K = num_classes
        assert len(class_sizes) == self.K, \
            'Expected {} class sizes, but got {}'.format(self.K, len(class_sizes))
        self.class_sizes = class_sizes
        self.device = device

        self.predictors = []
        for k in range(self.K):
            modules = []
            prev_size = self.inp_dim
            for hdim in hidden_dims:
                modules.extend([nn.Linear(prev_size, hdim), nn.ReLU()])
                prev_size = hdim
            modules.extend([nn.Linear(prev_size, self.class_sizes[k]), 
                            nn.Softmax()])
            self.predictors.append(nn.Sequential(*modules).to(self.device))
        self.predictors = nn.ModuleList(self.predictors)
        self.loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, z_unk: Tensor) -> List[Tensor]:
        pred_ys = [pred(z_unk) for pred in self.predictors]
        return pred_ys

    def loss_fn(self, pred_ys: List[Tensor], real_ys: List[Tensor]) -> Tensor:
        assert len(real_ys) == self.K
        for k in range(self.K):
            assert pred_ys[k].shape[-1] == self.class_sizes[k], \
                "Unexpected number of predicted classes: {} instead of {}".format(pred_ys[k].shape, self.class_sizes[k])
            assert len(real_ys[k].shape) == 1, "Expected integer classes! Should have shape (N,), not {}".format(
                real_ys[k].shape)

        N, T, _ = pred_ys[0].shape
        loss = sum(self.loss(pred_ys[k].view(-1, self.class_sizes[k]), torch.stack([real_ys[k] for _ in range(T)], dim=1).view(-1))
                   for k in range(self.K))
        return loss
    
    
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    
    Adopted from https://github.com/reml-lab/mTAN.
    """
    def __init__(self, latent_dim: int, value_dim: int, kq_dim: int, num_heads: int,
                 device: str) -> None:
        """
        Construct multi-head attention module.
        
        Parameters:
            latent_dim (int): latent dimension of transformer
            value_dim (int): dimension of transformer values
            kq_dim (int): dimension of key and query for transformer
            num_heads (int): number of attention heads H to use
            device (str): device for model
        """
        super().__init__()
        self.device = device
        self.kq_dim = kq_dim
        self.num_heads = num_heads
        self.ld = latent_dim
        self.vdim = value_dim

        self.attn_linear = nn.ModuleList([
            nn.Linear(self.kq_dim, self.num_heads * self.kq_dim),
            nn.Linear(self.kq_dim, self.num_heads * self.kq_dim)])
        self.attn_out = nn.Linear(self.ld * num_heads, self.ld)

    def forward(self, value: Tensor, key: Tensor, query: Tensor,
                mask: Tensor = None) -> Tuple[Tensor]:
        """
        Parameters:
            value (Tensor): transformer values, shape N x T x `value_dim`
            key (Tensor): transformer keys, shape Nk x Tk x `kq_dim`
            query (Tensor): transformer queries, shape Nq x Tq x `kq_dim`
            mask (Tensor): mask for keys, shape N x T, `mask[i, t] = 1` indicates
                that `value[i, t]` was measured
        
        Returns:
            out_rep (Tensor): latent representations in the reference space
            p_attn (Tensor): attention paid to each of the keys frome each of the
                queries
        """
        N, T, _ = value.shape  # N x T x value_dim
        N_q, T_q, _ = query.shape
        N_k, T_k, _ = key.shape
        head_queries = self.attn_linear[0](
            query.view(-1, self.kq_dim)).view(N_q, T_q, self.num_heads, self.kq_dim).transpose(2, 1)  # N_q x h x T_q x kq_dim
        head_keys = self.attn_linear[1](
            key.view(-1, self.kq_dim)).view(
            N_k, T_k, self.num_heads, self.kq_dim
        ).transpose(2, 1)  # N x h x T_q x kq_dim
        scores = torch.matmul(head_queries, head_keys.transpose(-2, -1)
                             ) / np.sqrt(self.kq_dim)  # N x h x T_q x T
        scores = scores.unsqueeze(-1).repeat_interleave(
            self.vdim, dim=-1)  # N x h x T_q x T x vdim
        if mask is not None:
            mask = mask.unsqueeze(1)  # N x 1 x T (so same for each head)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = nn.functional.softmax(scores, dim=-2)

        out_rep = torch.sum(p_attn * value.unsqueeze(1).unsqueeze(-3), 
                            -2).transpose(1, 2).contiguous().view(
            N, -1, self.num_heads * self.vdim)
        out_rep = self.attn_out(out_rep)
        return out_rep, p_attn


class SagittariusLayers(nn.Module):
    """
    Main transformer functionality network.
    """
    def __init__(self, num_classes: int, class_sizes: List[int], 
                 input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int,
                 num_ref_points: int, minT: int, maxT: int,
                 device: str, num_cont: int = None, 
                 other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None
                ) -> None:
        """
        Parameters:
            num_classes (int): number of experimental variables C
            class_sizes (List[int]): number of possible values of each experimental
                variable; must have length `num_classes`
            input_latent_dim (int): latent dimension of input values for transformer
            output_latent_dim (int): latent dimension of output values for
                transformer
            temporal_dim (int): embedding dimension V for 0th temporal variable
            cat_dims (List[int]): embedding dimension for experimental variables in
                transformer
            num_heads (int): number of heads H for multi-head attention
            num_ref_points (int): number of regularly-spaced points S+1 to use in
                the reference space
            minT (int): minimum time point theta_0^0 to anchor the reference space
            maxT (int): maximum time point theta_0^1 to anchor the reference space
            device (str): device to put the model on
            num_cont (int): number of continuous "time" variables B or None if 1
                continuous variable
            other_temporal_dims (List[int]): embedding dimension V_b for the bth 
                temporal variable with b>0; must have length `num_cont` - 1 or None
                if `num_cont` = 1
            other_minT (List[int]): minimum time point theta_b^0 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
            other_maxT (List[int]): maximum time point theta_b^1 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
        """
        super().__init__()
        self.K = num_classes
        assert len(class_sizes) == self.K
        self.class_sizes = class_sizes
        self.value_dim = input_latent_dim
        self.ld = output_latent_dim
        self.temp_dims = [temporal_dim] if num_cont is None else \
            [temporal_dim] + other_temporal_dims
        self.kq_dim = int(np.sum(self.temp_dims) + np.sum(cat_dims))
        self.num_heads = num_heads
        self.num_ref_points = num_ref_points
        self.minT = minT
        self.maxT = maxT
        self.device = device
        
        self.class_embedders = nn.ModuleList([
            nn.Embedding(class_sizes[k] + 1, cat_dims[k]).to(self.device) 
            for k in range(self.K)])
        self.ref_ts = torch.linspace(minT, maxT, num_ref_points).to(self.device)
        if num_cont is not None:
            self.other_ref_ts = [torch.linspace(
                other_minT[i], other_maxT[i], num_ref_points).to(self.device)
                                 for i in range(num_cont - 1)]
            self.other_minT = other_minT
            self.other_maxT = other_maxT
        
        self.attn_network = MultiHeadAttention(
            self.ld, self.value_dim, self.kq_dim, self.num_heads, self.device
        ).to(self.device)

    def get_class_embeddings(self, ys: List[Tensor]) -> Tensor:
        """
        Compute transformer embedding for experimental variables.
        
        Parameters:
            ys (List[Tensor]): experimental variables; should have length
                `num_classes`, ys[i] has shape N x T x 1
        
        Returns:
            embeddings for ys, shape N x T x sum(`cat_dims`)
        """
        return torch.cat([
            self.class_embedders[k](ys[k]) for k in range(self.K)], dim=-1)

    def get_time_embeddings(self, ts: List[Tensor]) -> Tensor:
        """
        Compute temporal embedding for continuous variables.
        
        Parameters:
            ts (List[Tensor]): time variables; should have length `num_cont`,
                ts[i] has shape N x T
        
        Returns:
            embedded time points, shape N x T x sum(temp_dims)
        """
        # N, Ti = ts[i].shape
        N = ts[0].shape[0]

        z_ts = [torch.zeros(N, ts[i].shape[1], self.temp_dims[i]
                           ).to(self.device) for i in range(len(ts))]
        position = [48. * ts[i].unsqueeze(2) for i in range(len(ts))
                   ]  # N x T x 1
        div_term = [torch.exp(
            torch.arange(0, self.temp_dims[i], 2) * -(
                np.log(10.0) / self.temp_dims[i])).to(self.device)
                    for i in range(len(ts))]  # self.temp_dim / 2

        for i in range(len(ts)):
            z_ts[i][:, :, 0::2] = torch.sin(position[i] * div_term[i])
            z_ts[i][:, :, 1::2] = torch.cos(position[i] * div_term[i])
        return torch.cat(z_ts, dim=-1)  # N x T x self.temp_dims


class SagittariusTransformerEncoder(SagittariusLayers):
    def __init__(self, num_classes: int, class_sizes: List[int], 
                 input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int,
                 num_ref_points: int, minT: int, maxT: int,
                 device: str, num_cont: int = None,
                 other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None
                ) -> None:
        """
        Constructor for Sagittarius's transformer encoder layers.
        
        Parameters:
            num_classes (int): number of experimental variables C
            class_sizes (List[int]): number of possible values of each experimental
                variable; must have length `num_classes`
            input_latent_dim (int): latent dimension of input values for transformer
            output_latent_dim (int): latent dimension of output values for
                transformer
            temporal_dim (int): embedding dimension V for 0th temporal variable
            cat_dims (List[int]): embedding dimension for experimental variables in
                transformer
            num_heads (int): number of heads H for multi-head attention
            num_ref_points (int): number of regularly-spaced points S+1 to use in
                the reference space
            minT (int): minimum time point theta_0^0 to anchor the reference space
            maxT (int): maximum time point theta_0^1 to anchor the reference space
            device (str): device to put the model on
            num_cont (int): number of continuous "time" variables B or None if 1
                continuous variable
            other_temporal_dims (List[int]): embedding dimension V_b for the bth 
                temporal variable with b>0; must have length `num_cont` - 1 or None
                if `num_cont` = 1
            other_minT (List[int]): minimum time point theta_b^0 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
            other_maxT (List[int]): maximum time point theta_b^1 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
        """
        super().__init__(num_classes, class_sizes, input_latent_dim,
                         output_latent_dim, temporal_dim, cat_dims,
                         num_heads, num_ref_points, minT, maxT, device, num_cont,
                         other_temporal_dims, other_minT, other_maxT)

    def forward(self, z_seq: Tensor, ys: List[Tensor], ts: Tensor, mask: Tensor,
                additional_ts: List[Tensor] = None) -> Tensor:
        """
        Parameters:
            z_seq (Tensor): Embedded sequence measurements, shape N x T x ld
            ys (List[Tensor]): Experimental variables; must have length
                `num_classes`, ys[i] shape N x T x 1
            ts (Tensor): Time point measurements, shape N x T
            mask (Tensor): Observation mask, where `mask[i, t] = 1` indicates that
                `ts[i, t]` was measured
            additional_ts (Tensor): Time point measurements for bth temporal
                variable with b > 0, shape N x T; can be None if `num_cont` = 1
                
        Returns:
            out (Tensor): embedded sequence in regular reference space
        """
        assert len(ys) == self.K
        N, T, d = z_seq.shape  # value to use for transformer
        assert d == self.value_dim

        # build key from data
        if self.K > 0:
            z_ys = self.get_class_embeddings(ys)  # N x T x sum(cat_dims)
        ts_list = [ts] if additional_ts is None else [ts] + additional_ts
        z_ts = self.get_time_embeddings(ts_list)  # N x T x self.temp_dims
        if self.K > 0:
            key = torch.cat([z_ys, z_ts], dim=-1)  # N x T x self.kq_dim
        else:
            key = z_ts

        # build query from reference points
        if self.K > 0:
            z_unk = self.get_class_embeddings(
                [torch.stack([torch.tensor([Dk for _ in range(1)]).to(self.device) 
                              for _ in range(self.num_ref_points)], dim=1)
                 for Dk in self.class_sizes])  # N x T x sum(cat_dims)
        ref_t_list = [self.ref_ts.unsqueeze(0)] if additional_ts is None else \
            [self.ref_ts.unsqueeze(0)] + [self.other_ref_ts[i].unsqueeze(0) 
                                          for i in range(len(additional_ts))]
        z_reft = self.get_time_embeddings(ref_t_list)  # 1 x T x self.temp_dim
        if self.K > 0:
            query = torch.cat([z_unk, z_reft], dim=-1)  # N x T x self.kq_dim
        else:
            query = z_reft

        out, _ = self.attn_network(
            z_seq, key, query, mask=torch.stack([mask for _ in range(self.ld)],
                                                dim=-1))
        return out


class SagittariusTransformerDecoder(SagittariusLayers):
    def __init__(self, num_classes: int, class_sizes: List[int], 
                 input_latent_dim: int, output_latent_dim: int,
                 temporal_dim: int, cat_dims: List[int], num_heads: int,
                 num_ref_points: int, minT: int, maxT: int,
                 device: str, num_cont: int = None, 
                 other_temporal_dims: List[int] = None,
                 other_minT: List[int] = None, other_maxT: List[int] = None
                ) -> None:
        """
        Constructor for Sagittarius's transformer decoder layers.
        
        Parameters:
            num_classes (int): number of experimental variables C
            class_sizes (List[int]): number of possible values of each experimental
                variable; must have length `num_classes`
            input_latent_dim (int): latent dimension of input values for transformer
            output_latent_dim (int): latent dimension of output values for
                transformer
            temporal_dim (int): embedding dimension V for 0th temporal variable
            cat_dims (List[int]): embedding dimension for experimental variables in
                transformer
            num_heads (int): number of heads H for multi-head attention
            num_ref_points (int): number of regularly-spaced points S+1 to use in
                the reference space
            minT (int): minimum time point theta_0^0 to anchor the reference space
            maxT (int): maximum time point theta_0^1 to anchor the reference space
            device (str): device to put the model on
            num_cont (int): number of continuous "time" variables B or None if 1
                continuous variable
            other_temporal_dims (List[int]): embedding dimension V_b for the bth 
                temporal variable with b>0; must have length `num_cont` - 1 or None
                if `num_cont` = 1
            other_minT (List[int]): minimum time point theta_b^0 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
            other_maxT (List[int]): maximum time point theta_b^1 to anchor the
                reference space; must have length `num_cont` - 1; can be None if
                `num_cont` = 1
        """
        super().__init__(num_classes, class_sizes, input_latent_dim,
                         output_latent_dim, temporal_dim, cat_dims,
                         num_heads, num_ref_points, minT, maxT, device, num_cont,
                         other_temporal_dims, other_minT, other_maxT)

    def forward(self, emb_seq: Tensor, ys: List[Tensor], ts: Tensor, 
                additional_ts: List[Tensor] = None) -> Tensor:
        """
        Parameters:
            z_seq (Tensor): Embedded sequence measurements, shape N x T x ld
            ys (List[Tensor]): Experimental variables; must have length
                `num_classes`, ys[i] shape N x T x 1
            ts (Tensor): Time point measurements, shape N x T
            mask (Tensor): Observation mask, where `mask[i, t] = 1` indicates that
                `ts[i, t]` was measured
            additional_ts (Tensor): Time point measurements for bth temporal
                variable with b > 0, shape N x T; can be None if `num_cont` = 1
                
        Returns:
            out (Tensor): embedded sequence in regular reference space
        """
        assert len(ys) == self.K
        N, T, d = emb_seq.shape  # value to use for transformer
        assert d == self.ld

        # build query from data
        if self.K > 0:
            z_ys = self.get_class_embeddings(ys)  # N x T x sum(cat_dims)
        ts_list = [ts] if additional_ts is None else [ts] + additional_ts
        
        z_ts = self.get_time_embeddings(ts_list)  # N x T x self.temp_dims
        if self.K > 0:
            query = torch.cat([z_ys, z_ts], dim=-1)  # N x T x self.kq_dim
        else:
            query = z_ts

        # build key from reference points
        if self.K > 0:
            z_unk = self.get_class_embeddings(
                [torch.tensor([Dk for _ in range(self.num_ref_points)]
                             ).to(self.device) for Dk in 
                 self.class_sizes]).unsqueeze(0)  # 1 x T_ref x sum(cat_dims)
        ref_t_list = [self.ref_ts.unsqueeze(0)] if additional_ts is None else \
            [self.ref_ts.unsqueeze(0)] + [self.other_ref_ts[i].unsqueeze(0) 
                                          for i in range(len(additional_ts))]
        z_reft = self.get_time_embeddings(
            ref_t_list)  # 1 x T_ref x self.temp_dim
        if self.K > 0:
            key = torch.cat([z_unk, z_reft], dim=-1)  # N x T x self.kq_dim
        else:
            key = z_reft

        out, _ = self.attn_network(emb_seq, key, query, mask=None)
        return out
    
    
class cVAE(nn.Module):
    """
    cVAE encoder/decoder network for measurement embedding.
    """
    def __init__(self,
                 input_size: int,
                 latent_dim: int,
                 num_classes: int,
                 class_sizes: List[int],
                 hidden_dims: List[int],
                 device: str = 'cpu') -> None:
        """
        Constructor for cVAE model.
        
        Parameters:
            input_size (int): Dimension of input measurements
            latent_dim (int): Dimension of gaussian latent generative space
            num_classes (int): Number of experimental variables C
            class_sizes (List[int]): Number of possible values for each experimental
                variable; must have length `num_classes`
            hidden_dims (List[int]): Dimension of each hidden layer in the symmetric
                MLPs for encoder/decoder; len(`hidden_dims`) indicates the number of
                hidden layers
            device (str): Device for the model to run on
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.class_sizes = class_sizes
        self.latent_dim = latent_dim
        self.device = device
        self.hidden_dims = hidden_dims  # need this for ODE interaction!

        prev_d = input_size + int(np.sum(class_sizes))
        enc_layers = []
        for i in range(len(hidden_dims)):
            enc_layers.append(nn.Sequential(
                nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            prev_d = hidden_dims[i]
        self.encoder = nn.Sequential(*enc_layers)
        self.mu_fc = nn.Linear(prev_d, latent_dim)
        self.var_fc = nn.Linear(prev_d, latent_dim)

        prev_d = latent_dim + int(np.sum(class_sizes))
        dec_layers = []
        for i in range(len(hidden_dims)-1, -1, -1):
            dec_layers.append(nn.Sequential(
                nn.Linear(prev_d, hidden_dims[i]), nn.ReLU()))
            prev_d = hidden_dims[i]
        dec_layers.append(nn.Sequential(nn.Linear(prev_d, input_size)))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> List[Tensor]:
        """
        Parameters:
            x (Tensor): input measurement
        
        Returns:
            mu (Tensor): mean of gaussian distribution
            logvar (Tensor): log variance of gaussian distribution
        """
        x = self.encoder(x)
        return self.mu_fc(x), self.var_fc(x)

    def decode(self, x: Tensor) -> Tensor:
        """
        Parameters:
            x (Tensor): point in gaussian latent space
        
        Returns:
            data-space generation from x
        """
        return self.decoder(x)

    def reparameterize(self, mu: Tensor, log_var: Tensor, **kwargs) -> Tensor:
        """
        Parameters:
            mu (Tensor): mean of gaussian distribution
            logvar (Tensor): log variance of gaussian distribution
        
        Returns:
            sample from gaussian distribution parameterized by mu, log_var
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if 'return_eps' in kwargs and kwargs['return_eps']:
            return eps * std + mu, eps
        return eps * std + mu

    def loss_fn(self, pred_y, y, mu, log_var, beta, **kwargs) -> dict:
        """
        Parameters:
            pred_y (Tensor): simulated measurement
            y (Tensor): measurement
            mu (Tensor): mean of gaussian distribution
            log_var (Tensor): log variance of gaussian distribution
            beta (float): weight for KL divergence term
            rec_loss_type (str): 'mse' (or None) for mean squared error loss; 'bce'
                for binary cross entropy loss
            
        Returns:
            dictionary of losses with entries for:
                "loss": overall loss
                "MSE": reconstruction loss
                "KLD": kl divergence loss
        """
        if 'rec_loss' in kwargs:
            rec_loss_type = kwargs['rec_loss']
        else:
            rec_loss_type = 'mse'
        assert rec_loss_type in {'mse', 'bce'}, \
            "Unrecognized loss {}".format(rec_loss_type)

        if rec_loss_type == 'mse':
            recons_y_loss = F.mse_loss(pred_y, y, reduction='mean')
        else:
            M = pred_y.shape[-1]
            # clamp pred_y to between [0, 1]
            pred_y = torch.clamp(pred_y, min=0, max=1)
            recons_y_loss = F.binary_cross_entropy(
                pred_y.view(-1, M), y.view(-1, M), reduction='mean')

        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_y_loss + beta * kld_loss
        return {'loss': loss, 'MSE': recons_y_loss, 'KLD': -kld_loss}
