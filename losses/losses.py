import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def normal_adaptive_weighting(logps, gamma=0.5, alpha=11.7, normalizer=10):
    """
    Uncertainty-aware adaptive weighting for normal samples (Normal Adaptive Weighting).

    Reweights normal samples based on uncertainty estimation so that hard (low-likelihood)
    normal samples contribute larger gradient signals in contrastive learning.

    Args:
        logps: log-likelihoods, shape (N, ).
        gamma: gamma hyperparameter.
        alpha: alpha hyperparameter.
    """
    logps = logps / normalizer
    mask_larger = logps > -0.2
    mask_lower = logps <= -0.2
    probs = torch.exp(logps)
    fl_weights = alpha * (1 - probs).pow(gamma) * torch.abs(logps)
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_larger] = 1.0 
    weights[mask_lower] = fl_weights[mask_lower]

    return weights


def abnormal_adaptive_weighting(logps, gamma=2, alpha=0.53, normalizer=10):
    """
    Uncertainty-aware adaptive weighting for abnormal samples (Abnormal Adaptive Weighting).

    Reweights abnormal samples based on uncertainty estimation so that ambiguous (high-likelihood)
    abnormal samples receive stronger optimization drive in contrastive learning.

    Args:
        logps: log-likelihoods, shape (N, ).
        gamma: gamma hyperparameter.
        alpha: alpha hyperparameter.
    """
    logps = logps / normalizer
    mask_larger = logps > -1.0
    mask_lower = logps <= -1.0
    probs = torch.exp(logps)
    fl_weights = alpha * (1 + probs).pow(gamma) * (1 / torch.abs(logps))
    weights = fl_weights.new_zeros(fl_weights.shape)
    weights[mask_lower] = 1.0 
    weights[mask_larger] = fl_weights[mask_larger]

    return weights


def compute_uncertainty_aware_boundary(logps, mask, pos_beta=0.05, margin_tau=0.1, normalizer=10):
    """
    Uncertainty-aware decision boundary computation (Uncertainty-Aware Boundary Estimation).

    Quantifies sample uncertainty via spatial feature dispersion, adaptively adjusts
    contrastive boundary and decision threshold to improve discrimination on ambiguous samples.

    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, )
        pos_beta: position hyperparameter: beta
        margin_tau: margin hyperparameter: tau
    """
    normal_logps = logps[mask == 0].detach()
    n_idx = int(((mask == 0).sum() * pos_beta).item())
    sorted_indices = torch.sort(normal_logps)[1]
    
    n_idx = sorted_indices[n_idx]
    b_n = normal_logps[n_idx]  # normal boundary
    b_n = b_n / normalizer

    b_a = b_n - margin_tau  # abnormal boundary

    return b_n, b_a


def compute_adaptive_boundary_contrastive_loss(logps, mask, boundaries, normalizer=10, weights=None):
    """
    Adaptive Boundary Contrastive Loss.

    Performs semi-push-pull contrastive learning on normal and abnormal samples
    based on uncertainty-aware decision boundaries to strengthen boundary discrimination.

    Args:
        logps: log-likelihoods, shape (N, )
        mask: 0 for normal, 1 for abnormal, shape (N, 1)
        boundaries: (normal_boundary, abnormal_boundary)
    """
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    b_a = boundaries[1]
    anomaly_logps = logps[mask == 1]    
    anomaly_logps_inter = anomaly_logps[anomaly_logps >= b_a]
    loss_a = anomaly_logps_inter - b_a

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
        ano_weights = weights[mask == 1][anomaly_logps >= b_a]
        loss_a = loss_a * ano_weights
    
    loss_n = torch.mean(loss_n)
    loss_a = torch.mean(loss_a)

    return loss_n, loss_a


def compute_adaptive_boundary_contrastive_loss_normal(logps, mask, boundaries, normalizer=10, weights=None):
    logps = logps / normalizer
    b_n = boundaries[0]  # normal boundaries
    normal_logps = logps[mask == 0]
    normal_logps_inter = normal_logps[normal_logps <= b_n]
    loss_n = b_n - normal_logps_inter

    if weights is not None:
        nor_weights = weights[mask == 0][normal_logps <= b_n]
        loss_n = loss_n * nor_weights
    
    loss_n = torch.mean(loss_n)

    return loss_n

class DiceLoss(nn.Module):
    """Dice Loss function"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class BCEWithLogitsLoss(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.bce_loss(pred, target)


class UncertaintyBoundary(nn.Module):
    """Uncertainty-Aware Boundary Optimization.

    Core innovation: quantifies sample uncertainty via spatial feature dispersion,
    adaptively adjusts contrastive boundary and decision threshold.
    - Maintains mean and log-variance (learnable parameters) under diagonal covariance approx per class.
    - Boundary defined as class center mean minus k times mean std (uncertainty-scaled boundary).
    - Weights positively correlated with uncertainty (mean std) for adaptive weighting.
    - Supports curriculum boundary scheduling tau(t) to dynamically adjust boundary strictness.
    - Gradient clipping hooks ensure training stability.
    """

    def __init__(self, feature_dim, k=2.0, lambda_w=0.1, margin_tau=0.1, max_grad_clip=1.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.lambda_w = lambda_w
        self.margin_tau = margin_tau
        self.max_grad_clip = float(max_grad_clip)

        # mean and log-variance per class (diagonal covariance)
        self.mu = nn.ParameterDict()
        self.log_var = nn.ParameterDict()

    def _create_category_params(self, key, device=None, dtype=None):
        # create category params on demand and register gradient clipping hook
        dev = device
        if dev is None:
            dev = torch.device('cpu')
        mu_p = nn.Parameter(torch.zeros(self.feature_dim, device=dev, dtype=dtype))
        lv_p = nn.Parameter(torch.zeros(self.feature_dim, device=dev, dtype=dtype))

        # register gradient clipping hook (capture clip value via closure)
        clip_val = self.max_grad_clip
        mu_p.register_hook(lambda g, m=clip_val: torch.clamp(g, -m, m))
        lv_p.register_hook(lambda g, m=clip_val: torch.clamp(g, -m, m))

        self.mu[key] = mu_p
        self.log_var[key] = lv_p

    def forward(self, features, labels, categories):
        """
        Args:
            features: Tensor, shape (N, D)
            labels: Tensor or list, normal=0 or 'normal', abnormal=1 or other
            categories: list/tensor of category ids (per-sample)
        Returns:
            scalar loss: weighted semi-push-pull loss (averaged over samples)
        """
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)

        device = features.device
        dtype = features.dtype

        losses = []

        for i, (feat, label, c) in enumerate(zip(features, labels, categories)):
            key = str(c)
            if key not in self.mu:
                self._create_category_params(key, device=device, dtype=dtype)

            mu_c = self.mu[key]
            log_var_c = self.log_var[key]

            # precision = 1/var (diagonal)
            precision = torch.exp(-log_var_c)

            # approximate log-likelihood (omitting constant terms)
            diff = feat - mu_c
            log_prob = -0.5 * torch.sum(diff * diff * precision + log_var_c)

            # standard deviation and boundary (using scalar boundary for simplicity)
            std = torch.exp(0.5 * log_var_c)
            b_n = torch.mean(mu_c) - self.k * torch.mean(std)
            b_a = b_n - self.margin_tau

            # uncertainty and sample weight (higher variance -> higher weight)
            uncertainty = torch.mean(std)
            weight = 1.0 + self.lambda_w * uncertainty

            # semantics: for normal, expect log_prob >= b_n (otherwise loss = b_n - log_prob)
            #            for anomaly, expect log_prob <= b_a (otherwise loss = log_prob - b_a)
            is_normal = False
            if isinstance(label, str):
                is_normal = (label.lower() == 'normal')
            else:
                # assume 0 means normal
                try:
                    is_normal = (int(label) == 0)
                except Exception:
                    is_normal = False

            if is_normal:
                loss = F.relu(b_n - log_prob)
            else:
                loss = F.relu(log_prob - b_a)

            losses.append(weight * loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        return torch.stack(losses).mean()

    def set_margin_tau(self, tau):
        """Update the margin tau used by this UncertaintyBoundary.

        Accepts numeric `tau` and stores it as a float so training can adapt τ(t).
        """
        try:
            self.margin_tau = float(tau)
        except Exception:
            # fallback: keep previous value if conversion fails
            pass

    @torch.no_grad()
    def update_boundary_momentum(self, batch_features, batch_mask=None, momentum=0.99, key='default'):
        """
        Momentum update for mu and log_var using current batch statistics (normal samples only).

        Args:
            batch_features: Tensor shape (N, D)
            batch_mask: Tensor shape (N,) with 0 for normal, 1 for anomaly (optional)
            momentum: momentum coefficient in [0,1)
            key: category key (string) to update corresponding params
        """
        if not isinstance(batch_features, torch.Tensor):
            batch_features = torch.tensor(batch_features)

        device = batch_features.device
        dtype = batch_features.dtype

        k = str(key)
        if k not in self.mu:
            # create params on-the-fly on same device/dtype
            self._create_category_params(k, device=device, dtype=dtype)

        mu_p = self.mu[k]
        lv_p = self.log_var[k]

        if batch_mask is not None:
            try:
                mask = batch_mask.view(-1)
            except Exception:
                mask = batch_mask
            normal_idx = (mask == 0)
            if normal_idx.numel() == 0 or normal_idx.sum() == 0:
                return
            feats = batch_features[normal_idx]
        else:
            feats = batch_features

        if feats.numel() == 0:
            return

        mu_batch = feats.mean(dim=0)
        var_batch = feats.var(dim=0, unbiased=False) + 1e-6

        # momentum update on the .data to avoid adding to computation graph
        mu_p.data = momentum * mu_p.data.to(device=device, dtype=dtype) + (1.0 - momentum) * mu_batch.to(device=device, dtype=dtype)
        lv_p.data = momentum * lv_p.data.to(device=device, dtype=dtype) + (1.0 - momentum) * torch.log(var_batch.to(device=device, dtype=dtype))