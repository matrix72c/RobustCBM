import torch

def _pairwise_sq_dists(x):
    # x: [B, d]
    x2 = (x * x).sum(dim=1, keepdim=True)  # [B,1]
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    return dist2.clamp_min(0.0)


def _rbf_kernel(x, sigma=None, eps=1e-12):
    # x: [B, d]
    with torch.no_grad():
        dist2 = _pairwise_sq_dists(x)
        if sigma is None:
            # median trick (avoid zeros on diagonal)
            triu = dist2[torch.triu_indices(dist2.size(0), dist2.size(1), offset=1).unbind()]
            med = torch.median(triu) if triu.numel() > 0 else dist2.mean()
            sigma2 = med + eps
        else:
            sigma2 = sigma ** 2
    K = torch.exp(-_pairwise_sq_dists(x) / (2.0 * sigma2))
    return K


def _linear_kernel(x):
    return x @ x.t()


def _center_gram(K):
    B = K.size(0)
    H = torch.eye(B, device=K.device, dtype=K.dtype) - (1.0 / B)
    return H @ K @ H  # exact centering


def hsic_biased(zc, zv, kernel_c='rbf', kernel_v='rbf', sigma_c=None, sigma_v=None, eps=1e-12):
    r"""
    zc: [B, k], zv: [B, l]
    return scalar HSIC_b \in R_+
    """
    if kernel_c == 'rbf':
        K = _rbf_kernel(zc, sigma_c, eps)
    elif kernel_c == 'linear':
        K = _linear_kernel(zc)
    else:
        raise ValueError('unsupported kernel_c')

    if kernel_v == 'rbf':
        L = _rbf_kernel(zv, sigma_v, eps)
    elif kernel_v == 'linear':
        L = _linear_kernel(zv)
    else:
        raise ValueError('unsupported kernel_v')

    Kc = _center_gram(K)
    Lc = _center_gram(L)
    B = zc.size(0)
    hsic = torch.trace(Kc @ Lc) / ((B - 1.0) ** 2 + eps)
    return hsic


def nhsic(zc, zv, kernel_c='rbf', kernel_v='rbf', sigma_c=None, sigma_v=None, eps=1e-12):
    """
    Normalized HSIC (in the form of CKA). More numerically stable, roughly in [0, 1].
    """
    if kernel_c == 'rbf':
        K = _rbf_kernel(zc, sigma_c, eps)
    elif kernel_c == 'linear':
        K = _linear_kernel(zc)
    else:
        raise ValueError('unsupported kernel_c')

    if kernel_v == 'rbf':
        L = _rbf_kernel(zv, sigma_v, eps)
    elif kernel_v == 'linear':
        L = _linear_kernel(zv)
    else:
        raise ValueError('unsupported kernel_v')

    Kc = _center_gram(K)
    Lc = _center_gram(L)

    num = torch.trace(Kc @ Lc)
    denom = torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc) + eps)
    return num / (denom + eps)


def standardize(z, eps=1e-5):
    return (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + eps)