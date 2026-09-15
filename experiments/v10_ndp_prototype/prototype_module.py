"""Single ID class prototypes and fixed training-only score normalization."""
import numpy as np
import torch
import torch.nn.functional as F


class PrototypeAccumulator:
    def __init__(self, num_classes, feature_dim, device):
        self.sums = torch.zeros(num_classes, feature_dim, dtype=torch.float64, device=device)
        self.counts = torch.zeros(num_classes, dtype=torch.int64, device=device)

    def update(self, features, inverse, labels):
        if features.ndim != 2 or features.shape[1] != self.sums.shape[1]:
            raise ValueError('Feature dimension mismatch')
        if len(labels) != len(inverse) or not torch.isfinite(features).all():
            raise ValueError('Nonfinite features or label/point mismatch')
        c = len(self.counts)
        keep = (labels >= 0) & (labels < c)
        # Exact raw-point GT weighting; no majority-vote voxel label shortcut.
        bins = torch.bincount(inverse[keep] * c + labels[keep], minlength=len(features)*c)
        weights = bins.reshape(len(features), c)
        self.sums += weights.T.double() @ features.double()
        self.counts += weights.sum(0)

    def finish(self):
        means = self.sums / self.counts.clamp_min(1)[:, None]
        norms = means.norm(dim=1)
        valid = (self.counts > 0) & (norms > 1e-12)
        if not valid.any() or not torch.isfinite(means).all():
            raise ValueError('No valid prototypes or nonfinite means')
        if ((self.counts > 0) & ~valid).any():
            raise ValueError('Observed class has zero-norm mean')
        p = F.normalize(means, dim=1).float()
        return {'prototypes': p.cpu(), 'valid_mask': valid.cpu(),
                'class_ids': torch.arange(len(valid)), 'class_counts': self.counts.cpu(),
                'norm_before': norms.cpu(), 'norm_after': p.norm(dim=1).cpu(),
                'feature_dim': p.shape[1], 'num_classes': p.shape[0]}


def prototype_scores(features, bank):
    prototypes = bank['prototypes'].to(features.device)
    valid = bank['valid_mask'].to(features.device)
    if not valid.any() or features.shape[1] != prototypes.shape[1]:
        raise ValueError('Invalid prototype bank')
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
        raise ValueError('Nonfinite feature/prototype')
    similarity = F.normalize(features, dim=1) @ prototypes[valid].T
    nearest_sim, indices = similarity.clamp(-1, 1).max(dim=1)
    classes = bank['class_ids'].to(features.device)[valid][indices]
    return 1-nearest_sim, classes, nearest_sim


class Moments:
    """Mergeable population moments, stable even with large biased scores."""
    def __init__(self):
        self.n = 0; self.mean = 0.; self.m2 = 0.
        self.minimum = float('inf'); self.maximum = -float('inf')

    def update(self, values):
        x = np.asarray(values, dtype=np.float64)
        if not len(x):
            return
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite normalization sample')
        n = len(x); mean = float(x.mean()); delta = mean-self.mean
        self.m2 += float(np.square(x-mean).sum()) + delta*delta*self.n*n/(self.n+n)
        self.mean += delta*n/(self.n+n); self.n += n
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))

    def result(self):
        if self.n == 0:
            raise ValueError('No ID normalization samples')
        return {'count': self.n, 'mean': self.mean, 'std': float(np.sqrt(self.m2/self.n)),
                'min': self.minimum, 'max': self.maximum}


def fuse(ndp, proto, statistics, alpha, eps):
    if not 0 <= alpha <= 1 or eps <= 0:
        raise ValueError('Invalid alpha/epsilon')
    n, p = statistics['S_ndp'], statistics['S_proto']
    return ((1-alpha)*(np.asarray(ndp, np.float64)-n['mean'])/(n['std']+eps)
            + alpha*(np.asarray(proto, np.float64)-p['mean'])/(p['std']+eps))
