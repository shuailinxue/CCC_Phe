"""One explicit count scale for LR edge inputs and directed CCC supervision."""
import numpy as np


def normalize_lr_counts(values, panel_totals, mode='native', target_sum=10000.):
    """Scale unlogged RNA using totals over the FULL measured gene panel.

    Values may contain only LR genes; never recompute the denominator from that
    subset. Complex geometric means and sqrt(L*R) are applied AFTER scaling.
    Native mode exists for reading historical experiment configurations.
    """
    if mode not in ('native','panel_total'):
        raise ValueError('Unknown LR count normalization')
    values=np.asarray(values,dtype=np.float32)
    totals=np.asarray(panel_totals,dtype=np.float32)
    if values.ndim!=2 or totals.shape!=(len(values),):
        raise ValueError('LR expression and full-panel totals must align')
    if not np.isfinite(target_sum) or target_sum<=0:
        raise ValueError('LR normalization target must be positive')
    if not np.isfinite(values).all() or (values<0).any() or not np.isfinite(totals).all() or (totals<0).any():
        raise ValueError('Expected finite nonnegative RNA counts and panel totals')
    if mode=='native':return values
    return values*(float(target_sum)/np.maximum(totals,1))[:,None]


def lr_normalization_contract(mode='native',target_sum=10000.):
    if mode not in ('native','panel_total') or not np.isfinite(target_sum) or target_sum<=0:
        raise ValueError('Invalid LR expression normalization contract')
    return dict(mode=mode,target_sum=float(target_sum),
        denominator='All measured RNA genes per cell; never the LR subset',
        transform='unlogged counts; normalization precedes complex geometric mean and LR sqrt product')
