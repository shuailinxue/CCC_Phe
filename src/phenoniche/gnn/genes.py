"""Deterministic LR-complete expression input, with training-variance supplements."""
import numpy as np
from ..ccc.lr_atlas import canonicalize_complex


def select_encoder_genes(genes, features, variance, budget, policy='lr_plus_variance'):
    names=tuple(str(g).upper() for g in genes)
    if len(set(names))!=len(names):raise ValueError('Ambiguous panel gene symbols')
    variance=np.asarray(variance)
    if variance.shape!=(len(names),) or not np.isfinite(variance).all():raise ValueError('Invalid training gene variances')
    required=set()
    for row in features.itertuples():
        required.update(canonicalize_complex(row.ligand));required.update(canonicalize_complex(row.receptor))
    panel=set(names);missing=sorted(required-panel)
    rank=np.argsort(-variance,kind='stable');limit=min(int(budget),len(names))
    if limit<1:raise ValueError('Encoder gene budget must be positive')
    if policy=='variance':indices=rank[:limit]
    elif policy=='lr_plus_variance':
        if missing:raise ValueError('CCC member genes absent from panel: '+', '.join(missing))
        if len(required)>limit:raise ValueError(f'Gene budget {limit} cannot contain {len(required)} required LR members')
        mandatory=np.flatnonzero(np.isin(names,list(required)))
        supplement=rank[~np.isin(rank,mandatory)][:limit-len(mandatory)]
        indices=np.sort(np.concatenate((mandatory,supplement)))
    else:raise ValueError('Unknown encoder gene selection policy')
    selected={names[i] for i in indices}
    covered=sum(set(canonicalize_complex(r.ligand)+canonicalize_complex(r.receptor))<=selected for r in features.itertuples())
    audit={'policy':policy,'variance_fit':'training physical cells only','requested_genes':int(budget),
           'selected_genes':len(indices),'required_LR_members':len(required),'LR_members_in_panel':len(required&panel),
           'LR_members_in_encoder':len(required&selected),'missing_panel_members':missing,
           'missing_encoder_members':sorted(required-selected),'CCC_features_fully_covered':covered,
           'CCC_features_total':len(features),'complex_members_included':True}
    return indices,audit
