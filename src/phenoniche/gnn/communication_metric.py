"""Dictionary-aware distance for program mixtures, with separate total activity.

Program-coordinate distance treats similar dictionary rows as unrelated axes.
This embedding instead preserves the distance between reconstructed relative
CCC profiles in the feature-scaled reconstruction space. No condition or cell
type labels are used. The encoder, Z, H and reconstruction remain unchanged.
"""
import numpy as np
from ..simulation.baselines import dictionary_program_projection


def communication_embedding(p,activity,h,feature_scale,fit_rows,activity_weight=.25):
    p=np.asarray(p,np.float32);activity=np.asarray(activity,np.float32)
    if p.ndim!=2 or len(activity)!=len(p) or len(fit_rows)<2:
        raise ValueError('Invalid program/profile fitting dimensions')
    projection=dictionary_program_projection(h,feature_scale)
    profile=p@projection
    center=profile[fit_rows].mean(0)
    scale=max(float(np.sqrt(np.mean(np.sum((profile[fit_rows]-center)**2,axis=1)))),1e-12)
    profile=(profile-center)/scale
    log_activity=np.log1p(activity);activity_center=float(np.median(log_activity[fit_rows]))
    activity_scale=max(float(log_activity[fit_rows].std()),.1)
    values=np.column_stack([profile,activity_weight*(log_activity-activity_center)/activity_scale]).astype(np.float32)
    audit={'metric':'relative CCC reconstruction distance + separate weak total activity',
           'profile_scale':scale,'activity_center':activity_center,'activity_scale':activity_scale,
           'activity_weight':float(activity_weight),'fit_centers':len(fit_rows),
           'condition_labels_used':False,'cell_type_labels_used':False,
           'profile_definition':'P @ H / feature_scale; one global scalar normalization, no per-program whitening'}
    return values,projection,audit
