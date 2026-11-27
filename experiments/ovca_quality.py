"""Matched-window network summaries and sender/receiver variability."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from phenoniche.v1.storage import Network

p=argparse.ArgumentParser()
p.add_argument('net',type=Path)
a=p.parse_args()
d=Network(a.net)
metrics=[]
for i in range(len(d.ids)):
    s=d.support[i][...,d.group]
    row,e,v=d.entries(np.array([i]))
    metrics.append([d.mass[i].sum(),(s>0).sum(),len(e),s.sum(),v[:,0].sum(),
                    np.mean(s[s>0]) if s.any() else 0,len(e)/max((s>0).sum(),1)])
frame=pd.DataFrame(metrics,columns=['cells','supported_edges','positive_edges','sender_trials',
                                    'positive_trials','mean_support','positive_edge_fraction'])
frame.insert(0,'id',d.ids)
frame.to_csv(a.net/'quality_centers.tsv',sep='\t',index=False)
pairs=[]
for i,sender in enumerate(d.names):
    for j,receiver in enumerate(d.names):
        per=np.zeros(len(d.ids))
        nonzero=np.zeros(len(d.ids))
        support=d.support[:,i,j,:][:,d.group].sum(axis=1)
        for c in np.flatnonzero(support):
            sl=slice(*d.ptr[c:c+2])
            mask=d.edge[sl]//d.nl==i*d.nt+j
            per[c]=d.stats[sl][mask,3].sum()/support[c]
            nonzero[c]=d.stats[sl][mask,0].sum()/support[c]
        x=per[support>0]
        if len(x):
            pairs.append(dict(sender=sender,receiver=receiver,windows=len(x),mean=x.mean(),
                              sd=x.std(),median=np.median(x),q10=np.quantile(x,.1),q90=np.quantile(x,.9),
                              detected_trial_fraction=nonzero[support>0].mean()))
pd.DataFrame(pairs).to_csv(a.net/'quality_type_pairs.tsv',sep='\t',index=False)
out=dict(centers=len(d.ids),train=int(d.train.sum()),test=int(d.test.sum()),genes=len(d.genes),
         panel_lr=d.nl,positive_lr=len(np.unique(d.edge%d.nl)),trial_positive_fraction=float(
             frame.positive_trials.sum()/frame.sender_trials.sum()),
         quantiles=frame.iloc[:,1:].quantile([.1,.5,.9]).to_dict())
(a.net/'quality.json').write_text(json.dumps(out,indent=2))
print(out)
