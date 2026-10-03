"""Fixed collapse and graph audits; no clustering parameter selection."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

def latent_audit(z):
    z=np.asarray(z,dtype=np.float64);entropy=-(z*np.log(np.maximum(z,1e-15))).sum(1)
    variance=z.var(0);sv=np.linalg.svd(z,compute_uv=False);mean=z.mean(0)
    result={'Z_entropy':float(entropy.mean()),'effective_active_programs':float(np.exp(-(mean*np.log(np.maximum(mean,1e-15))).sum())),
            'mean_per_cell_effective_programs':float(np.exp(entropy).mean()),'singular_max':float(sv.max()),'singular_min':float(sv.min()),
            'near_constant_fraction':float((variance<1e-8).mean())}
    result.update({f'Z_variance_{i+1}':float(v) for i,v in enumerate(variance)})
    return result

def graph_audit(graph,labels):
    degree=np.diff(graph.tocsr().indptr);sizes=np.bincount(labels);sizes=sizes[sizes>0]
    out={'n_niches':len(sizes),'median_niche_size':float(np.median(sizes)),'min_niche_size':int(sizes.min()),
         'fraction_cells_niches_lt20':float(sizes[sizes<20].sum()/len(labels)),
         'graph_connected_components':int(connected_components(graph,directed=False,return_labels=False)),
         'mean_degree':float(degree.mean())}
    out.update({f'degree_{k}':float(v) for k,v in zip(['min','q25','median','q75','q95','max'],np.quantile(degree,[0,.25,.5,.75,.95,1]))})
    return out

def dictionary_audit(h,features,folder,tag):
    summaries=[]
    for k,row in enumerate(h):
        p=np.maximum(row,1e-30)
        record={'program':k+1,'entropy':float(-(row*np.log(p)).sum()),'effective_features':float(np.exp(-(row*np.log(p)).sum())),
                'fraction_below_1pct_max':float((row<.01*row.max()).mean()),'weight_sum':float(row.sum())}
        for title,cols in [('sender',['sender']),('receiver',['receiver']),('pair',['sender','receiver']),('LR',['ligand','receptor'])]:
            grouped=features[cols].assign(weight=row).groupby(cols).weight.sum().sort_values(ascending=False)
            item=grouped.index[0];record['top_'+title]=str(item);record['top_'+title+'_weight']=float(grouped.iloc[0])
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(folder/f'{tag}_program_summary.csv',index=False)
