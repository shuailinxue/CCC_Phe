"""Notebook presentation only; fitting and scientific calculations live in other modules."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,HTML
from .data import ROOT

METHODS=['Composition-only','CCC-flat','CCC-autoencoder-no-prior','CCC-tensor','Proposed']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})

def folder(dataset):return ROOT/'outputs'/dataset

def table(path):
    t=pd.read_csv(path)
    display(HTML('<div style="max-height:450px;overflow:auto">'+t.to_html(index=False,max_rows=None)+'</div>'))
    return t

def context(dataset):return np.load(folder(dataset)/'cache/context.npz')

def save(fig,dataset,name):
    out=folder(dataset)/'figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{name}.png',dpi=300,bbox_inches='tight');display(fig);plt.close(fig)

def audit():
    rows=[]
    for ds in ['hbc1','xenium5k']:
        a=json.loads((folder(ds)/'tensor_audit.json').read_text())
        if (folder(ds)/'input_matrix_qc.json').exists():display(pd.Series(json.loads((folder(ds)/'input_matrix_qc.json').read_text())).to_frame(ds+' input QC'))
        rows.append({k:a.get(k) for k in ['dataset','N','C','L','retained_features','nnz','sparsity','neighbors','sigma','annotation_status']})
        display(Markdown(f"**{ds} original labels → fixed coarse taxonomy.**"))
        display(pd.DataFrame([{'original_label':x,'n_cells':n,'coarse_type':a['taxonomy'][x]} for x,n in a['original_label_counts'].items()]))
    display(pd.DataFrame(rows))

def spatial(dataset,cell_types=False):
    c=context(dataset);xy=c['coords'];a=json.loads((folder(dataset)/'tensor_audit.json').read_text())
    methods=['cell_type'] if cell_types else ['Composition-only','CCC-flat','Proposed']
    fig,axes=plt.subplots(1,len(methods),figsize=(5*len(methods),5),squeeze=False)
    for ax,method in zip(axes.ravel(),methods):
        labels=c['cell_types'] if cell_types else np.load(folder(dataset)/f'seed_40700_{method}_labels.npy')
        palette=plt.get_cmap('turbo',int(labels.max())+1)
        if cell_types:
            from .data import TAXONOMY
            from matplotlib.colors import to_rgba
            canonical=sorted(set(TAXONOMY.values()));base=plt.get_cmap('tab20',len(canonical))
            colors={name:base(i) for i,name in enumerate(canonical)};colors['Unknown']=to_rgba('#bdbdbd')
            palette=lambda i:colors[a['types'][int(i)]]
            plotted_colors=np.array([palette(i) for i in range(len(a['types']))])[labels]
        else:plotted_colors=labels
        points=ax.scatter(xy[:,0],xy[:,1],c=plotted_colors,cmap=None if cell_types else palette,s=.15,linewidths=0,rasterized=True)
        if not cell_types:fig.colorbar(points,ax=ax,fraction=.025,pad=.02,label='Niche ID (0-based)')
        ax.set_aspect('equal');ax.invert_yaxis();ax.set(xlabel='x (µm)',ylabel='y (µm)',title=f'{dataset} · {method}')
        if cell_types:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([],[],marker='o',ls='',color=palette(i),label=t,markersize=4) for i,t in enumerate(a['types'])],bbox_to_anchor=(1.02,1),loc='upper left',frameon=False)
    save(fig,dataset,'cell_types' if cell_types else 'spatial_niches')

def tensor_audit():
    for ds in ['hbc1','xenium5k']:
        a=json.loads((folder(ds)/'tensor_audit.json').read_text())
        display(Markdown(f"**{ds}:** conceptual shape `{a['conceptual_shape']}`; retained `{a['retained_features']:,}` directed entries per anchor; sparse zero fraction `{a['sparsity']:.3f}`. No Top-N input selection."))
        display(pd.read_csv(folder(ds)/'features.csv').head(10))
        display(pd.DataFrame([{k:a[k] for k in ['coverage','pair_support_fraction','pair_support_min','dense_tensor_bytes_avoided']}]))

def config():
    display(Markdown('Directed tensor → sender/receiver/LR mode projections → LayerNorm/MLP/Softplus → nonnegative Z. Decoder: nonnegative retained-edge H. Composition NMF is independent; fusion acts only on normalized graphs.'))
    for ds in ['hbc1','xenium5k']:
        display(Markdown(f'**{ds} frozen configuration**'));display(pd.Series(json.loads((folder(ds)/'config.json').read_text())).to_frame('value'))

def training(dataset):
    display(pd.read_csv(folder(dataset)/'baseline_metrics.csv'))
    if (folder(dataset)/'representation_diagnostics.csv').exists():
        display(pd.read_csv(folder(dataset)/'representation_diagnostics.csv'))
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    for tag,ax in zip(['no_prior','tensor'],axes):
        t=pd.read_csv(folder(dataset)/f'seed_40700_{tag}_loss.csv')
        for col in ['total','reconstruction','lowrank','graph','sparse','group']:
            ax.plot(t.epoch,t[col],label=col)
        ax.set(xlabel='Epoch',ylabel='Unweighted component (total is weighted)',title=tag,yscale='symlog');ax.legend(fontsize=7)
    save(fig,dataset,'loss_curves')

def latent_map(dataset):
    from umap import UMAP
    z=np.load(folder(dataset)/'seed_40700_tensor_Z.npy');labels=np.load(folder(dataset)/'seed_40700_Proposed_labels.npy')
    ids=np.sort(np.random.default_rng(40700).choice(len(z),min(20000,len(z)),replace=False))
    path=folder(dataset)/'cache/umap_visualization.npz'
    if path.exists():p=np.load(path);ids,emb=p['ids'],p['embedding']
    else:
        emb=UMAP(n_neighbors=15,min_dist=.1,random_state=40700,n_jobs=1).fit_transform(z[ids]);np.savez(path,ids=ids,embedding=emb)
    fig,ax=plt.subplots(figsize=(6,4));ax.scatter(emb[:,0],emb[:,1],c=labels[ids],cmap='turbo',s=2,linewidths=0,rasterized=True)
    ax.set(title=f'{dataset} · CCC latent (fixed random display sample, n={len(ids):,})',xlabel='UMAP 1',ylabel='UMAP 2')
    save(fig,dataset,'latent_umap')

def programs(dataset):
    f=pd.read_csv(folder(dataset)/'features.csv');h=np.load(folder(dataset)/'seed_40700_tensor_H.npy')
    pairs=f.sender+' → '+f.receiver;names=sorted(pairs.unique());pair_matrix=np.stack([h[:,pairs==name].sum(1) for name in names],axis=1)
    pd.DataFrame(pair_matrix,index=np.arange(1,len(h)+1),columns=names).to_csv(folder(dataset)/'program_pair_weights.csv',index_label='program')
    normalized=pair_matrix/np.maximum(pair_matrix.sum(1,keepdims=True),1e-12)
    fig,ax=plt.subplots(figsize=(max(10,len(names)*.16),5));im=ax.imshow(normalized,aspect='auto',cmap='magma')
    ax.set_xticks(np.arange(len(names)),names,rotation=90,fontsize=6);ax.set_yticks(np.arange(len(h)),np.arange(1,len(h)+1));ax.set_ylabel('CCC program');ax.set_title(dataset+' · program × sender→receiver');fig.colorbar(im,ax=ax,label='Fraction of program weight')
    save(fig,dataset,'program_pair_heatmap')
    edges=table(folder(dataset)/'tensor_top_edges.csv')
    fig,axes=plt.subplots(4,4,figsize=(20,17))
    for k,ax in enumerate(axes.ravel(),1):
        d=edges[edges.program==k].head(5).iloc[::-1]
        names=d.sender+' → '+d.receiver+' | '+d.ligand+'–'+d.receptor
        ax.barh(np.arange(len(d)),d.weight,color='#267f9b');ax.set_yticks(np.arange(len(d)),names,fontsize=6);ax.set_title(f'Program {k}');ax.set_xlabel('H weight')
    fig.tight_layout();save(fig,dataset,'program_top_lr')
    for name in ['niche_composition','niche_program_activity','niche_top_edges']:
        display(Markdown('**'+name+'**'));table(folder(dataset)/(name+'.csv'))

def baselines():
    results=[]
    for ds in ['hbc1','xenium5k']:
        t=pd.read_csv(folder(ds)/'baseline_metrics.csv');t.insert(0,'dataset',ds);results.append(t)
    t=pd.concat(results);summary=t.groupby(['dataset','method']).agg(n_niches=('n_niches','mean'),spatial_agreement=('spatial_agreement','mean'),heldout_MAE=('heldout_MAE','mean'),heldout_MSE=('heldout_MSE','mean')).reset_index()
    summary.to_csv(ROOT/'outputs/baseline_summary.csv',index=False);display(summary)
    display(Markdown('Composition-only has no CCC decoder, so reconstruction is NA. Held-out metrics use the same fixed 1,024 anchors and held-out entries, including retained zeros. Spatial agreement is descriptive, not niche accuracy.'))

def stability():
    for ds in ['hbc1','xenium5k']:
        fig,ax=plt.subplots(figsize=(6,4))
        t=pd.read_csv(folder(ds)/'stability.csv');display(t.groupby('method')[['ARI','NMI','program_matched_cosine']].agg(['mean','std']))
        ax.boxplot([t.loc[t.method==m,'ARI'] for m in METHODS],tick_labels=METHODS);ax.tick_params(axis='x',rotation=70);ax.set(title=ds,ylabel='Between-seed ARI')
        save(fig,ds,'stability')

def sensitivity():
    for ds in ['hbc1','xenium5k']:display(Markdown(f'**{ds}**'));table(folder(ds)/'alpha_sensitivity.csv')

def matching():
    out=ROOT/'outputs/cross_dataset';table(out/'matched_programs.csv')
    s=np.load(out/'similarity.npy');fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(s,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax,label='Cosine on shared retained CCC')
    ax.set(xlabel='Prime 5K program',ylabel='HBC1 program');ax.set_xticks(range(len(s)),range(1,len(s)+1));ax.set_yticks(range(len(s)),range(1,len(s)+1));save(fig,'cross_dataset','program_matching')
    display(Markdown(json.loads((out/'audit.json').read_text())['limitation']))
    display(Markdown('With very few shared retained entries, matching cannot establish biological reproducibility. Low shared-weight fractions or highly redundant programs further limit this comparison.'))
    if (out/'program_coverage.csv').exists():table(out/'program_coverage.csv')

def summary():
    baselines()
    for ds in ['hbc1','xenium5k']:
        warnings=[line for line in (folder(ds)/'run.log').read_text().splitlines() if 'Warning:' in line]
        if warnings:display(Markdown(f'**{ds} runtime warnings:** '+ '; '.join(sorted(set(warnings)))))
    for ds in ['hbc1','xenium5k']:
        s=pd.read_csv(folder(ds)/'stability.csv').groupby('method')[['ARI','program_matched_cosine']].mean()
        display(Markdown(f"**{ds}:** five seeds completed. Proposed mean ARI = {s.loc['Proposed','ARI']:.3f}; tensor program matched cosine = {s.loc['CCC-tensor','program_matched_cosine']:.3f}."))
    for ds in ['hbc1','xenium5k']:
        d=pd.read_csv(folder(ds)/'representation_diagnostics.csv')
        d=d[d.model.str.endswith('tensor')]
        display(Markdown(f"**{ds} redundancy audit:** median within-model H cosine = {d.H_other_program_cosine_median.median():.3f}; median centered latent first-PC fraction = {d.centered_Z_first_component_variance_fraction.median():.3f}. Tensor latent maximum column std ranges from {d.maximum_latent_column_std.min():.2e} to {d.maximum_latent_column_std.max():.2e}; near-zero variation indicates collapse, not meaningful stability."))
    display(Markdown('Cross-dataset analysis is paused at the user’s request. Results above are within-dataset only. Near-constant latents and redundant programs prevent interpreting high clustering agreement as biological success. Five fixed epochs are not a convergence guarantee.'))
