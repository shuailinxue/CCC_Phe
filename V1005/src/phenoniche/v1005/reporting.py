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
    display(Markdown('Directed tensor → sender/receiver/LR mode projections → LayerNorm/MLP/Softmax → simplex Z. Decoder: H normalized to sum one on every decode. Fixed observed row mass retains CCC intensity; no learned scale. Low-rank penalty is disabled. Composition NMF is independent; fusion acts only on normalized graphs.'))
    for ds in ['hbc1','xenium5k']:
        display(Markdown(f'**{ds} frozen configuration**'));display(pd.Series(json.loads((folder(ds)/'config.json').read_text())).to_frame('value'))

def training(dataset):
    table(folder(dataset)/'baseline_metrics.csv')
    if (folder(dataset)/'representation_diagnostics.csv').exists():table(folder(dataset)/'representation_diagnostics.csv')
    checkpoints=[]
    for p in sorted(folder(dataset).glob('seed_*_training.json')):
        checkpoints.append({'model':p.stem,**json.loads(p.read_text())})
    display(pd.DataFrame(checkpoints))
    for tag in ['no_prior','tensor']:
        t=pd.read_csv(folder(dataset)/f'seed_40700_{tag}_loss.csv')
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for col in ['total','reconstruction','validation_reconstruction']:
            axes[0,0].plot(t.epoch,t[col],label=col)
        axes[0,0].set(title=tag+' · training / validation',ylabel='Density-scaled Huber loss');axes[0,0].legend(fontsize=7)
        axes[0,1].plot(t.epoch,t.neighbor_distance,label='G0 neighbors');axes[0,1].plot(t.epoch,t.random_distance,label='Random pairs');axes[0,1].set_title('Latent squared distance');axes[0,1].legend()
        axes[0,2].plot(t.epoch,t.Z_entropy,label='Mean Z entropy');axes[0,2].plot(t.epoch,t.effective_active_programs,label='Effective active programs');axes[0,2].set_title('Program usage');axes[0,2].legend()
        variance=[c for c in t if c.startswith('Z_variance_')]
        im=axes[1,0].imshow(np.log10(np.maximum(t[variance].to_numpy().T,1e-15)),aspect='auto',origin='lower',extent=[.5,len(t)+.5,.5,len(variance)+.5],cmap='viridis')
        axes[1,0].set(title='Per-dimension Z variance',ylabel='Program');fig.colorbar(im,ax=axes[1,0],label='log10 variance')
        axes[1,1].plot(t.epoch,t.singular_max,label='Maximum');axes[1,1].plot(t.epoch,t.singular_min,label='Minimum');axes[1,1].set(title='Sampled Z singular values',yscale='symlog');axes[1,1].legend()
        axes[1,2].plot(t.epoch,t.near_constant_fraction,label='Near-constant fraction');axes[1,2].plot(t.epoch,t.graph_weight,label='Graph weight');axes[1,2].set_title('Collapse and graph ramp');axes[1,2].legend()
        for ax in axes.ravel():ax.set_xlabel('Epoch');ax.axvline(15.5,color='grey',ls=':',lw=.8)
        fig.tight_layout();save(fig,dataset,tag+'_training_diagnostics')
    display(Markdown('Near-constant means variance < 1e-8 on the fixed 4,096-anchor audit sample. Validation chooses the best checkpoint; separate test entries are never used for early stopping.'))

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
    table(folder(dataset)/'tensor_program_summary.csv')
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

def graph_selection(dataset):
    out=folder(dataset)/'graph_selection';summary=json.loads((out/'summary.json').read_text())
    display(Markdown(f'### {dataset} · normalized three-graph niche construction'))
    display(pd.read_csv(out/'graph_statistics.csv'))
    sweep=pd.read_csv(out/'resolution_sweep.csv');beta=pd.read_csv(out/'beta_sensitivity.csv')
    display(Markdown('**Beta sensitivity at the automatically selected resolution**'));display(beta)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(sweep.resolution,sweep.n_niches,'o-');axes[0,0].axhspan(5,25,color='#4daf4a',alpha=.12)
    axes[0,0].set(xscale='log',xlabel='Leiden resolution',ylabel='n niches',title='Resolution vs niche count')
    axes[0,1].plot(sweep.resolution,sweep.spatial_agreement,'o-',color='#377eb8')
    axes[0,1].set(xscale='log',xlabel='Leiden resolution',ylabel='Spatial agreement',title='Resolution vs local agreement')
    axes[1,0].plot(sweep.resolution,sweep.ARI_previous,'o-',label='ARI to previous')
    axes[1,0].plot(sweep.resolution,sweep.NMI_previous,'o-',label='NMI to previous')
    axes[1,0].set(xscale='log',xlabel='Leiden resolution',ylabel='Score',title='Adjacent-resolution stability');axes[1,0].legend()
    if summary['selected']:
        labels=np.load(out/'selected_labels.npy');sizes=np.bincount(labels);sizes=sizes[sizes>0]
        axes[1,1].hist(sizes,bins=min(50,len(sizes)),color='#984ea3');axes[1,1].set(xlabel='Niche size',ylabel='Count',title='Selected niche size distribution')
        chosen=float(summary['resolution'])
        for ax in axes.ravel()[:3]:ax.axvline(chosen,color='#e41a1c',ls='--',lw=1)
    else:axes[1,1].text(.5,.5,'No eligible resolution',ha='center',va='center',transform=axes[1,1].transAxes)
    fig.tight_layout();save(fig,dataset,'three_graph_resolution_diagnostics')
    display(Markdown('**Selection rule:** '+summary['selection_reason']));display(pd.Series(summary).to_frame('value'))
    historical=pd.read_csv(folder(dataset)/'baseline_metrics.csv')
    historical=historical[(historical.seed==40700)&(historical.method=='Proposed')].iloc[0]
    display(Markdown('**Before vs after graph construction (same seed 40700)**'))
    display(pd.DataFrame([
        {'result':'Before: CCC + composition, resolution=1','n_niches':historical.n_niches,
         'spatial_agreement':historical.spatial_agreement,'median_niche_size':historical.median_niche_size,
         'tiny_niche_fraction':historical.fraction_cells_niches_lt20},
        {'result':'After: CCC + composition + spatial, selected resolution','n_niches':summary['n_niches'],
         'spatial_agreement':summary['spatial_agreement'],'median_niche_size':summary['median_niche_size'],
         'tiny_niche_fraction':summary['fraction_cells_niches_lt20']}]))
    if summary['selected']:
        xy=context(dataset)['coords'];labels=np.load(out/'selected_labels.npy')
        fig,ax=plt.subplots(figsize=(8,7));points=ax.scatter(xy[:,0],xy[:,1],c=labels,cmap=plt.get_cmap('turbo',int(labels.max())+1),s=.15,linewidths=0,rasterized=True)
        fig.colorbar(points,ax=ax,fraction=.025,pad=.02,label='Selected niche ID (0-based)')
        ax.set_aspect('equal');ax.invert_yaxis();ax.set(xlabel='x (µm)',ylabel='y (µm)',title=f'{dataset} · CCC + 0.2 composition + 0.1 spatial')
        save(fig,dataset,'three_graph_selected_spatial_niches')

def matching():
    out=ROOT/'outputs/cross_dataset';table(out/'matched_programs.csv')
    s=np.load(out/'similarity.npy');fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(s,vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax,label='Cosine on shared retained CCC')
    ax.set(xlabel='Prime 5K program',ylabel='HBC1 program');ax.set_xticks(range(len(s)),range(1,len(s)+1));ax.set_yticks(range(len(s)),range(1,len(s)+1));save(fig,'cross_dataset','program_matching')
    display(Markdown(json.loads((out/'audit.json').read_text())['limitation']))
    display(Markdown('With very few shared retained entries, matching cannot establish biological reproducibility. Low shared-weight fractions or highly redundant programs further limit this comparison.'))
    if (out/'program_coverage.csv').exists():table(out/'program_coverage.csv')

def summary():
    metrics={ds:pd.read_csv(folder(ds)/'baseline_metrics.csv').groupby('method').mean(numeric_only=True) for ds in ['hbc1','xenium5k']}
    stability={ds:pd.read_csv(folder(ds)/'stability.csv').groupby('method').mean(numeric_only=True) for ds in metrics}
    sections=[];collapse=[]
    for ds in metrics:
        d=pd.read_csv(folder(ds)/'representation_diagnostics.csv');d=d[d.model.str.endswith('tensor')]
        maximum=d.near_constant_fraction.max()
        state='仍有大面积近常数维度' if maximum>=.5 else '未见大面积近常数坍缩'
        collapse.append(f'{ds}: {state}；各 seed 的近常数维度比例 {d.near_constant_fraction.min():.1%}–{maximum:.1%}')
    sections.append('1. **Latent collapse：** Z/H 尺度自由度已被约束；'+ '；'.join(collapse)+'。阈值仅作诊断，不能替代空间与重构评估。')
    def comparison(a,b):
        values=[]
        for ds,t in metrics.items():
            x,y=t.loc[a],t.loc[b]
            values.append(f'{ds}: {a}/{b} 的 test MSE={x.heldout_MSE:.5f}/{y.heldout_MSE:.5f}，空间一致率={x.spatial_agreement:.3f}/{y.spatial_agreement:.3f}，seed ARI={stability[ds].loc[a,"ARI"]:.3f}/{stability[ds].loc[b,"ARI"]:.3f}')
        return '；'.join(values)
    warm_checkpoints=[]
    for ds in metrics:
        checkpoints=[json.loads(p.read_text()) for p in folder(ds).glob('seed_*_tensor_training.json')]
        count=sum(c['best_graph_weight']==0 for c in checkpoints)
        warm_checkpoints.append(f'{ds}: {count}/{len(checkpoints)} 个 tensor 最佳 checkpoint 来自未加 graph loss 的 warm-up')
    sections.append('2. **无 prior vs graph prior：** '+comparison('CCC-autoencoder-no-prior','CCC-tensor')+'。'+ '；'.join(warm_checkpoints)+'。这些 warm-up checkpoint 不能作为 graph prior 有效的证据；不把单个指标当作生物学准确率。')
    sections.append('3. **Tensor vs flat：** '+comparison('CCC-tensor','CCC-flat')+'。')
    gains=[]
    for ds,t in metrics.items():
        gains.append(f'{ds}: 弱融合的空间一致率变化 {t.loc["Proposed","spatial_agreement"]-t.loc["CCC-tensor","spatial_agreement"]:+.3f}；ARI 变化 {stability[ds].loc["Proposed","ARI"]-stability[ds].loc["CCC-tensor","ARI"]:+.3f}')
    sections.append('4. **Composition 弱融合：** '+'；'.join(gains)+'。重构与 Z/H 不因图融合改变。')
    per_dataset=[];fragmentation=[]
    for ds,t in metrics.items():
        q=t.loc['Proposed'];r=t.loc['CCC-tensor']
        per_dataset.append(f'{ds}: Proposed 平均 {q.n_niches:.1f} 个 niche，空间一致率 {q.spatial_agreement:.3f}，test MSE {q.heldout_MSE:.5f}')
        fragmentation.append(f'{ds}: tensor 图平均 {r.graph_connected_components:.1f} 个连通分量，小于20 cells 的 niche 所占细胞比例 {r.fraction_cells_niches_lt20:.2%}；融合后 {q.fraction_cells_niches_lt20:.2%}')
    sections.append('5. **各自结果：** '+'；'.join(per_dataset)+'。两个数据不做跨数据集共享验证。')
    sections.append('6. **历史碎片化基线：** '+'；'.join(fragmentation)+'。这些是本轮三图融合与 resolution sweep 之前、resolution=1.0 的已保存结果。')
    failed=[ds for ds,t in metrics.items() if t.loc['CCC-tensor','heldout_MSE']>t.loc['CCC-flat','heldout_MSE'] and t.loc['CCC-tensor','spatial_agreement']<t.loc['CCC-flat','spatial_agreement']]
    sections.append('7. **失败与限制：** '+(('、'.join(failed)+' 的 tensor 在重构和空间一致率上仍不如 CCC-flat，明确记录失败。') if failed else '指标需综合解释，不能仅凭单项改善宣称成功。')+'不继续调参美化结果；所有 NMF 收敛警告与 early-stopping checkpoint 信息保留在训练结果中。')
    for text in sections:display(Markdown(text))
