"""Read-only real-result reporting. No model fitting is called from this module."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from ..storage import OUTPUT_DIR,STORAGE_ROOT

COLORS=['#4477AA','#EE6677','#228833','#CCBB44','#66CCEE','#AA3377','#BBBBBB',
        '#332288','#44AA99','#DDCC77','#882255','#AA4499','#999933','#88CCEE','#CC6677']


def style():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'figure.dpi':115,'savefig.dpi':600,'figure.facecolor':'white','axes.facecolor':'white'})


class GNNReport:
    def __init__(self,output):
        style();self.output=Path(output);self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)
        self.tables=self.output/'report_tables';self.tables.mkdir(exist_ok=True)
        self.config=json.loads((self.output/'config.json').read_text())
        self.summary=json.loads((self.output/'summary.json').read_text())
        self.audit=json.loads((self.output/'audit.json').read_text())
        self.audit.update(self.audit.get('regulation',{}))
        self.prep=np.load(self.output/'cache/preprocessing.npz')
        self.features=pd.read_csv(self.output/'cache/features.csv')
        self.Z=np.load(self.output/'Z.npy',mmap_mode='r');self.P=np.load(self.output/'P.npy',mmap_mode='r');self.H=np.load(self.output/'H.npy')
        self.C_hat=np.load(self.output/'CCC_reconstruction.npy',mmap_mode='r')
        self.history=pd.read_csv(self.output/'training.csv');self.selection=pd.read_csv(self.output/'K_selection.csv')
        self.assignments=pd.read_csv(self.output/'selected_model/niche_assignments.csv.gz')
        self.k=int(self.summary['selected_K']);self.labels=self.assignments.niche.to_numpy()-1
        self.Q=np.load(self.output/'selected_model/Q.npy',mmap_mode='r')
        self.niche_H=np.load(self.output/'selected_model/H.npy')
        self.contribution=pd.read_csv(self.output/'selected_model/program_contribution.csv',index_col=0)
        self.categories=sorted(self.assignments.cell_type.unique())
        self.type_codes=pd.Categorical(self.assignments.cell_type,categories=self.categories).codes
        self.xy=self.assignments[['x','y']].to_numpy(np.float32)
        self.legacy=pd.read_csv(OUTPUT_DIR/'selected_model/niche_assignments.csv.gz')
        if not np.array_equal(self.assignments.cell_id,self.legacy.cell_id): raise ValueError('GNN/AE cell IDs are not aligned')
        self.old_labels=self.legacy.niche.to_numpy()-1
        self.old_k=int(self.legacy.niche.max())
        self.feature_metrics=pd.read_csv(self.output/'reconstruction_feature_metrics.csv')
        self.reconstruction_metrics=json.loads((self.output/'reconstruction_metrics.json').read_text())
        self.sources={}

    def save(self,fig,name):
        fig.savefig(self.figures/(name+'.png'),dpi=600,bbox_inches='tight',facecolor='white')
        fig.savefig(self.figures/(name+'.pdf'),bbox_inches='tight',facecolor='white')
        self.sources[name]=name
        return fig

    def overview(self):
        return pd.Series({'Cells / microenvironments':len(self.Z),'Xenium genes':self.audit['panel_genes'],
          'Fixed directed CCC features':len(self.features),'Effective training features':int(self.prep['feature_mask'].sum()),
          'GNN input genes (excluding activity targets)':len(self.prep['encoder_indices']),
          'Communication programs':self.Z.shape[1],'Selected niches':self.k,'Biological loss mode':self.config['mode'],
          'Upstream prior modules':self.audit['upstream']['modules'],
          'Downstream prior modules':self.audit['downstream']['modules'],
          'Spatial blocks (µm)':self.config['block_um'],'Train centers':int((self.assignments['split']==0).sum()),
          'Validation centers':int((self.assignments['split']==1).sum()),'Test centers':int((self.assignments['split']==2).sum()),
          'Purged boundary centers (mapped after training)':int((self.assignments['split']==-1).sum()),
          'Annotation':'Provisional outcome-blind broad marker labels','Clinical supervision':'None'}).to_frame('Value')

    def training_curves(self):
        fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
        for ax,key,title in zip(axes,('ccc','upstream','downstream'),('CCC reconstruction','Sender regulon alignment','Receiver response alignment')):
            for prefix,color in (('train',COLORS[0]),('validation',COLORS[1])):
                column=prefix+'_'+key
                if column in self.history: ax.plot(self.history.epoch,self.history[column],color=color,lw=1.5,label=prefix)
            ax.set(xlabel='Epoch',ylabel='Masked standardized Huber loss',title=title)
            ax.legend()
        return self.save(fig,'training_curves')

    def reconstruction(self):
        source=np.load(STORAGE_ROOT/'data/ccc/cache/ccc.npy',mmap_mode='r');cols=self.prep['columns']
        test=np.flatnonzero(self.assignments['split'].to_numpy()==2)
        sample=np.sort(np.random.default_rng(40700).choice(test,min(25000,len(test)),replace=False))
        actual=np.asarray(source[sample])[:,cols].sum(1);pred=np.asarray(self.C_hat[sample]).sum(1)
        fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
        image=axes[0].hexbin(np.log1p(actual),np.log1p(pred),gridsize=60,mincnt=1,bins='log',cmap='Blues',linewidths=0)
        low=min(np.log1p(actual).min(),np.log1p(pred).min());high=max(np.log1p(actual).max(),np.log1p(pred).max())
        axes[0].plot([low,high],[low,high],ls='--',lw=.8,c='#555555')
        axes[0].set(xlabel='Observed log(1 + total CCC)',ylabel='Predicted log(1 + total CCC)',title=f'Held-out intensity (n={len(sample):,})')
        fig.colorbar(image,ax=axes[0],label='Cells per hexagon (log scale)')
        valid=self.feature_metrics[self.feature_metrics.effective_mask & self.feature_metrics.test_R2.notna()]
        axes[1].hist(valid.test_R2.clip(-1,1),bins=35,color=COLORS[0],edgecolor='white')
        axes[1].axvline(0,c='#777777',ls='--',lw=.8);axes[1].set(xlabel='CCC feature test R² (clipped at −1)',ylabel='Directed CCC features',title='Feature reconstruction quality')
        splits=['train','validation','test'];values=[self.reconstruction_metrics[x]['scaled_huber'] for x in splits]
        axes[2].bar(splits,values,color=[COLORS[0],COLORS[3],COLORS[1]])
        axes[2].set(ylabel='Masked standardized Huber loss',title='Disjoint spatial-block evaluation')
        pd.DataFrame({'cell_id':self.assignments.cell_id.iloc[sample].to_numpy(),'observed_total_CCC':actual,'predicted_total_CCC':pred}).to_csv(self.tables/'reconstruction_scatter.csv',index=False)
        return self.save(fig,'CCC_reconstruction')

    def program_usage(self):
        mean_p=np.asarray(self.P).mean(0);mean_z=np.asarray(self.Z).mean(0);dominant=np.bincount(np.asarray(self.P).argmax(1),minlength=len(mean_p))/len(self.P)
        frame=pd.DataFrame({'program':np.arange(1,len(mean_p)+1),'mean_mixture':mean_p,'mean_exposure':mean_z,'dominant_fraction':dominant})
        frame.to_csv(self.tables/'program_usage.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(13,3.4),layout='constrained');x=frame.program
        for ax,values,title,ylabel in zip(axes,(mean_p*100,mean_z,dominant*100),('Relative program usage','Absolute CCC program activity','Dominant-program assignments'),('Mean mixture (%)','Mean Z (native CCC units)','Cells (%)')):
            ax.bar(x,values,color=COLORS[0],width=.8);ax.set(xlabel='Program',ylabel=ylabel,title=title);ax.set_xticks(x[::2])
        return self.save(fig,'program_usage'),frame

    def dictionary_heatmap(self):
        chosen=np.unique(np.concatenate([np.argsort(-row,kind='stable')[:3] for row in self.H]))
        chosen=np.asarray(sorted(chosen,key=lambda i:(self.features.iloc[i].sender,self.features.iloc[i].receiver,self.features.iloc[i].ligand)))
        abbr={'Epithelial':'Epi','Fibroblast':'Fib','Endothelial':'Endo','Macrophage':'Mac','Myoepithelial':'Myo','Pericyte':'Peri','Unassigned':'Unk','Mast cell':'Mast','T cell':'T','B cell':'B'}
        label=[f'{abbr.get(self.features.iloc[i].sender,self.features.iloc[i].sender)}→{abbr.get(self.features.iloc[i].receiver,self.features.iloc[i].receiver)} | {self.features.iloc[i].ligand}–{self.features.iloc[i].receptor}' for i in chosen]
        fig,ax=plt.subplots(figsize=(max(10,.19*len(chosen)),8.8),layout='constrained')
        sns.heatmap(np.log1p(self.H[:,chosen]*1000),cmap='Blues',ax=ax,xticklabels=label,yticklabels=[f'P{i:02d}' for i in range(1,len(self.H)+1)],cbar_kws={'label':'log(1 + 1000 × H)'})
        ax.tick_params(axis='x',rotation=90,labelsize=6);ax.tick_params(axis='y',rotation=0,labelsize=8)
        ax.set(xlabel='Representative directed CCC features',ylabel='Communication program',title='Nonnegative CCC dictionary H')
        pd.DataFrame(self.H,index=np.arange(1,len(self.H)+1),columns=self.features.ccc).to_csv(self.tables/'H_full.csv',index_label='program')
        return self.save(fig,'H_dictionary')

    def k_selection(self):
        fig,axes=plt.subplots(1,4,figsize=(13,3.3),layout='constrained')
        for ax,column,title in zip(axes,('silhouette','seed_stability','max_niche_cell_fraction','selection_score'),('Validation silhouette','Clustering seed stability (ARI)','Largest niche fraction','Combined selection score')):
            ax.plot(self.selection.K,self.selection[column],c=COLORS[0],marker='o',ms=3,lw=1.2)
            ax.axvline(self.k,c=COLORS[1],ls='--',lw=1.1);ax.set(xlabel='K',ylabel=column.replace('_',' '),title=title);ax.set_xticks([4,6,8,10,12,15])
        fig.suptitle(f'K fitted on training blocks, selected on validation blocks: K={self.k}',fontsize=12)
        return self.save(fig,'K_selection')

    def _spatial(self,ax,color,title,colors=None):
        if colors is None: colors=COLORS
        ax.scatter(self.xy[:,0],self.xy[:,1],c=np.asarray(colors)[color],s=.16,lw=0,rasterized=True)
        ax.set(title=title,xlabel='x (µm)',ylabel='y (µm)');ax.set_aspect('equal');ax.invert_yaxis()
        # Coordinates retain source µm scale; no cropping/downsampling.
        x0=float(self.xy[:,0].min()+.04*np.ptp(self.xy[:,0]));y0=float(self.xy[:,1].max()-.03*np.ptp(self.xy[:,1]))
        ax.plot([x0,x0+500],[y0,y0],color='#333333',lw=2);ax.text(x0+250,y0-60,'500 µm',ha='center',fontsize=7)

    def spatial_overview(self):
        fig,axes=plt.subplots(1,2,figsize=(12,6.8),layout='constrained')
        self._spatial(axes[0],self.type_codes,'Broad cell-type annotation');self._spatial(axes[1],self.labels,f'GNN communication niches (K={self.k})')
        for ax,names in ((axes[0],self.categories),(axes[1],[f'Niche {i}' for i in range(1,self.k+1)])):
            handles=[mpl.lines.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=4,label=name) for i,name in enumerate(names)]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=4,fontsize=7)
        return self.save(fig,'cell_types_and_niches')

    def each_niche(self):
        rows=int(np.ceil(self.k/4));fig,axes=plt.subplots(rows,4,figsize=(14,3.6*rows),layout='constrained',squeeze=False)
        for i,ax in enumerate(axes.flat):
            if i>=self.k: ax.set_axis_off();continue
            selected=self.labels==i
            ax.scatter(*self.xy.T,c='#E0E0E0',s=.11,lw=0,rasterized=True)
            ax.scatter(*self.xy[selected].T,c=COLORS[i],s=.22,lw=0,rasterized=True)
            ax.set_title(f'Niche {i+1} | n={selected.sum():,}',fontsize=10);ax.set_aspect('equal');ax.invert_yaxis();ax.set_axis_off()
            x0=float(self.xy[:,0].min()+.04*np.ptp(self.xy[:,0]));y0=float(self.xy[:,1].max()-.03*np.ptp(self.xy[:,1]))
            ax.plot([x0,x0+500],[y0,y0],color='#333333',lw=1.5)
            ax.text(x0+250,y0-60,'500 µm',ha='center',fontsize=6)
        fig.suptitle('Spatial distribution of individual GNN niches',fontsize=13)
        return self.save(fig,'each_niche_spatial')

    def composition(self):
        counts=pd.crosstab(self.assignments.niche,self.assignments.cell_type).reindex(index=range(1,self.k+1),columns=self.categories,fill_value=0)
        fraction=counts.div(counts.sum(1),axis=0);background=counts.sum(0)/counts.to_numpy().sum()
        enrichment=np.log2((fraction+1e-6)/(background+1e-6))
        counts.to_csv(self.tables/'niche_celltype_counts.csv');fraction.to_csv(self.tables/'niche_celltype_fraction.csv');enrichment.to_csv(self.tables/'niche_celltype_log2_enrichment.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        bottom=np.zeros(self.k)
        for i,name in enumerate(self.categories):
            axes[0].bar(np.arange(1,self.k+1),fraction[name],bottom=bottom,color=COLORS[i],label=name,width=.82);bottom+=fraction[name].to_numpy()
        axes[0].set(xlabel='Niche',ylabel='Cell fraction',title='Center-cell type composition',ylim=(0,1));axes[0].set_xticks(range(1,self.k+1));axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,fontsize=7)
        sns.heatmap(enrichment,center=0,cmap=sns.diverging_palette(240,30,as_cmap=True),vmin=-4,vmax=4,ax=axes[1],cbar_kws={'label':'log₂ enrichment vs tissue'})
        axes[1].set(xlabel='Cell type',ylabel='Niche',title='Descriptive cell-type enrichment');axes[1].tick_params(axis='x',rotation=60)
        return self.save(fig,'niche_celltype_composition'),fraction,enrichment

    def local_composition(self):
        # Existing frozen neighborhood-composition cache is report-only.
        values=np.load(STORAGE_ROOT/'data/ccc/cache/local_composition.npy',mmap_mode='r')
        manifest=json.loads((STORAGE_ROOT/'data/ccc/cache/input_manifest.json').read_text())
        categories=manifest['cell_types'];mean=np.stack([np.asarray(values[self.labels==i]).mean(0) for i in range(self.k)])
        global_mean=np.asarray(values).mean(0);enrich=np.log2((mean+1e-6)/(global_mean+1e-6))
        fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
        sns.heatmap(enrich,cmap=sns.diverging_palette(240,30,as_cmap=True),center=0,vmin=-3,vmax=3,xticklabels=categories,yticklabels=range(1,self.k+1),ax=ax,cbar_kws={'label':'log₂ local-composition enrichment'})
        ax.set(xlabel='Neighbor cell type',ylabel='Niche',title='Physical-neighborhood composition (report only)');ax.tick_params(axis='x',rotation=55)
        frame=pd.DataFrame(mean,index=range(1,self.k+1),columns=categories);frame.to_csv(self.tables/'niche_local_celltype_fraction.csv')
        return self.save(fig,'niche_local_composition')

    def niche_programs(self):
        hard=np.stack([np.asarray(self.Z[self.labels==i]).mean(0) for i in range(self.k)])
        relative=hard/np.maximum(hard.sum(1,keepdims=True),1e-8)
        frame=pd.DataFrame(hard,index=range(1,self.k+1),columns=[f'P{i:02d}' for i in range(1,self.Z.shape[1]+1)]);frame.to_csv(self.tables/'niche_program_activity.csv')
        fig,axes=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
        for ax,values,title,label in zip(axes,(np.log1p(hard),relative),('Absolute program activity','Relative program composition'),('log(1 + mean Z)','Fraction of mean Z')):
            sns.heatmap(values,cmap='Blues',ax=ax,xticklabels=frame.columns,yticklabels=frame.index,cbar_kws={'label':label})
            ax.set(xlabel='Communication program',ylabel='Niche',title=title);ax.tick_params(axis='x',rotation=90,labelsize=7)
        return self.save(fig,'niche_program_activity'),frame

    def interpretation(self):
        up=self.summary.get('upstream_residual_test_R2');down=self.summary.get('downstream_residual_test_R2')
        if up is not None and down is not None:
            print(f'Held-out biological residual R²: upstream {up:.4f}, downstream {down:.4f}.')
            if max(up,down)<.01:
                print('Regulatory predictive signal is very weak. These prior-linked annotations do not validate signaling mechanisms or establish benefit from biological losses; training ablations are still needed.')
        ccc=pd.read_csv(self.output/'selected_model/top_ccc.csv')
        tables={'CCC':ccc}
        for branch in ('upstream','downstream'):
            path=self.output/('niche_'+branch+'.csv')
            tables[branch]=pd.read_csv(path) if path.is_file() else pd.DataFrame()
        records=[]
        for niche in range(1,self.k+1):
            row={'niche':niche,'cells':int((self.labels==niche-1).sum()),
                'leading_programs':', '.join(f'P{i+1:02d}' for i in np.argsort(-self.contribution.iloc[niche-1].to_numpy())[:3]),
                'representative_CCC':'; '.join(ccc.query('niche==@niche').ccc.head(3))}
            for branch in ('upstream','downstream'):
                table=tables[branch]
                if len(table):
                    selected=table.query('niche==@niche').head(3)
                    row[branch]='; '.join(f'{x.cell_type}: {x.entity}' for x in selected.itertuples())
                    if branch=='downstream': row['response_targets']='; '.join(', '.join(str(x).split(';')[:8]) for x in selected.targets)
                else: row[branch]='No covered prior module'
            records.append(row)
        overview=pd.DataFrame(records);overview.to_csv(self.tables/'niche_biological_summary.csv',index=False)
        return overview,tables

    def biological_heatmaps(self):
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        for ax,branch,title in zip(axes,('upstream','downstream'),('Sender upstream regulators','Receiver ligand-linked response modules')):
            table=pd.read_csv(self.output/('niche_'+branch+'.csv'))
            top=(table.assign(absolute=table.prior_loading.abs()).groupby('module').absolute.max().sort_values(ascending=False).head(18).index)
            values=table.pivot_table(index='niche',columns='module',values='prior_loading',aggfunc='first').reindex(index=range(1,self.k+1),columns=top).fillna(0)
            labels=[str(x).replace('|',' | ') for x in top]
            sns.heatmap(values,cmap=sns.diverging_palette(240,30,as_cmap=True),center=0,ax=ax,xticklabels=labels,yticklabels=range(1,self.k+1),cbar_kws={'label':'Signed prior-linked dictionary loading'})
            ax.set(xlabel='Cell-type-specific module',ylabel='Niche',title=title);ax.tick_params(axis='x',rotation=90,labelsize=7)
        return self.save(fig,'niche_regulatory_response_modules')

    def comparison(self):
        fig,axes=plt.subplots(1,3,figsize=(16,6.1),layout='constrained')
        self._spatial(axes[0],self.old_labels,f'Original AE niches (K={self.old_k})');self._spatial(axes[1],self.labels,f'GNN niches (K={self.k})')
        for ax,k in ((axes[0],self.old_k),(axes[1],self.k)):
            handles=[mpl.lines.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=3,label=f'Niche {i+1}') for i in range(k)]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=4,fontsize=6)
        counts=pd.crosstab(self.old_labels+1,self.labels+1);fraction=counts.div(counts.sum(1),axis=0)
        sns.heatmap(fraction,cmap='Blues',ax=axes[2],cbar_kws={'label':'Fraction of AE niche assigned to GNN niche'})
        axes[2].set(xlabel='GNN niche',ylabel='Original AE niche',title='Cell-level correspondence')
        counts.to_csv(self.tables/'AE_GNN_cell_contingency.csv')
        stats=pd.Series({'Cells':len(self.labels),'Original AE K':self.old_k,'GNN K':self.k,
            'Adjusted Rand index':adjusted_rand_score(self.old_labels,self.labels),
            'Normalized mutual information':normalized_mutual_info_score(self.old_labels,self.labels),
            'Comparison scope':'Descriptive same-tissue correspondence; original AE used overlapping random folds'}).to_frame('Value')
        stats.to_csv(self.tables/'AE_GNN_comparison.csv')
        return self.save(fig,'AE_GNN_comparison'),stats
