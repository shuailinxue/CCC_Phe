"""Compact niche-first comparison: observed tissue, shared niches, case novelty.

Figure contract: paired spatial image plate plus quantitative summaries; same
physical cells/geometry and labels across comparisons; no population p-values.
Python-only plotting; 450 dpi PNG, editable PDF/SVG and CSV on NAS.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from .colitis_reporting import ColitisReport,ACTIVITY
from .skin_paper_context import IMMUNE_TYPES,niche_immune_summary


class DermatomyositisReport(ColitisReport):
    def __init__(self,output):
        self.output=Path(output);self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)
        self.summary=json.loads((self.output/'summary.json').read_text());self.k=self.summary['selected_K']
        self.cells=pd.read_csv(self.output/'assignments.csv.gz');self.types=self.summary['cell_types']
        self.z=np.load(self.output/'gnn/Z.npy',mmap_mode='r');self.h=np.load(self.output/'gnn/H.npy')
        self.features=pd.read_csv(self.output/'input/features.csv')
        self.freq=pd.read_csv(self.output/'condition_niche_fractions.csv',index_col=0)
        self.novel=pd.read_csv(self.output/'case_novelty/cell_scores.csv.gz')
        if not np.array_equal(self.novel.cell_id.astype(str),self.cells.cell_id.astype(str)):
            raise ValueError('Novelty and niche cell order differ')
        self.candidates=pd.read_csv(self.output/'case_novelty/candidate_niches.csv')
        colors=list(plt.get_cmap('tab20').colors)
        # Keep later niche IDs legible on pale tissue backgrounds as well.
        high_contrast=[to_hex(colors[i]) for i in range(0,20,2)]+['#3F3C8C','#BC2D6B','#167D72','#A06A00','#6F364E']
        self.niche_colors=high_contrast[:self.k]
        # Quiet structural populations; immune lineages stay easy to distinguish.
        type_palette={'Fibroblasts':'#9CAFB9','Epidermis_keranocytes':'#C59B85',
            'Folicular_keratinocytes':'#D8B67B','Endothelial':'#477FA9','Pericytes':'#86ADBF',
            'Lymphatic':'#84A3A3','Smooth_muscle':'#687F86','Myeloid':'#D87938',
            'Lymphoid':'#248C6E','B_Plasma':'#8463AD','Mast':'#BD708F',
            'Melanocytes':'#8D715B','Eccrine_gland':'#C5B975','Apocrine_gland':'#9DA36F',
            'Sebaceous_gland':'#B7A06B','Adipocytes':'#D2CBB4','Schwann_cells':'#A8A0BA'}
        self.type_colors=[type_palette.get(t,to_hex(colors[i%len(colors)])) for i,t in enumerate(self.types)]
        self.composition=np.load(self.output/'input/composition.npy',mmap_mode='r')
        self.immune_fraction=np.asarray(self.composition)[:,[i for i,t in enumerate(self.types) if t in IMMUNE_TYPES]].sum(1)
        self.immune_niches=niche_immune_summary(self.cells,self.composition,self.types)
        self.immune_niches.to_csv(self.output/'niche_immune_context.csv',index=False)
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
            'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
            'svg.fonttype':'none','legend.frameon':False,'figure.dpi':120})

    def overview(self):
        quality=pd.read_csv(self.output/'section_quality_and_selection.csv')
        selected=quality.loc[quality.selected,['condition','patient','image','cells','median_counts','median_genes','inflammation_local_p90']].copy()
        selected['retained_cells']=[int(((self.cells.patient==r.patient)&(self.cells.image==r.image)).sum()) for r in selected.itertuples()]
        paper=self.output/'paper_guided_section_audit.csv'
        if paper.exists():
            source=pd.read_csv(paper)
            selected=selected.merge(source[['patient','image','author_Dermis_Disease_aggregates','author_largest_Dermis_Disease_cells']],on=['patient','image'],how='left',validate='one_to_one')
        return selected.reset_index(drop=True)

    def paper_context(self):
        """Separate dermal states and immune architecture without removing cells.

        The third column is a display mask only: exact CCC physical-pair immune
        composition >=20%. All cells remain in training and whole-tissue maps.
        """
        fig,axes=plt.subplots(2,3,figsize=(14,8.1))
        fig.subplots_adjust(left=.015,right=.99,top=.94,bottom=.25,hspace=.12,wspace=.08)
        fibro=self.cells.cell_type.eq('Fibroblasts').to_numpy()
        immune=self.cells.cell_type.isin(IMMUNE_TYPES).to_numpy()
        fine=self.cells.cell_type_fine.astype(str)
        fibro_states=sorted(fine[fibro].unique())
        immune_states=fine[immune].value_counts().head(11).index.tolist()
        if (~fine[immune].isin(immune_states)).any():immune_states.append('Other immune')
        dark=[to_hex(c) for c in plt.get_cmap('tab20').colors[::2]]+['#A56F51','#6A6772']
        fibro_colors={'reticular_fib':'#748DA3','papillary_fib':'#C4984E','vascular_fib':'#3C8D82',
            'CCL19_vascular_fib':'#36675F','CXCL10_fib':'#B84F77','SFRP1_fib':'#8E73AC',
            'ASPN_fib':'#A56F51','cyclic_fib':'#E09845'}
        fp=['#ECEEF0']+[fibro_colors.get(t,dark[i%len(dark)]) for i,t in enumerate(fibro_states)]
        ip=['#ECEEF0']+dark[:len(immune_states)]
        fcode=np.zeros(len(self.cells),int);icode=fcode.copy()
        for i,t in enumerate(fibro_states,1):fcode[fibro&fine.eq(t).to_numpy()]=i
        for i,t in enumerate(immune_states,1):
            match=~fine.isin(immune_states).to_numpy() if t=='Other immune' else fine.eq(t).to_numpy()
            icode[immune&match]=i
        near_immune=self.immune_fraction>=.2
        ncode=np.where(near_immune,self.cells.niche.to_numpy(),0)
        for row,g in enumerate(('HC','DM')):
            ids=np.flatnonzero(self.cells.condition.to_numpy()==g)
            patient=self.cells.iloc[ids].patient.iloc[0].replace('patient_','')
            self.tissue(axes[row,0],ids,fcode[ids],f'{g} · {patient}\nFibroblast states',fp)
            self.tissue(axes[row,1],ids,icode[ids],f'{g} · {patient}\nImmune-cell states',ip)
            self.tissue(axes[row,2],ids,ncode[ids],f'{g} · {patient}\nGNN niches in immune-rich neighborhoods',['#ECEEF0']+self.niche_colors)
        groups=[(fibro_states,fp[1:]),(immune_states,ip[1:]),([f'N{i+1}' for i in range(self.k)],self.niche_colors)]
        for col,(labels,palette) in enumerate(groups):
            handles=[Line2D([],[],marker='o',ls='',color=color,label=label.replace('_',' '),markersize=4) for label,color in zip(labels,palette)]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=((col+.5)/3,.235),ncol=2 if col<2 else 4,
                       fontsize=7,columnspacing=.8,handletextpad=.25)
        fig.text(.5,.025,'Gray = other cells/neighborhoods. Right: immune physical-pair weight ≥20% (display only). All cells trained jointly; original µm coordinates.',ha='center',fontsize=8,color='#555555')
        self.cells[['cell_id','condition','niche','cell_type_fine']].assign(pair_weighted_immune_fraction=self.immune_fraction,
            immune_rich_display=near_immune).to_csv(self.output/'immune_architecture_display.csv.gz',index=False)
        return self.save(fig,'paper_guided_cell_states_and_immune_niches')

    def immune_spotlight_niche(self):
        # A declared interpretation focus, never a label/model-selection rule.
        priority=self.immune_niches
        eligible=priority[(priority.pair_weighted_immune_fraction>=.15)&(priority.DM_minus_HC>0)&(priority.DM_cells>=60)]
        return int((eligible if len(eligible) else priority).iloc[0].niche)

    def spatial(self):
        fig,axes=plt.subplots(2,2,figsize=(11,8));fig.subplots_adjust(bottom=.18,hspace=.18,wspace=.12)
        for j,condition in enumerate(('HC','DM')):
            ids=np.flatnonzero(self.cells.condition.to_numpy()==condition);local=self.cells.iloc[ids]
            code=pd.Categorical(local.cell_type,categories=self.types).codes
            patient=local.patient.iloc[0].replace('patient_','')
            self.tissue(axes[0,j],ids,code,f'{condition} · {patient} · cell types · {len(ids):,} cells',self.type_colors,compact=False)
            self.tissue(axes[1,j],ids,local.niche.to_numpy()-1,f'{condition} · communication niches · K={self.k}',self.niche_colors,compact=False)
        types=[Line2D([],[],marker='o',ls='',color=self.type_colors[i],label=t,markersize=4) for i,t in enumerate(self.types)]
        niches=[Line2D([],[],marker='o',ls='',color=self.niche_colors[i],label=f'N{i+1}',markersize=4) for i in range(self.k)]
        fig.legend(handles=types,loc='lower center',bbox_to_anchor=(.5,.045),ncol=5,fontsize=8)
        fig.legend(handles=niches,loc='lower center',bbox_to_anchor=(.5,.005),ncol=min(self.k,10),fontsize=8)
        return self.save(fig,'paired_tissue_and_shared_niches')

    def abundance(self):
        x=np.arange(self.k);fig,grid=plt.subplots(2,2,figsize=(12,7.8),layout='constrained')
        axes=grid[0]
        for j,(g,color) in enumerate((('HC','#3976AC'),('DM','#D66A58'))):
            axes[0].bar(x+(j-.5)*.36,self.freq.loc[g].to_numpy(float)*100,width=.36,label=g,color=color)
        axes[0].set(xticks=x,xticklabels=[f'N{i+1}' for i in x],ylabel='% of section cells',title='Shared-niche abundance · one section per condition');axes[0].legend()
        difference=self.freq.loc['DM'].to_numpy(float)-self.freq.loc['HC'].to_numpy(float)
        order=np.argsort(difference)
        axes[1].barh(x,difference[order]*100,color=np.where(difference[order]>=0,'#D66A58','#3976AC'))
        axes[1].set(yticks=x,yticklabels=[f'N{i+1}' for i in order],xlabel='DM − HC (percentage points)',title='Which niches expand in the DM section?');axes[1].axvline(0,color='#888888',lw=.7)
        niche=self.immune_spotlight_niche()-1
        for j,g in enumerate(('HC','DM')):
            ids=np.flatnonzero(self.cells.condition.to_numpy()==g)
            highlight=(self.cells.iloc[ids].niche.to_numpy()==niche+1).astype(int)
            palette=['#DFE2E6',to_hex(self.niche_colors[niche])]
            fraction=float(self.freq.loc[g,str(niche+1)])
            self.tissue(grid[1,j],ids,highlight,f'{g} · N{niche+1}\n{fraction:.3%} · {int(highlight.sum()):,} cells',palette)
        return self.save(fig,'shared_niche_abundance')

    def novelty(self):
        fig,axes=plt.subplots(2,2,figsize=(11,8));fig.subplots_adjust(bottom=.08,hspace=.22,wspace=.15)
        n=int(self.novel.candidate_niche.max());palette=['#D9DDE2']+[to_hex(c) for c in plt.get_cmap('Dark2').colors[:n]]
        for j,g in enumerate(('HC','DM')):
            ids=np.flatnonzero(self.cells.condition.to_numpy()==g)
            ratio=self.novel.novelty_score_ratio.to_numpy()[ids]
            im=self.tissue(axes[0,j],ids,ratio,f'{g} · communication deviation\nComposition-matched healthy background',cmap=ACTIVITY,vmax=max(2,float(np.quantile(self.novel.novelty_score_ratio,.99))))
            self.tissue(axes[1,j],ids,self.novel.candidate_niche.to_numpy()[ids],f'{g} · candidate case-emergent niches',palette,compact=False)
        fig.colorbar(im,ax=axes[0,:],fraction=.025,pad=.02,label='Score / healthy threshold (1 = threshold)')
        handles=[Line2D([],[],ls='',marker='o',color=palette[0],label='Background',markersize=4)]
        handles.extend(Line2D([],[],ls='',marker='o',color=palette[i],label=f'R{i}',markersize=4) for i in range(1,n+1))
        fig.legend(handles=handles,loc='lower center',ncol=6)
        return self.save(fig,'healthy_calibrated_case_niches')

    def comparison(self):
        """Same locations: ordinary joint discovery versus added novelty analysis."""
        fig,axes=plt.subplots(2,3,figsize=(14,7));fig.subplots_adjust(wspace=.1,hspace=.16,bottom=.14)
        n=int(self.novel.candidate_niche.max())
        added=['#A52A62','#513C8B','#C66C26','#1D6E6B','#6A4337'][:n]
        palette=[to_hex(c) for c in self.niche_colors]+added
        overlay=self.cells.niche.to_numpy()-1;candidate=self.novel.candidate_niche.to_numpy()
        overlay[candidate>0]=self.k+candidate[candidate>0]-1
        for row,g in enumerate(('HC','DM')):
            ids=np.flatnonzero(self.cells.condition.to_numpy()==g);local=self.cells.iloc[ids]
            self.tissue(axes[row,0],ids,pd.Categorical(local.cell_type,categories=self.types).codes,f'{g} · cell types',self.type_colors)
            self.tissue(axes[row,1],ids,local.niche.to_numpy()-1,f'{g} · joint niches',self.niche_colors)
            self.tissue(axes[row,2],ids,overlay[ids],f'{g} · joint niches + candidate additions',palette)
        handles=[Line2D([],[],marker='o',ls='',color=palette[j],label=f'N{j+1}',markersize=4) for j in range(self.k)]
        handles.extend(Line2D([],[],marker='o',ls='',color=added[j],label=f'R{j+1}',markersize=4) for j in range(n))
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),ncol=min(10,len(handles)),fontsize=7)
        fig.text(.5,.015,'Same model and cells. Original N labels remain outside candidate R regions; no retraining or forced group matching.',ha='center',fontsize=8)
        return self.save(fig,'joint_vs_case_emergent_summary')

    def comparison_table(self):
        path=self.output/'selection_and_metric_comparison.csv'
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def training(self):
        history=pd.read_csv(self.output/'gnn/training.csv');selection=pd.read_csv(self.output/'gnn/K_selection.csv')
        best=json.loads((self.output/'gnn/training_monitor_metrics.json').read_text())['best_epoch']
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
        axes[0].plot(history.epoch,history.train_ccc,label='All centers',color='#3976AC');axes[0].plot(history.epoch,history.monitor_ccc,label='Training-subset monitor',color='#D66A58')
        axes[0].axvline(best,ls=':',color='#888888');axes[0].set(xlabel='Epoch',ylabel='CCC Huber loss',title=f'NMF initialization + joint training · best {best}');axes[0].legend(fontsize=7)
        axes[1].plot(selection.K,selection.selection_score,'o-',color='#745190');axes[1].axvline(self.k,ls=':',color='#888888');axes[1].set(xlabel='K',ylabel='Unsupervised score',title=f'Joint niche selection · K={self.k}')
        target=np.load(self.output/'input/ccc.npy',mmap_mode='r');pred=np.load(self.output/'ccc_reconstruction.npy',mmap_mode='r')
        rng=np.random.default_rng(7);rows=rng.choice(len(target),min(1000,len(target)),replace=False)
        mask=np.load(self.output/'input/preprocessing.npz')['feature_mask'];x=np.log1p(target[rows][:,mask]).ravel();y=np.log1p(pred[rows][:,mask]).ravel()
        pick=rng.choice(len(x),min(30000,len(x)),replace=False);axes[2].hexbin(x[pick],y[pick],gridsize=45,mincnt=1,cmap='Blues',bins='log',rasterized=True)
        lim=max(float(np.quantile(x,.99)),float(np.quantile(y,.99)),.1);axes[2].plot([0,lim],[0,lim],ls='--',color='#888888',lw=.7)
        axes[2].set(xlabel='Observed log1p CCC',ylabel='Reconstructed log1p CCC',title='Observed vs reconstructed communication')
        return self.save(fig,'training_and_CCC_reconstruction')

    def tables(self):
        top=pd.read_csv(self.output/'niche_top10_CCC.csv')
        selected=self.immune_niches.query('DM_minus_HC>0').head(2).niche.tolist()
        return top[top.niche.isin(selected)].round({'mean_reconstructed_CCC':3})
