"""Niche-first paired tissue evidence; Python image plate + quantitative panels.

Contract: whole-tissue shared background and locally coherent exceptions, then
separate composition shifts from composition-matched observed CCC changes.
Original coordinates and all retained centers; no geometric warping or invented
success labels. PNG450dpi, editable PDF/SVG and source CSV remain on NAS.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .colitis_reporting import ColitisReport
from .paired_skin import input_root

TYPE_COLORS={'Fibroblasts':'#A2B4BC','Epidermis_keranocytes':'#C7A087','Folicular_keratinocytes':'#D6B67B',
 'Endothelial':'#477FA9','Pericytes':'#86ADBF','Lymphatic':'#84A3A3','Smooth_muscle':'#687F86',
 'Myeloid':'#D87938','Lymphoid':'#248C6E','B_Plasma':'#8463AD','Mast':'#BD708F','Melanocytes':'#8D715B',
 'Eccrine_gland':'#C5B975','Apocrine_gland':'#9DA36F','Sebaceous_gland':'#B7A06B','Adipocytes':'#D2CBB4','Schwann_cells':'#A8A0BA','Unresolved':'#E1E3E5'}


class ConditionNicheReport(ColitisReport):
    def __init__(self,source):
        self.output=Path(source);self.analysis=self.output/'condition_niches';self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)
        self.input=input_root(source)
        self.summary=json.loads((self.output/'summary.json').read_text());self.cells=pd.read_csv(self.analysis/'assignments.csv.gz')
        self.valid=self.cells.analysis_center.to_numpy(bool) if 'analysis_center' in self.cells else np.ones(len(self.cells),bool)
        self.types=self.summary['cell_types'];self.k=self.summary['selected_K'];self.z=np.load(self.output/'gnn/Z.npy',mmap_mode='r');self.h=np.load(self.output/'gnn/H.npy')
        self.features=pd.read_csv(self.input/'features.csv');self.candidates=pd.read_csv(self.analysis/'candidate_niches_with_evidence.csv')
        self.labels=sorted(self.cells.label.unique(),key=lambda x:(x[0]!='B',int(x[1:])))
        base=[x for x in self.labels if x.startswith('B')];novel=[x for x in self.labels if x.startswith('R')]
        quiet=['#6C8EA4','#9BB7AA','#B5A894','#969BB8','#BFAFC2','#9CB6B8','#B6BE9C','#B8A194','#A8ABB0','#B0B9CC','#B5C3AF','#CCBAAA','#9EAABD','#B9A6B2','#AFB6A0']
        vivid=['#A62B61','#6A4297','#D17C28','#187F80','#B44B39','#547523']
        self.colors={**{x:quiet[i%len(quiet)] for i,x in enumerate(base)},**{x:vivid[i%len(vivid)] for i,x in enumerate(novel)}}
        self.niche_colors=[self.colors[x] for x in self.labels];self.type_colors=[TYPE_COLORS.get(x,plt.get_cmap('tab20').colors[i%20]) for i,x in enumerate(self.types)]
        self.view_types=self.types;self.view_colors=self.type_colors;self.view_cell_type=self.cells.cell_type.astype(str).to_numpy()
        if 'PT' in self.types and 'TAL' in self.types:
            families={'PT':'Proximal tubule','TAL':'Loop of Henle','DTL_ATL':'Loop of Henle','DCT':'Distal tubule','CNT':'Distal tubule',
                'PC':'Collecting duct','IC A':'Collecting duct','IC B':'Collecting duct','Podo':'Glomerular epithelium','PEC':'Glomerular epithelium',
                'MC1':'Mural / mesangial','VSMC':'Mural / mesangial','Fibroblast':'Fibroblast',
                'CD4+':'T / NK cells','CD8+':'T / NK cells','NK':'T / NK cells','B':'B / plasma cells','Plasma':'B / plasma cells',
                'Macro':'Myeloid cells','cDC':'Myeloid cells','mDC':'Myeloid cells','pDC':'Myeloid cells','Neutrophil':'Myeloid cells','Baso_Mast':'Myeloid cells','Immune_unknown':'Unresolved immune'}
            families.update({t:'Endothelium' for t in self.types if t.startswith('EC_')})
            self.view_cell_type=np.asarray([families.get(t,t) for t in self.view_cell_type]);self.view_types=sorted(set(self.view_cell_type))
            palette=['#8B6AA7','#D9AE61','#5B96B1','#70A7A1','#CA8262','#A8BCA0','#B6A3A0','#D46B60','#4179A3','#6CA682','#AAB0B8','#B3A0C3']
            self.view_colors=palette[:len(self.view_types)]
        analysis_audit=json.loads((self.analysis/'audit.json').read_text());self.control=analysis_audit['control'];self.case=analysis_audit['case']
        self.joint_colors=[plt.get_cmap('tab20').colors[i] for i in list(range(0,20,2))+list(range(1,20,2))][:self.k]
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'pdf.fonttype':42,'svg.fonttype':'none',
            'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'figure.dpi':120})

    def overview(self):
        rows=[]
        for group,local in self.cells.loc[self.valid].groupby('condition',sort=False):
            rows.append({'section':local.image.iloc[0],'status':{'NL':'Nonlesional AD control','AD':'Lesional AD'}.get(group,group),
                'cells':len(local),'matrix_gene_columns':self.summary['genes'],'CCC_features':self.summary['CCC_features'],
                'effective_CCC_features':self.summary['effective_CCC_features']})
        return pd.DataFrame(rows)

    def atlas(self):
        fig,axes=plt.subplots(2,3,figsize=(14,9.5));fig.subplots_adjust(left=.02,right=.99,top=.94,bottom=.26,hspace=.15,wspace=.1)
        for row,g in enumerate((self.control,self.case)):
            ids=np.flatnonzero((self.cells.condition.to_numpy()==g)&self.valid);local=self.cells.iloc[ids];patient=local.image.iloc[0]
            self.tissue(axes[row,0],ids,pd.Categorical(self.view_cell_type[ids],categories=self.view_types).codes,f'{g} · {patient} · cell lineages',self.view_colors)
            self.tissue(axes[row,1],ids,local.niche.to_numpy()-1,f'{g} · joint GNN niches · K={self.k}',self.joint_colors)
            code=pd.Categorical(local.label,categories=self.labels).codes
            self.tissue(axes[row,2],ids,code,f'{g} · shared background + new states',self.niche_colors)
        self._legend(fig,self.view_types,self.view_colors,.085,6,8)
        self._legend(fig,self.labels,self.niche_colors,.018,min(10,len(self.labels)),8)
        fig.text(.5,.215,'B = control background niches; R = spatially coherent exceptions. Same rule in both sections; original µm coordinates.',ha='center',fontsize=8)
        return self.save(fig,'shared_background_and_case_niches')

    @staticmethod
    def _legend(fig,labels,colors,y,ncol,size=8):
        handles=[Line2D([],[],marker='o',ls='',color=c,label=l.replace('_',' '),markersize=4) for l,c in zip(labels,colors)]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=max(1,ncol),fontsize=size,columnspacing=1.,handletextpad=.3)

    def abundance(self):
        shown=self.cells.loc[self.valid]
        freq=pd.crosstab(shown.condition,shown.label,normalize='index').reindex(index=[self.control,self.case],columns=self.labels,fill_value=0)
        freq.to_csv(self.analysis/'all_niche_fractions.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained');x=np.arange(len(self.labels))
        axes[0].bar(x-.18,freq.loc[self.control]*100,.36,color='#477FA9',label=self.control);axes[0].bar(x+.18,freq.loc[self.case]*100,.36,color='#C76B63',label=self.case)
        axes[0].set(xticks=x,xticklabels=self.labels,ylabel='% of section centers',title='Shared and new niche abundance');axes[0].legend(fontsize=8)
        cand=self.candidates.sort_values('case_fraction',ascending=False)
        if len(cand):
            axes[1].barh(np.arange(len(cand)),cand.case_fraction*100,color=[self.colors[t] for t in cand.niche])
            axes[1].set(yticks=np.arange(len(cand)),yticklabels=cand.niche,xlabel='% of case centers',title='Exceptional states · observed control counts annotated');axes[1].invert_yaxis()
            for i,r in enumerate(cand.itertuples()):axes[1].text(r.case_fraction*100+.15,i,f'Control: {r.control_cells} cells',va='center',fontsize=8)
        else:axes[1].text(.5,.5,'No coherent exceptional state passed the rule',ha='center',transform=axes[1].transAxes)
        return self.save(fig,'niche_abundance')

    def spotlight(self):
        selected=self.candidates.query('case_enriched').sort_values('case_fraction',ascending=False).niche.tolist()[:4]
        if not selected:selected=self.candidates.niche.tolist()[:4]
        fig,axes=plt.subplots(max(1,len(selected)),2,figsize=(10,3.6*max(1,len(selected))),squeeze=False)
        fig.subplots_adjust(hspace=.18,wspace=.08)
        if not selected:
            for ax in axes.flat:ax.axis('off')
            axes[0,0].text(.5,.5,'No exceptional niches passed the prespecified rule',ha='center',transform=axes[0,0].transAxes)
        for row,label in enumerate(selected):
            for col,g in enumerate((self.control,self.case)):
                ids=np.flatnonzero((self.cells.condition.to_numpy()==g)&self.valid);hit=(self.cells.label.to_numpy()[ids]==label).astype(int)
                self.tissue(axes[row,col],ids,hit,f'{g} · {label} · {hit.mean():.2%} · {hit.sum():,} centers',['#E5E7EA',self.colors[label]])
        return self.save(fig,'exceptional_niche_spatial_maps')

    def matched_communication(self):
        comp=pd.read_csv(self.analysis/'matched_composition.csv',index_col=0);top=pd.read_csv(self.analysis/'matched_observed_top_CCC.csv')
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,.5*len(comp)+2)),layout='constrained',gridspec_kw={'width_ratios':[1,1.5]})
        if not len(comp):
            for ax in axes:ax.text(.5,.5,'No exceptional-state comparison available',ha='center',transform=ax.transAxes)
            return self.save(fig,'composition_matched_observed_CCC')
        im=axes[0].imshow(comp.to_numpy()*100,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
        axes[0].set(xticks=np.arange(len(comp.columns)),xticklabels=comp.columns,yticks=np.arange(len(comp)),yticklabels=comp.index,title='Compare the actual matched cell mixtures')
        plt.setp(axes[0].get_xticklabels(),rotation=60,ha='right',fontsize=7);fig.colorbar(im,ax=axes[0],shrink=.65,label='% physical-pair weight')
        selected=top.groupby('niche',sort=False).head(3).drop_duplicates('feature');f=selected.feature.to_numpy(int)
        values=np.load(self.analysis/'matched_observed_relative_CCC.npy')[:,f]
        values=np.log1p(values*10000);values/=np.maximum(values.max(0,keepdims=True),1e-8)
        im=axes[1].imshow(values,aspect='auto',cmap='PuRd',vmin=0,vmax=1)
        names=[f'{r.sender}→{r.receiver}\n{r.LR}' for r in selected.itertuples()]
        axes[1].set(xticks=np.arange(len(f)),xticklabels=names,yticks=np.arange(len(comp)),yticklabels=comp.index,title='Observed CCC · relative profiles · same matched centers')
        plt.setp(axes[1].get_xticklabels(),rotation=65,ha='right',fontsize=7);fig.colorbar(im,ax=axes[1],shrink=.65,label='Column-normalized log relative CCC')
        return self.save(fig,'composition_matched_observed_CCC')

    def programs(self):
        program=pd.read_csv(self.analysis/'niche_program_activity.csv',index_col=0);comp=pd.read_csv(self.analysis/'niche_composition.csv',index_col=0)
        v=np.log1p(program.to_numpy());v=(v-v.mean(0))/np.maximum(v.std(0),.05)
        fig,axes=plt.subplots(1,2,figsize=(14,max(4,.34*len(program)+2)),layout='constrained',gridspec_kw={'width_ratios':[1.5,1]})
        im=axes[0].imshow(v,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2);axes[0].set(xticks=np.arange(32),xticklabels=np.arange(1,33),yticks=np.arange(len(program)),yticklabels=program.index,xlabel='CCC program',title='Niche × learned communication program');fig.colorbar(im,ax=axes[0],shrink=.65,label='Column z score of log1p mean Z')
        axes[0].tick_params(axis='x',labelsize=7)
        im=axes[1].imshow(comp.to_numpy()*100,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100);axes[1].set(xticks=np.arange(len(comp.columns)),xticklabels=comp.columns,yticks=np.arange(len(comp)),yticklabels=comp.index,title='Niche × spatially weighted cell composition');plt.setp(axes[1].get_xticklabels(),rotation=60,ha='right',fontsize=7);fig.colorbar(im,ax=axes[1],shrink=.65,label='%')
        return self.save(fig,'niche_program_and_composition')

    def training(self):
        history=pd.read_csv(self.output/'gnn/training.csv');bg=pd.read_csv(self.analysis/'background_K_selection.csv')
        fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
        axes[0].plot(history.epoch,history.train_ccc,label='All training centers',color='#477FA9');axes[0].plot(history.epoch,history.monitor_ccc,label='Training-subset monitor',color='#C76B63');axes[0].set(xlabel='Epoch',ylabel='Feature-scaled CCC Huber',title='NMF initialization + joint GNN training');axes[0].legend(fontsize=7)
        axes[1].plot(bg.K,bg.score,'o-',color='#7C6698');selected=int(bg.loc[bg.selected,'K'].iloc[0]);axes[1].axvline(selected,ls=':',color='#777777');axes[1].set(xlabel='Number of background niches',ylabel='Condition-blind control-only score',title=f'Background selection · K={selected}')
        raw=np.load(self.input/'ccc.npy',mmap_mode='r');pred=np.load(self.output/'ccc_reconstruction.npy',mmap_mode='r');rng=np.random.default_rng(7);rows=rng.choice(len(raw),min(300,len(raw)),replace=False);cols=rng.choice(raw.shape[1],min(300,raw.shape[1]),replace=False)
        x=np.log1p(raw[rows][:,cols]).ravel();y=np.log1p(pred[rows][:,cols]).ravel();axes[2].hexbin(x,y,gridsize=40,mincnt=1,cmap='Blues',bins='log',rasterized=True);limit=max(float(np.quantile(x,.995)),float(np.quantile(y,.995)),.1);axes[2].plot([0,limit],[0,limit],ls='--',color='#777777',lw=.7);axes[2].set(xlabel='Observed log1p C',ylabel='Reconstructed log1p C',title='Original CCC reconstruction')
        return self.save(fig,'training_and_reconstruction')

    def top_CCC(self):
        table=pd.read_csv(self.analysis/'niche_top10_CCC.csv');return table[table.niche.str.startswith('R')].round({'reconstructed_CCC':3})

    def evidence(self):
        composition=pd.read_csv(self.analysis/'niche_composition.csv',index_col=0);rows=[]
        for r in self.candidates.itertuples():
            context=' · '.join(f'{name} {value:.0%}' for name,value in composition.loc[r.niche].nlargest(3).items())
            rows.append({'Niche':r.niche,'主要细胞组成':context,'健康中心数':r.control_cells,'病例中心数':r.case_cells,
                '组成匹配距离':round(r.mean_matched_composition_Hellinger,3),'同组成CCC差异支持':r.same_composition_CCC_supported})
        return pd.DataFrame(rows)

    def source_selection(self):
        """Always expose anatomical ROI selection and the unmodified full section."""
        origin=self.input.parent
        if not (origin/'healthy_full_section_ROI.csv.gz').exists():return None
        fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
        for ax,group,color in zip(axes,(self.control,self.case),('#477FA9','#C76B63')):
            path=origin/f'{group}_full_section_ROI.csv.gz'
            if group==self.control and not path.exists():path=origin/'healthy_full_section_ROI.csv.gz'
            if path.exists():
                frame=pd.read_csv(path);selected=frame.selected_ROI.to_numpy(bool)
                ax.scatter(frame.x,frame.y,s=.3,c='#DADDE0',rasterized=True);ax.scatter(frame.x[selected],frame.y[selected],s=.4,c=color,rasterized=True)
                ax.set_title(f'{group} · whole section and anatomical ROI')
            else:
                ids=np.flatnonzero((self.cells.condition.to_numpy()==group)&self.valid);xy=self.cells.iloc[ids][['x','y']].to_numpy()
                ax.scatter(xy[:,0],xy[:,1],s=.6,c=color,rasterized=True);ax.set_title(f'{group} · whole biopsy · all retained centers')
            ax.set_aspect('equal');ax.invert_yaxis();ax.set(xlabel='Original x (µm)',ylabel='Original y (µm)')
        return self.save(fig,'original_sections_and_anatomical_ROI')
