"""Observed receiver RNA → prior-projected targets → local spatial association.

Three compact quantitative plates, all with saved source data. Same-cell-type
within-DKD comparisons are primary. Projected potentials and spatial intervals
are descriptive, not direct targets, activation signs or patient-level tests.
"""
import re
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .kidney_downstream_reporting import KidneyDownstreamReport
from .kidney_response import FOCUS,ccc_genes,is_immunoglobulin


class KidneyResponseReport(KidneyDownstreamReport):
    def __init__(self,source):
        super().__init__(source)
        self.response=self.downstream/'receiver_response'
        self.effects=pd.read_csv(self.response/'receiver_gene_effects.csv.gz')
        self.links=pd.read_csv(self.response/'ligand_target_links.csv')
        self.associations=pd.read_csv(self.response/'spatial_associations.csv')

    def receiver_states(self):
        modules=pd.read_csv(self.response/'hallmark_response_modules.csv')
        fig,axes=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
        source=[];exclude=ccc_genes(self.features)
        for row,typ in enumerate(['CD8+','Macro']):
            t=self.effects[(self.effects.receiver==typ)&~self.effects.gene.isin(exclude)&(self.effects.niche_detection>=.15)&(self.effects.background_detection>=.05)]
            t=t[~t.gene.map(is_immunoglobulin)].nlargest(10,'log2FC').sort_values('log2FC')
            color=['#A76880' if v>0 else '#6297B2' for v in t.log2FC]
            axes[row,0].barh(range(len(t)),t.log2FC,color=color,height=.65)
            axes[row,0].axvline(0,color='#777777',lw=.7)
            n= int((self.valid&self.cells.condition.eq('DKD').to_numpy()&self.cells.label.eq('R2').to_numpy()&self.cells.cell_type.eq(typ).to_numpy()).sum())
            bg=int((self.valid&self.cells.condition.eq('DKD').to_numpy()&self.cells.label.str.startswith('B').to_numpy()&self.cells.cell_type.eq(typ).to_numpy()).sum())
            axes[row,0].set(yticks=range(len(t)),yticklabels=t.gene,xlabel='Observed RNA log₂FC · R2 / DKD shared',title=f'{typ} · same-type receiver genes · n={n} / {bg}')
            source.append(t)
            local=modules[modules.receiver==typ];delta=local[local.group.eq('DKD R2')].set_index('module').mean_score
            chosen=delta.abs().nlargest(6).index
            wide=local.pivot(index='module',columns='group',values='mean_score').reindex(index=chosen,columns=['Healthy','DKD shared','DKD R2'])
            im=axes[row,1].imshow(wide,cmap='RdBu_r',vmin=-.4,vmax=.4,aspect='auto')
            names=[name.replace(' Signaling','').replace('Response','resp.') for name in wide.index]
            axes[row,1].set(yticks=range(len(wide)),yticklabels=names,xticks=range(3),xticklabels=wide.columns,title=f'{typ} · observed non-LR response modules')
            axes[row,1].tick_params(axis='y',labelsize=8)
            fig.colorbar(im,ax=axes[row,1],shrink=.7,label='Mean standardized RNA module score')
            wide.to_csv(self.response/f'displayed_{typ.replace("+","pos")}_Hallmark_scores.csv')
        pd.concat(source).to_csv(self.response/'displayed_receiver_genes.csv',index=False)
        self.letters(axes);return self.save(fig,'downstream_05_receiver_expression_states')

    def ligand_targets(self):
        fig,axes=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
        source=[]
        for row,typ in enumerate(['CD8+','Macro']):
            ligands=['CXCL14','MIF'] if typ=='CD8+' else ['APP','MIF']
            links=self.links[(self.links.receiver==typ)&self.links.ligand.isin(ligands)]
            unique=links.drop_duplicates('target');available=unique[unique.niche_detection>=.1]
            chosen=available.reindex(available.log2FC.abs().sort_values(ascending=False).index).head(10).target.tolist()
            # Potential values absent from a ligand's fixed top30 are masked.
            prior=links.pivot(index='ligand',columns='target',values='prior_potential').reindex(index=ligands,columns=chosen)
            relative=prior.div(prior.max(axis=1),axis=0)
            im=axes[row,0].imshow(relative,cmap='BuPu',vmin=0,vmax=1,aspect='auto')
            axes[row,0].set(yticks=range(len(ligands)),yticklabels=ligands,xticks=range(len(chosen)),xticklabels=chosen,title=f'TAL ligands → {typ} · projected NicheNet targets')
            fig.colorbar(im,ax=axes[row,0],shrink=.6,label='Relative prior potential within ligand')
            values=[];groups=['Healthy','DKD shared','DKD R2']
            dk=self.cells.condition.eq('DKD').to_numpy();ct=self.cells.cell_type.eq(typ).to_numpy()
            masks=[self.valid&~dk&ct,self.valid&dk&ct&self.cells.label.str.startswith('B').to_numpy(),self.valid&dk&ct&self.cells.label.eq('R2').to_numpy()]
            base=self.rna[np.flatnonzero(masks[1])][:,self.genes.get_indexer(chosen)].toarray();mean=base.mean(0);sd=np.maximum(base.std(0),.25)
            for name,mask in zip(groups,masks):
                ids=np.flatnonzero(mask);value=(np.asarray(self.rna[ids][:,self.genes.get_indexer(chosen)].mean(0)).ravel()-mean)/sd
                values.append(value)
                source.extend({'receiver':typ,'group':name,'target':g,'cells':len(ids),'observed_standardized_RNA':float(v)} for g,v in zip(chosen,value))
            im2=axes[row,1].imshow(values,cmap='RdBu_r',vmin=-.5,vmax=.5,aspect='auto')
            axes[row,1].set(yticks=range(3),yticklabels=[f'{name} (n={mask.sum()})' for name,mask in zip(groups,masks)],xticks=range(len(chosen)),xticklabels=chosen,title=f'{typ} · corresponding measured target RNA')
            fig.colorbar(im2,ax=axes[row,1],shrink=.6,label='Mean standardized observed RNA')
            for ax in axes[row]:plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=8)
            links[links.target.isin(chosen)].to_csv(self.response/f'displayed_{typ.replace("+","pos")}_prior_links.csv',index=False)
        pd.DataFrame(source).to_csv(self.response/'displayed_target_expression.csv',index=False)
        fig.supxlabel('Prior potentials are unsigned and indirect; colors on the right are observed expression, not predicted activation.',fontsize=8)
        self.letters(axes);return self.save(fig,'downstream_06_ligand_target_evidence')

    def spatial_response(self):
        data=pd.read_csv(self.response/'receiver_cells.csv.gz');means=pd.read_csv(self.response/'response_by_group.csv')
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');source=[]
        for col,(receiver,ligand) in enumerate(FOCUS):
            local=data[(data.receiver==receiver)&(data.ligand==ligand)];summary=self.associations[(self.associations.receiver==receiver)&(self.associations.ligand==ligand)]
            t=means[(means.receiver==receiver)&(means.ligand==ligand)].set_index('group').reindex(['Healthy','DKD shared','DKD R2'])
            axes[0,col].bar(range(3),t.mean_target_score,color=['#C3CDD2','#67A19B',self.colors['R2']],width=.65)
            axes[0,col].axhline(0,color='#888888',lw=.6)
            axes[0,col].set(xticks=range(3),xticklabels=t.index,ylabel='Mean prior-projected target RNA score',title=f'TAL {ligand} → {receiver} · receiver targets')
            plt.setp(axes[0,col].get_xticklabels(),rotation=25,ha='right',fontsize=8)
            for j,r in enumerate(t.itertuples()):axes[0,col].text(j,r.mean_target_score,f' n={r.cells}',ha='center',va='bottom' if r.mean_target_score>=0 else 'top',fontsize=8)
            low=min(float(t.mean_target_score.min()),0.);high=max(float(t.mean_target_score.max()),0.)
            span=max(high-low,.01);axes[0,col].set_ylim(low-.15*span,high+.2*span)
            if not len(summary) or not len(local):
                axes[1,col].text(.5,.5,'Insufficient measured target / spatial coverage',ha='center',transform=axes[1,col].transAxes);continue
            counts=local.groupby('spatial_block').size();valid=counts[counts>=3].index
            blocks=local[local.spatial_block.isin(valid)].groupby('spatial_block')[['exposure_residual','target_residual']].mean()
            r2=local.groupby('spatial_block').label.apply(lambda v:(v=='R2').mean()).reindex(blocks.index)
            # Display units standardized for readable coordinates; correlation unchanged.
            x=blocks.exposure_residual.to_numpy();y=blocks.target_residual.to_numpy();x/=max(x.std(),1e-8);y/=max(y.std(),1e-8)
            sc=axes[1,col].scatter(x,y,c=r2,cmap='Purples',norm=Normalize(0,1),s=20,lw=.3,edgecolors='#666666')
            if np.std(x)>1e-8:
                fit=np.polyfit(x,y,1);xx=np.linspace(x.min(),x.max(),50);axes[1,col].plot(xx,np.polyval(fit,xx),c='#666666',lw=1)
            r=summary.iloc[0]
            axes[1,col].set(xlabel='Adjusted TAL ligand exposure · block mean',ylabel='Adjusted receiver target RNA · block mean',title=f'{r.blocks} spatial blocks · r={r.block_correlation:.2f}\n95% block interval [{r.block_CI_low:.2f}, {r.block_CI_high:.2f}]')
            fig.colorbar(sc,ax=axes[1,col],shrink=.55,label='R2 receiver fraction')
            source.extend({'receiver':receiver,'ligand':ligand,'spatial_block':int(b),'exposure_residual_scaled':float(a),'target_residual_scaled':float(c),'R2_fraction':float(d)} for b,a,c,d in zip(blocks.index,x,y,r2))
        pd.DataFrame(source).to_csv(self.response/'displayed_spatial_blocks.csv',index=False)
        fig.supxlabel('Adjusted for receiver library size/detected genes, own receptor RNA, TAL opportunity and local composition; blocks are not patients.',fontsize=8)
        self.letters(axes);return self.save(fig,'downstream_07_spatial_receiver_response')

    def evidence_table(self):
        coverage=pd.read_csv(self.response/'prior_coverage.csv');activity=pd.read_csv(self.response/'ligand_activity.csv')
        means=pd.read_csv(self.response/'response_by_group.csv');rows=[];display=[]
        original=pd.read_csv(self.downstream/'R2_same_sender_receiver_CCC.csv')
        for receiver,ligand in FOCUS:
            a=activity[(activity.receiver==receiver)&(activity.ligand==ligand)].iloc[0]
            c=coverage[(coverage.receiver==receiver)&(coverage.ligand==ligand)].iloc[0]
            g=means[(means.receiver==receiver)&(means.ligand==ligand)&means.group.eq('DKD R2')]
            s=self.associations[(self.associations.receiver==receiver)&(self.associations.ligand==ligand)]
            rows.append({'receiver':receiver,'ligand':ligand,'projected_targets':int(c.retained_targets),
                'global_top100_covered':int(c.global_top100_covered),'prior_AUPRC_gain':round(a.AUPRC_minus_prevalence,3),
                'R2_target_score':round(g.mean_target_score.iloc[0],3) if len(g) else np.nan,
                'spatial_r':round(s.block_correlation.iloc[0],3) if len(s) else np.nan,
                'spatial_interval':f'[{s.block_CI_low.iloc[0]:.2f}, {s.block_CI_high.iloc[0]:.2f}]' if len(s) else 'Insufficient coverage'})
            lr=ligand+('–CXCR4' if receiver=='CD8+' else '–CD74')
            obs=original[(original.receiver==receiver)&(original.LR==lr)].iloc[0]
            ci=f'r={s.block_correlation.iloc[0]:.2f} [{s.block_CI_low.iloc[0]:.2f}, {s.block_CI_high.iloc[0]:.2f}]' if len(s) else '覆盖不足'
            display.append({'候选通讯':f'TAL → {receiver} · {lr}',
                '同类细胞对内CCC占比（健康→R2）':f'{obs.control_within_pair_fraction:.1%} → {obs.case_within_pair_fraction:.1%}',
                'R2非LR靶基因评分（同片共有背景=0）':round(float(g.mean_target_score.iloc[0]),3) if len(g) else np.nan,
                '调整后空间关联及95%块区间':ci})
        table=pd.DataFrame(rows);table.to_csv(self.response/'mechanism_summary.csv',index=False)
        shown=pd.DataFrame(display);shown.to_csv(self.response/'displayed_mechanism_summary.csv',index=False);return shown
