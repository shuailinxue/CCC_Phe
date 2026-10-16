"""Niche-first evidence plate: real geometry → contacts → LR → observed RNA.

Python-only figures; shared scales and actual physical coordinates. Every
quantitative panel exports its source table. No fitted trajectory or severity
correlation is inferred from this one-section-per-condition case study.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .condition_reporting import ConditionNicheReport


class KidneyDownstreamReport(ConditionNicheReport):
    def __init__(self,source):
        super().__init__(source)
        self.downstream=self.output/'downstream'
        self.rna=sp.load_npz(self.downstream/'observed_log_expression.npz')
        import anndata as ad
        a=ad.read_h5ad(self.input/'tissue.h5ad',backed='r')
        self.genes=pd.Index(a.var_names);a.file.close()
        self.xy=self.cells[['x','y']].to_numpy()

    @staticmethod
    def letters(axes):
        for letter,ax in zip('abcdefgh',np.asarray(axes).ravel()):
            ax.text(-.1,1.06,letter,transform=ax.transAxes,weight='bold',fontsize=13)

    def contacts_and_CCC(self):
        contacts=pd.read_csv(self.downstream/'physical_contact_enrichment.csv')
        ccc=pd.read_csv(self.downstream/'directed_CCC_networks.csv')
        fig,axes=plt.subplots(2,4,figsize=(15,8),layout='constrained')
        for col,niche in enumerate(['R1','R2','R3','R4']):
            edges=contacts[contacts.niche==niche];mass={t:0. for t in self.types}
            for r in edges.itertuples():
                mass[r.type_a]+=r.observed_weight;mass[r.type_b]+=r.observed_weight
            types=sorted(mass,key=mass.get,reverse=True)[:8];n=len(types)
            matrix=np.full((n,n),np.nan);sig=np.zeros((n,n),bool)
            for r in edges.itertuples():
                if r.type_a not in types or r.type_b not in types:continue
                i,j=types.index(r.type_a),types.index(r.type_b)
                if r.null_mean_weight>=.5:
                    matrix[i,j]=matrix[j,i]=r.log2_enrichment
                    sig[i,j]=sig[j,i]=r.q_spatial_shuffle<.05 and r.log2_enrichment>0
            im=axes[0,col].imshow(matrix,cmap='RdBu_r',vmin=-2,vmax=2)
            y,x=np.where(sig);axes[0,col].scatter(x,y,s=5,c='#242424')
            axes[0,col].set(title=f'{niche} · spatial contact enrichment',xticks=range(n),xticklabels=types,yticks=range(n),yticklabels=types)
            local=ccc[ccc.niche==niche].pivot(index='sender',columns='receiver',values='relative_CCC').reindex(index=types,columns=types).fillna(0)
            im2=axes[1,col].imshow(local*100,cmap='BuPu',vmin=0,vmax=float(ccc.relative_CCC.max()*100))
            axes[1,col].set(title=f'{niche} · directed observed CCC',xticks=range(n),xticklabels=types,yticks=range(n),yticklabels=types,xlabel='Receiver',ylabel='Sender')
            for ax in axes[:,col]:
                plt.setp(ax.get_xticklabels(),rotation=60,ha='right',fontsize=7);ax.tick_params(axis='y',labelsize=7)
        fig.colorbar(im,ax=list(axes[0]),shrink=.7,label='log₂ observed / spatial-shuffle expectation')
        fig.colorbar(im2,ax=list(axes[1]),shrink=.7,label='% of niche CCC mass')
        self.letters(axes)
        fig.suptitle('Local cell arrangement and molecular communication are separate evidence',fontsize=13)
        return self.save(fig,'downstream_01_contacts_and_CCC')

    def _spatial(self,ax,rows,colors,title,size=4):
        xy=self.xy[rows];ax.scatter(xy[:,0],xy[:,1],c=colors,s=size,lw=0,rasterized=True)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_visible(False)
        span=np.ptp(xy,axis=0);x=xy[:,0].min();y=xy[:,1].max()+span.max()*.05
        ax.plot([x,x+100],[y,y],c='#333333',lw=1.5);ax.text(x+50,y+span.max()*.03,'100 µm',ha='center',va='top',fontsize=8)
        ax.set_ylim(y+span.max()*.11,xy[:,1].min()-span.max()*.03)

    def r2_mechanism(self):
        from sklearn.neighbors import NearestNeighbors
        fig,axes=plt.subplots(2,2,figsize=(12,9.5),layout='constrained')
        case=self.valid&self.cells.condition.eq('DKD').to_numpy();r2=case&self.cells.label.eq('R2').to_numpy();rows=np.flatnonzero(case)
        self._spatial(axes[0,0],rows,np.where(r2[rows],self.colors['R2'],'#E6E8EB'),'R2 within the DKD tissue',2)
        # Densest R2 location is chosen by coordinates only, without LR values.
        rrows=np.flatnonzero(r2);dist,_=NearestNeighbors(n_neighbors=min(40,len(rrows))).fit(self.xy[rrows]).kneighbors()
        center=self.xy[rrows[np.argmin(dist.mean(1))]];radius=200.
        ids=np.flatnonzero(case&(np.max(np.abs(self.xy-center),axis=1)<=radius));palette={'TAL':'#558DA8','Macro':'#CA8644','CD8+':'#8770B0'}
        colors=[palette.get(t,'#E0E3E7') for t in self.cells.cell_type.iloc[ids]]
        self._spatial(axes[0,1],ids,colors,'R2 local cells · actual physical geometry',9)
        # Arrow candidates require both measured ligand and receptor expression.
        edge=np.load(self.downstream/'R2_physical_contacts.npz');u=edge['u'];v=edge['v'];w=edge['weight']
        u0=np.r_[u,v];v0=np.r_[v,u];w0=np.r_[w,w]
        types=self.cells.cell_type.to_numpy();inside=np.zeros(len(self.cells),bool);inside[ids]=True
        keep=inside[u0]&inside[v0]&(types[u0]=='TAL')&(types[v0]=='CD8+')
        u0,v0,w0=u0[keep],v0[keep],w0[keep]
        lig=self.genes.get_loc('CXCL14');rec=self.genes.get_loc('CXCR4')
        strength=w0*np.sqrt(np.expm1(self.rna[u0,lig].toarray().ravel())*np.expm1(self.rna[v0,rec].toarray().ravel()))
        chosen=np.flatnonzero(strength>0);chosen=chosen[np.argsort(-strength[chosen])[:25]]
        for k in chosen:
            axes[0,1].annotate('',xy=self.xy[v0[k]],xytext=self.xy[u0[k]],arrowprops={'arrowstyle':'->','color':'#8F315F','lw':.7,'alpha':.6})
        axes[0,1].legend(handles=[Line2D([],[],ls='',marker='o',color=c,label=t,markersize=4) for t,c in palette.items()],fontsize=8,loc='upper right')
        axes[0,1].text(.02,.02,f'{len(chosen)} CXCL14–CXCR4 proximity candidates',transform=axes[0,1].transAxes,fontsize=8)
        pd.DataFrame({'sender_row':u0[chosen],'receiver_row':v0[chosen],'weighted_RNA_score':strength[chosen]}).to_csv(self.downstream/'R2_zoom_candidate_edges.csv',index=False)
        axes[0,0].plot([center[0]-radius,center[0]+radius,center[0]+radius,center[0]-radius,center[0]-radius],
            [center[1]-radius,center[1]-radius,center[1]+radius,center[1]+radius,center[1]-radius],c='#333333',lw=.8)
        table=pd.read_csv(self.downstream/'R2_same_sender_receiver_CCC.csv');picks=[]
        for receiver,g in table.groupby('receiver'):
            s=g.reindex(g.within_pair_difference.abs().sort_values(ascending=False).index).head(5);picks.append(s)
        selected=pd.concat(picks).sort_values(['receiver','within_pair_difference']);y=np.arange(len(selected))
        axes[1,0].barh(y-.16,selected.control_within_pair_fraction*100,.32,color='#86A8BA',label='Matched healthy')
        axes[1,0].barh(y+.16,selected.case_within_pair_fraction*100,.32,color=self.colors['R2'],label='R2')
        axes[1,0].set(yticks=y,yticklabels=[f'TAL → {r.receiver} · {r.LR}' for r in selected.itertuples()],xlabel='% of CCC within same sender–receiver pair',title='LR spectrum · original observed CCC')
        axes[1,0].tick_params(axis='y',labelsize=8);axes[1,0].legend(fontsize=8)
        selected.to_csv(self.downstream/'R2_displayed_LR_spectra.csv',index=False)
        genes=['CXCL14','APP','MIF','CXCR4','CD74'];groups=[];values=[]
        for typ in ['TAL','Macro','CD8+']:
            for name,mask in [('Healthy',self.valid&self.cells.condition.eq('HC').to_numpy()),('DKD shared',case&self.cells.label.str.startswith('B').to_numpy()),('DKD R2',r2)]:
                take=mask&self.cells.cell_type.eq(typ).to_numpy();rows=np.flatnonzero(take);x=self.rna[rows][:,self.genes.get_indexer(genes)]
                groups.append(f'{typ} · {name} (n={len(rows)})')
                values.append(np.asarray(x.mean(0)).ravel())
        values=np.asarray(values);im=axes[1,1].imshow(values,cmap='BuPu',aspect='auto',vmin=0,vmax=max(values.max(),.1))
        axes[1,1].set(xticks=range(len(genes)),xticklabels=genes,yticks=range(len(groups)),yticklabels=groups,title='Observed ligand / receptor RNA · same cell types')
        axes[1,1].tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=axes[1,1],shrink=.7,label='Mean log1p(TP10K)')
        pd.DataFrame(values,index=groups,columns=genes).to_csv(self.downstream/'R2_observed_LR_gene_means.csv')
        self.letters(axes);return self.save(fig,'downstream_02_R2_communication_mechanism')

    def tal_states(self):
        frame=pd.read_csv(self.downstream/'TAL_expression_state_map.csv.gz');fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
        palette={'Healthy shared':'#B8C3CA','DKD shared':'#5A9B95','DKD R2':self.colors['R2'],'DKD other R':'#D9AC6F'}
        for name,color in palette.items():
            sub=frame[frame.expression_group==name];axes[0,0].scatter(sub.UMAP1,sub.UMAP2,c=color,s=4,lw=0,label=f'{name} (n={len(sub)})',rasterized=True)
        axes[0,0].legend(fontsize=8);axes[0,0].set(title='TAL expression states · niches overlaid')
        for typ,color in [('TAL','#B8C3CA'),('iTAL','#B0635A')]:
            sub=frame[frame.cell_type_fine==typ];axes[0,1].scatter(sub.UMAP1,sub.UMAP2,c=color,s=4,lw=0,label=f'{typ} (n={len(sub)})',rasterized=True)
        axes[0,1].legend(fontsize=8);axes[0,1].set(title='Same embedding · original injury annotation')
        for ax in axes[0]:ax.set(xticks=[],yticks=[],xlabel='Expression UMAP 1',ylabel='Expression UMAP 2')
        effects=pd.read_csv(self.downstream/'same_cell_type_gene_effects.csv.gz');e=effects[(effects.niche=='R2')&(effects.cell_type=='TAL')]
        genes=[g for g in ['SLC12A1','UMOD','LCN2','SPP1','VCAM1','HAVCR1','VIM'] if g in self.genes]
        genes+= [g for g in e.query('niche_detection >= .2').nlargest(6,'log2FC').gene if g not in genes];genes=genes[:12]
        values=[];names=[]
        for name in ['Healthy shared','DKD shared','DKD R2']:
            rows=frame.loc[frame.expression_group==name,'source_row'].to_numpy(int);values.append(np.asarray(self.rna[rows][:,self.genes.get_indexer(genes)].mean(0)).ravel());names.append(f'{name} (n={len(rows)})')
        values=np.asarray(values);scaled=values/np.maximum(values.max(0),1e-8)
        im=axes[1,0].imshow(scaled,cmap='BuPu',vmin=0,vmax=1,aspect='auto');axes[1,0].set(xticks=range(len(genes)),xticklabels=genes,yticks=range(3),yticklabels=names,title='Observed TAL RNA · per-gene relative mean')
        plt.setp(axes[1,0].get_xticklabels(),rotation=60,ha='right');fig.colorbar(im,ax=axes[1,0],shrink=.6,label='Relative mean log1p(TP10K)')
        pd.DataFrame(values,index=names,columns=genes).to_csv(self.downstream/'TAL_state_marker_means.csv')
        state=pd.read_csv(self.downstream/'niche_local_and_whole_injury_states.csv');state=state[state.parent_type=='TAL']
        axes[1,1].bar(range(len(state)),state.injured_fraction*100,color=['#B8C3CA','#D9AC6F','#5A9B95',self.colors['R2']])
        axes[1,1].set(xticks=range(len(state)),xticklabels=state.group,ylabel='% iTAL among TAL cells',title='Injury state · niche-local versus whole tissue')
        plt.setp(axes[1,1].get_xticklabels(),rotation=25,ha='right',fontsize=8)
        for j,r in enumerate(state.itertuples()):axes[1,1].text(j,r.injured_fraction*100+1,f'{r.injured_cells}/{r.parent_cells}',ha='center',fontsize=8)
        axes[1,1].margins(y=.2);self.letters(axes);return self.save(fig,'downstream_03_TAL_cell_states')

    def pathways(self):
        from .kidney_downstream import hallmark_sets
        table=pd.read_csv(self.downstream/'hallmark_panel_enrichment.csv')
        sets,_=hallmark_sets(Path(json.loads((self.downstream/'audit.json').read_text())['gene_sets']).parent)
        effects=pd.read_csv(self.downstream/'same_cell_type_gene_effects.csv.gz')
        means=[]
        for r in table.itertuples():
            niche,parent=r.context.split(' · ')
            e=effects[(effects.niche==niche)&(effects.cell_type==parent)&effects.universe&effects.gene.isin(sets[r.pathway])]
            means.append(float(e.log2FC.mean()))
        table['mean_measured_gene_log2FC']=means
        table.to_csv(self.downstream/'hallmark_expression_effects.csv',index=False)
        contexts=table.context.unique();terms=[]
        for ctx in contexts:
            g=table[(table.context==ctx)&(table.covered_genes>=10)].sort_values('mean_measured_gene_log2FC',ascending=False)
            terms.extend(g.head(2).pathway)
        terms=list(dict.fromkeys(terms));shown=table[table.pathway.isin(terms)].copy()
        shown.to_csv(self.downstream/'displayed_pathway_enrichment.csv',index=False)
        fig,ax=plt.subplots(figsize=(10,max(4.5,.38*len(terms)+2)),layout='constrained')
        limit=max(float(shown.mean_measured_gene_log2FC.abs().max()),.1)
        for r in shown.itertuples():
            im=ax.scatter(list(contexts).index(r.context),terms.index(r.pathway),s=15+r.covered_genes*3,c=[r.mean_measured_gene_log2FC],cmap='RdBu_r',vmin=-limit,vmax=limit,edgecolors='#222222' if r.q_gene_overlap<.05 else '#BBBBBB',linewidths=1.2 if r.q_gene_overlap<.05 else .3)
        ax.set(xticks=range(len(contexts)),xticklabels=contexts,yticks=range(len(terms)),yticklabels=[t.replace('HALLMARK_','').replace('_',' ') for t in terms],title='Hallmark modules · observed same-cell-type expression shifts',xlim=(-.6,len(contexts)-.4))
        ax.invert_yaxis();plt.setp(ax.get_xticklabels(),rotation=25,ha='right');fig.colorbar(im,ax=ax,shrink=.6,label='Mean measured-gene log₂FC\nversus same-type DKD shared niches')
        handles=[ax.scatter([],[],s=15+n*3,facecolor='#CDC0D9',edgecolor='#888888',label=f'{n} genes') for n in [10,30,60]]
        ax.legend(handles=handles,title='Measured genes in module',loc='upper left',bbox_to_anchor=(1.13,.35),fontsize=8)
        if not (table.q_gene_overlap<.05).any():
            fig.supxlabel('Descriptive expression shifts; no gene-overlap enrichment passes BH q < 0.05.',fontsize=8)
        return self.save(fig,'downstream_04_cell_state_pathways')

    def top_genes(self):
        table=pd.read_csv(self.downstream/'same_cell_type_gene_effects.csv.gz')
        table=table[(table.niche=='R2')&(table.cell_type=='TAL')&(table.niche_detection>=.2)].nlargest(10,'log2FC')
        return table[['gene','log2FC','niche_detection','background_detection']].round(3)
