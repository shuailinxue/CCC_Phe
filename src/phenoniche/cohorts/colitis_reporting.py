"""Compact Python figures for observed colitis communication niches.

Contract: show the tissue locations, patient-level abundance and directed LR
content of learned niches, without treating colocalization as causal evidence.
Evidence: held-out spatial maps; nine patient fractions; Z/H interpretation;
CCC training/reconstruction. Quantitative grid plus tissue-image hero panels.
PNG (450 dpi), editable PDF/SVG and source CSVs are exported on NAS. No labels
or idealized outcomes are fabricated; no cell-level tests substitute for n=9.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from .colitis import CONDITIONS

GROUP_COLORS={'HC':'#3976AC','UC':'#D66A58','ICI':'#8D6DAF'}
ACTIVITY=LinearSegmentedColormap.from_list('colitis_activity',['#EEF0F2','#B6A7CD','#745190','#39134D'])

def packed_coordinates(frame,gap_um=70.):
    """Translate disconnected fragments into compact shelves, never deform them."""
    xy=frame[['x','y']].to_numpy(dtype=float).copy();cores=frame.core.to_numpy();boxes=[]
    for core in sorted(frame.core.unique()):
        ids=np.flatnonzero(cores==core);lower=xy[ids].min(0);span=np.ptp(xy[ids],axis=0)
        boxes.append((core,ids,lower,np.maximum(span,1.)))
    boxes.sort(key=lambda b:(-b[3][1],str(b[0])))
    width=max(max(b[3][0] for b in boxes),1.25*np.sqrt(sum(np.prod(b[3]+gap_um) for b in boxes)))
    left=top=row_height=0.;labels=[]
    for core,ids,lower,span in boxes:
        if left and left+span[0]>width:
            left=0.;top+=row_height+gap_um;row_height=0.
        xy[ids]+=np.array([left,top])-lower
        labels.append((str(core).split('_fragment')[-1],left+span[0]/2,top))
        left+=span[0]+gap_um;row_height=max(row_height,span[1])
    return xy,labels

class ColitisReport:
    def __init__(self,output):
        self.output=Path(output);self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)
        self.summary=json.loads((self.output/'summary.json').read_text());self.k=self.summary['selected_K']
        self.cells=pd.read_csv(self.output/'assignments.csv.gz');self.types=self.summary['cell_types']
        self.z=np.load(self.output/'gnn/Z.npy',mmap_mode='r');self.h=np.load(self.output/'gnn/H.npy')
        self.features=pd.read_csv(self.output/'input/features.csv')
        self.freq=pd.read_csv(self.output/'patient_niche_fractions.csv')
        self.patient=self.cells[['patient','condition','fold']].drop_duplicates().sort_values(['condition','patient'])
        palette=list(plt.get_cmap('tab20').colors)
        self.niche_colors=[palette[i] for i in list(range(0,20,2))+list(range(1,20,2))][:self.k]
        self.type_colors=list(plt.get_cmap('tab20').colors)[:len(self.types)]
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
            'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
            'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'figure.dpi':120})

    def save(self,fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(self.figures/(name+'.'+ext),dpi=450,bbox_inches='tight',pad_inches=.12)
        return fig

    @staticmethod
    def short(patient):
        return next((s for s in str(patient).split('_') if s.startswith('HS')),str(patient))

    def representative(self,condition,heldout=True):
        if self.summary.get('split_unit')=='joint_all':
            f=self.cells[self.cells.condition==condition]
            patient=f.groupby('patient').size().idxmax()
            return np.flatnonzero(self.cells.patient.to_numpy()==patient)
        f=self.cells[(self.cells.condition==condition)&((self.cells.fold==2) if heldout else True)]
        core=f.groupby('core').size().idxmax()
        return np.flatnonzero(self.cells.core.to_numpy()==core)

    def tissue(self,ax,rows,color,title,palette=None,cmap=None,vmax=None,compact=False):
        if compact:xy,labels=packed_coordinates(self.cells.iloc[rows])
        else:xy=self.cells.iloc[rows][['x','y']].to_numpy(dtype=float);labels=[]
        center=xy.mean(0);xy-=center
        if palette is not None:color=np.asarray(palette)[np.asarray(color,dtype=int)]
        im=ax.scatter(xy[:,0],xy[:,1],c=color,s=2.5,linewidths=0,rasterized=True,cmap=cmap,vmin=0 if cmap else None,vmax=vmax)
        ax.set_aspect('equal');ax.set_title(title,pad=8);ax.set_xticks([]);ax.set_yticks([])
        for label,x,y in labels:ax.text(x-center[0],y-center[1]-20,f'F{label}',ha='center',va='bottom',fontsize=7,color='#555555')
        for spine in ax.spines.values():spine.set_visible(False)
        span=max(np.ptp(xy,axis=0));x0=xy[:,0].min();y0=xy[:,1].max()+.055*span
        ax.plot([x0,x0+100],[y0,y0],color='#333333',lw=1.5);ax.text(x0+50,y0+.025*span,'100 µm',ha='center',va='top',fontsize=8)
        ax.set_ylim(xy[:,1].max()+.15*span,xy[:,1].min()-.035*span)
        return im

    def spatial(self,condition=None):
        """All retained fragments per patient; paired panels use identical cells."""
        if condition is None:groups=CONDITIONS
        elif condition in CONDITIONS:groups=(condition,)
        else:raise ValueError('Unknown condition: '+str(condition))
        fig,axes=plt.subplots(2*len(groups),3,figsize=(13,7.3*len(groups)+1),squeeze=False)
        fig.subplots_adjust(left=.02,right=.99,top=.94,bottom=.14 if len(groups)==1 else .06,hspace=.23,wspace=.12)
        manifest=[]
        for block,group in enumerate(groups):
            patients=sorted(self.cells.loc[self.cells.condition==group,'patient'].unique(),key=self.short)
            for col,patient in enumerate(patients):
                ids=np.flatnonzero(self.cells.patient.to_numpy()==patient);local=self.cells.iloc[ids]
                title=f'{group} · {self.short(patient)} · {len(ids):,} cells · {local.core.nunique()} fragments'
                types=pd.Categorical(local.cell_type,categories=self.types).codes
                self.tissue(axes[2*block,col],ids,types,title,self.type_colors,compact=True)
                self.tissue(axes[2*block+1,col],ids,local.niche.to_numpy()-1,f'{self.short(patient)} · niches · same cells',self.niche_colors,compact=True)
                manifest.extend({'patient':patient,'condition':group,'core':core,'cells':len(rows)} for core,rows in local.groupby('core'))
        types=[Line2D([],[],marker='o',ls='',color=self.type_colors[i],label=t,markersize=4) for i,t in enumerate(self.types)]
        niches=[Line2D([],[],marker='o',ls='',color=self.niche_colors[i],label=f'N{i+1}',markersize=4) for i in range(self.k)]
        fig.legend(handles=types,loc='lower center',bbox_to_anchor=(.5,.055 if len(groups)==1 else .025),ncol=7,fontsize=8,columnspacing=1,handletextpad=.3)
        fig.legend(handles=niches,loc='lower center',bbox_to_anchor=(.5,.015 if len(groups)==1 else .005),ncol=self.k,fontsize=8,columnspacing=1,handletextpad=.3)
        fig.suptitle('Cell types and communication niches · all retained tissue',fontsize=13,y=.99)
        fig.text(.5,.105 if len(groups)==1 else .05,'Disconnected fragments rearranged for display; shapes and µm scale preserved.',ha='center',fontsize=8,color='#555555')
        name='all_patient_spatial_niches'+('' if condition is None else '_'+condition)
        pd.DataFrame(manifest).to_csv(self.figures/(name+'_fragments.csv'),index=False)
        return self.save(fig,name)

    def groups(self):
        freq=self.freq.copy();order=[]
        for g in CONDITIONS:order.extend(freq.loc[freq.condition==g,'patient'].sort_values().tolist())
        matrix=freq.set_index('patient').loc[order,[str(i) for i in range(1,self.k+1)]]
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
        im=axes[0].imshow(matrix.to_numpy()*100,cmap='YlGnBu',aspect='auto',vmin=0)
        axes[0].set(xticks=np.arange(self.k),xticklabels=[f'N{i+1}' for i in range(self.k)],yticks=np.arange(9),yticklabels=[self.short(p) for p in order])
        for label,p in zip(axes[0].get_yticklabels(),order):label.set_color(GROUP_COLORS[freq.set_index('patient').loc[p,'condition']])
        axes[0].set_title('Niche abundance per patient');fig.colorbar(im,ax=axes[0],shrink=.7,label='% of patient cells')
        for j,g in enumerate(CONDITIONS):
            local=freq[freq.condition==g];xs=np.arange(self.k)+(j-1)*.22
            for _,r in local.iterrows():
                axes[1].scatter(xs,r[[str(i) for i in range(1,self.k+1)]].to_numpy(float),c=GROUP_COLORS[g],s=26,alpha=.7,marker={-1:'o',0:'o',1:'^',2:'D'}[int(r.fold)])
            axes[1].plot(xs,local[[str(i) for i in range(1,self.k+1)]].mean(0),c=GROUP_COLORS[g],lw=1,marker='_',ms=13,label=g)
        axes[1].set(xticks=np.arange(self.k),xticklabels=[f'N{i+1}' for i in range(self.k)],ylabel='Fraction of patient cells')
        axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_title('Each point is one patient · n=3 per group');axes[1].legend(ncol=3)
        return self.save(fig,'patient_niche_abundance')

    def patient_atlas(self):
        return self.spatial()

    def niche_heatmaps(self):
        programs=pd.read_csv(self.output/'niche_program_activity.csv',index_col=0).to_numpy()
        values=np.log1p(programs);values=(values-values.mean(0))/np.maximum(values.std(0),.05)
        comp=np.load(self.output/'niche_composition.npy')
        fig,axes=plt.subplots(1,2,figsize=(14,4.7),layout='constrained',gridspec_kw={'width_ratios':[1.8,1]})
        im=axes[0].imshow(values,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
        axes[0].set(xticks=np.arange(32),xticklabels=np.arange(1,33),yticks=np.arange(self.k),yticklabels=[f'N{i+1}' for i in range(self.k)],xlabel='Communication program')
        axes[0].tick_params(axis='x',labelsize=7);axes[0].set_title('Niche × program activity');fig.colorbar(im,ax=axes[0],shrink=.7,label='Column z score of log1p mean Z')
        im=axes[1].imshow(comp*100,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
        axes[1].set(xticks=np.arange(len(self.types)),xticklabels=self.types,yticks=np.arange(self.k),yticklabels=[f'N{i+1}' for i in range(self.k)])
        plt.setp(axes[1].get_xticklabels(),rotation=60,ha='right',fontsize=8)
        for i,j in zip(*np.where(comp>=.1)):axes[1].text(j,i,f'{comp[i,j]*100:.0f}',ha='center',va='center',fontsize=7,color='white' if comp[i,j]>.5 else '#333333')
        axes[1].set_title('Spatially weighted cell-type composition');fig.colorbar(im,ax=axes[1],shrink=.7,label='%')
        return self.save(fig,'niche_program_and_composition')

    def dictionary(self):
        cols=np.unique(np.argsort(-self.h,axis=1)[:,:2]);f=self.features.iloc[cols]
        values=self.h[:,cols]/np.maximum(self.h.max(1,keepdims=True),1e-12)
        labels=[f'{r.sender}→{r.receiver} | {r.ligand}–{r.receptor}' for r in f.itertuples()]
        fig,axes=plt.subplots(1,2,figsize=(14,9.5),gridspec_kw={'width_ratios':[.8,5]},layout='constrained')
        usage=np.asarray(self.z).mean(0);usage/=max(usage.sum(),1e-12)
        axes[0].barh(np.arange(32),usage,color='#699AAB');axes[0].invert_yaxis();axes[0].set(yticks=np.arange(32),yticklabels=[f'P{i+1}' for i in range(32)],xlabel='Mean Z share');axes[0].set_ylim(31.5,-.5)
        im=axes[1].imshow(values,aspect='auto',cmap=ACTIVITY,vmin=0,vmax=1)
        axes[1].set(yticks=np.arange(32),yticklabels=[f'P{i+1}' for i in range(32)],xticks=np.arange(len(cols)),xticklabels=labels)
        plt.setp(axes[1].get_xticklabels(),rotation=70,ha='right',fontsize=7)
        axes[1].tick_params(axis='y',labelsize=8);axes[1].set_title('Nonnegative CCC dictionary H · representative features')
        fig.colorbar(im,ax=axes[1],shrink=.5,label='Loading / program maximum')
        return self.save(fig,'program_dictionary_H')

    def program_spatial(self):
        means=pd.read_csv(self.output/'patient_program_activity.csv').groupby('condition')[[str(i) for i in range(32)]].mean()
        selected=[]
        for group in ('UC','ICI'):
            rank=np.argsort(-(means.loc[group]-means.loc['HC']).to_numpy())
            selected.append(next(int(j) for j in rank if int(j) not in selected))
        top=pd.read_csv(self.output/'program_top_CCC.csv');fig,axes=plt.subplots(2,3,figsize=(11.5,7.4),layout='constrained')
        for row,program in enumerate(selected):
            v=np.log1p(self.z[:,program]);vmax=max(float(np.quantile(v,.99)),1e-6)
            for col,group in enumerate(CONDITIONS):
                ids=self.representative(group)
                patient=self.short(self.cells.iloc[ids[0]].patient)
                im=self.tissue(axes[row,col],ids,v[ids],f'{group} · {patient} · P{program+1}',cmap=ACTIVITY,vmax=vmax,compact=self.summary.get('split_unit')=='joint_all')
            r=top[top.program==program+1].iloc[0]
            fig.colorbar(im,ax=axes[row,:],shrink=.6,label='log1p Z')
            axes[row,0].set_ylabel(f'P{program+1}: {r.sender}→{r.receiver}\n{r.LR}',fontsize=9)
        return self.save(fig,'disease_program_spatial_activity')

    def training(self):
        history=pd.read_csv(self.output/'gnn/training.csv');selection=pd.read_csv(self.output/'gnn/K_selection.csv')
        joint=self.summary.get('split_unit')=='joint_all'
        best=json.loads((self.output/('gnn/training_monitor_metrics.json' if joint else 'gnn/heldout_metrics.json')).read_text())['best_epoch']
        fig,axes=plt.subplots(1,3,figsize=(12.8,3.6),layout='constrained')
        axes[0].plot(history.epoch,history.train_ccc,color='#3976AC',label='Training tissue')
        axes[0].plot(history.epoch,history['monitor_ccc' if joint else 'validation_ccc'],color='#D66A58',label='Training-subset monitor' if joint else 'Validation tissue')
        axes[0].axvline(best,color='#777777',ls=':',lw=1);axes[0].set(xlabel='Epoch',ylabel='Scaled CCC Huber',title=f'Training · best epoch {best}');axes[0].legend(fontsize=8)
        axes[1].plot(selection.K,selection.selection_score,'o-',color='#745190');axes[1].axvline(self.k,ls=':',color='#777777');axes[1].set(xlabel='K',ylabel='Internal selection score' if joint else 'Validation selection score',title=f'Niche selection · K={self.k}',xticks=selection.K)
        rng=np.random.default_rng(7);rows=np.arange(len(self.cells)) if joint else np.flatnonzero(self.cells.fold.to_numpy()==2);rows=rng.choice(rows,min(2000,len(rows)),replace=False)
        target=np.load(self.output/'input/ccc.npy',mmap_mode='r');prediction=np.load(self.output/'ccc_reconstruction.npy',mmap_mode='r')
        mask=np.load(self.output/'input/preprocessing.npz')['feature_mask'];left=np.log1p(np.asarray(target[rows])[:,mask]).ravel();right=np.log1p(np.asarray(prediction[rows])[:,mask]).ravel()
        ids=rng.choice(len(left),min(30000,len(left)),replace=False);axes[2].hexbin(left[ids],right[ids],gridsize=45,mincnt=1,cmap='Blues',bins='log',rasterized=True)
        lim=max(float(np.quantile(left,.995)),float(np.quantile(right,.995)),.1);axes[2].plot([0,lim],[0,lim],ls='--',color='#777777',lw=1)
        axes[2].set(xlabel='Observed log1p CCC',ylabel='Reconstructed log1p CCC',title='Cohort reconstruction' if joint else 'Held-out tissue reconstruction')
        return self.save(fig,'training_and_reconstruction')

    def overview(self):
        s=self.summary
        return pd.Series({'Patients':s['patients'],'Cells':s['cells'],'Genes':s['genes'],
            'Cell types':len(s['cell_types']),'Tissue fragments':s['fragments'],
            'Panel-measurable LR pairs':s['panel_measurable_LR'],'Retained LR pairs':s['retained_LR'],
            'Effective CCC features':s['effective_CCC_features'],'Encoder genes':s['encoder_gene_selection']['selected_genes'],
            'Communication programs':32,'Selected niches':self.k}).to_frame('Dataset and model')

    def representative_ccc(self):
        top=pd.read_csv(self.output/'niche_top10_CCC.csv');enrichment=pd.read_csv(self.output/'condition_niche_enrichment.csv')
        chosen=[]
        for group in ('UC','ICI'):
            e=enrichment[(enrichment.condition==group)&(enrichment.mean_patient_fraction>=.03)].sort_values('log2_ratio_vs_HC',ascending=False)
            if len(e):chosen.append(int(e.iloc[0].niche))
        return top[top.niche.isin(chosen)].round({'mean_reconstructed_CCC':3})
