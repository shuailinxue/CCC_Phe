"""Four explicitly planted MI programs, and actual GNN recovery."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .generation import TYPES
from ..gnn.reporting import COLORS

REGIONS=['Tumor core','Normal exterior','Edge A · MI-1','Edge B · MI-2']
SHORT_TYPES=['Tumor','Normal','Fibro.','Macro.','T cell']


def regional_profiles(report):
    """Descriptive regional means; never treated as four planted MI dictionaries."""
    ccc=np.load(report.output/'input/ccc.npy',mmap_mode='r')
    composition=np.load(report.output/'input/composition_target.npy',mmap_mode='r')
    means=np.stack([np.asarray(ccc[report.truth==i]).mean(0) for i in range(4)])
    fractions=np.stack([np.asarray(composition[report.truth==i]).mean(0) for i in range(4)])
    selected=list(dict.fromkeys(int(j) for row in means for j in np.argsort(-row,kind='stable')[:3]))
    return fractions,means,selected


def summary_figure(report):
    r=report;sources=r.output/'report_tables';sources.mkdir(exist_ok=True)
    fractions,means,selected=regional_profiles(r)
    specs=r.mi_specs;true_h=np.load(r.output/'ground_truth/MI_dictionary.npy');order=np.argsort(specs.region.to_numpy())
    program_fractions=pd.read_csv(r.output/'ground_truth/MI_composition.csv',index_col=0).to_numpy()[order]
    names=[f'MI-{int(specs.MI.iloc[i])} · {specs.name.iloc[i]}' for i in order]
    chosen=list(dict.fromkeys(int(j) for i in order for j in np.flatnonzero(true_h[i])[:3]))
    pd.DataFrame(fractions,index=REGIONS,columns=TYPES).to_csv(sources/'summary_regional_composition.csv')
    pd.DataFrame(means,index=REGIONS,columns=r.features.ccc).to_csv(sources/'summary_regional_CCC.csv')
    r.features.iloc[selected].to_csv(sources/'summary_regional_selected_features.csv',index=False)
    fig=plt.figure(figsize=(11.5,8.2))
    for letter,x,y in [('a',.02,.94),('c',.345,.94),('e',.345,.52),('f',.02,.52)]:
        fig.text(x,y,letter,fontweight='bold',fontsize=12)
    fig.suptitle('CCC_Phe | Spatial cell types, regional communication and recovered niches',fontsize=12,y=.985)
    # Identical dimensions and coordinate limits for the two compact spatial maps.
    spatial_axes=[]
    for bottom,title,values in [(.645,'Cell types',pd.Categorical(r.cells.cell_type,categories=TYPES).codes),(.225,'Recovered niches · raw K=4',r.labels['CCC_Phe GNN (fixed K)'].to_numpy())]:
        ax=fig.add_axes([.065,bottom,.20,.28]);r.spatial(ax,values,title,True);ax.tick_params(labelsize=7);ax.title.set_fontsize(9);ax.set_xlabel('x (µm)',fontsize=8);ax.set_ylabel('y (µm)',fontsize=8);spatial_axes.append(ax)
    a,f=spatial_axes
    offset=float(np.median(r.xy[r.show,0]-r.cells.local_x.to_numpy()[r.show]))
    for radius in [r.cfg['core_radius_um'],r.cfg['core_radius_um']+r.cfg['edge_width_um']]:
        a.add_patch(Circle((offset,0),radius,fill=False,ec='#444444',ls='--',lw=.6))
    a.legend(handles=[plt.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=3,label=name) for i,name in enumerate(TYPES)],loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=7)
    labels=r.labels['CCC_Phe GNN (fixed K)'].to_numpy()
    f.legend(handles=[plt.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=3,label=f'Cluster {i}') for i in np.unique(labels)],loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,fontsize=7)
    score=r.metrics[(r.metrics.method=='CCC_Phe GNN (fixed K)')&(r.metrics.evaluation=='heldout_test')].iloc[0]
    f.text(.5,-.44,f'Test ARI {score.overall_ARI:.3f} · Edge F1 {score.edge_F1:.3f}\nEdge A/B accuracy {score.edge_AB_accuracy:.1%}',ha='center',transform=f.transAxes,fontsize=7)
    # Four measured regional communication profiles, not four invented MI labels.
    comp=fig.add_axes([.44,.765,.145,.14]);left=np.zeros(4)
    for i in range(5):
        comp.barh(range(4),program_fractions[:,i],left=left,color=COLORS[i],height=.7,lw=0);left+=program_fractions[:,i]
    comp.set(ylim=(3.5,-.5),xlim=(0,1));comp.set_yticks(range(4),names,fontsize=8);comp.set_xticks([0,.5,1],['0','50','100'],fontsize=7);comp.set_xlabel('Activity-weighted composition (%)',fontsize=8);comp.set_title('Cell-type composition',fontsize=9,pad=10)
    for spine in ['left','right','top']:comp.spines[spine].set_visible(False)
    ax=fig.add_axes([.635,.765,.315,.14]);im=ax.imshow(true_h[order][:,chosen],cmap='YlGnBu',vmin=0,vmax=float(true_h.max()),aspect='auto');ax.set_yticks(range(4),['']*4);ax.tick_params(axis='y',length=0)
    short={t:s for t,s in zip(TYPES,SHORT_TYPES)}
    features=r.features.iloc[chosen]
    feature_labels=[f'{short[v.sender]}→{short[v.receiver]} | {v.ligand}–{v.receptor}' for v in features.itertuples()]
    ax.set_xticks(range(len(chosen)),feature_labels,rotation=65,ha='right',fontsize=6.5);ax.set_title('Predefined MI dictionary · representative LR',fontsize=9,pad=10)
    bar=fig.colorbar(im,cax=fig.add_axes([.963,.765,.010,.14]));bar.ax.tick_params(labelsize=7);bar.set_label('True H loading',fontsize=7)
    fig.text(.44,.555,'Four predeclared programs, six LR pairs each; three LR per program shown above.\nComposition: physical-pair-weighted neighborhoods, weighted by true MI activity; not a training target.',fontsize=7)
    # Compare all four predeclared dictionaries with learned H on one native scale.
    ids=[int(j) for i in order for j in np.flatnonzero(true_h[i])]
    matches=r.match[r.match.scope=='test_all'].set_index('MI').loc[specs.MI.iloc[order]];programs=matches.program.to_numpy(int)-1
    true_h=np.load(r.output/'ground_truth/MI_dictionary.npy');matrix=np.vstack((true_h[order][:,ids],r.H[programs][:,ids]));pair_labels=(r.features.iloc[ids].ligand+'–'+r.features.iloc[ids].receptor).tolist()
    ax=fig.add_axes([.46,.225,.49,.28]);image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=float(matrix.max()),aspect='auto');ax.set_title('Four planted MIs → recovered CCC dictionary',fontsize=9,pad=10)
    rows=[f'True MI-{int(specs.MI.iloc[i])}' for i in order]+[f'Learned P{p+1:02d}' for p in programs];ax.set_yticks(range(8),rows,fontsize=8);ax.set_xticks(range(len(ids)),pair_labels,rotation=65,ha='right',fontsize=6)
    ax.set_xlabel('Core: Tumor→Tumor | Normal: Normal→Normal | Edge A/B: Tumor→Fibroblast',fontsize=7)
    bar=fig.colorbar(image,cax=fig.add_axes([.963,.225,.010,.28]));bar.ax.tick_params(labelsize=7);bar.set_label('Native H loading',fontsize=7)
    pd.DataFrame(matrix,index=rows,columns=pair_labels).to_csv(sources/'summary_dictionary_loadings.csv')
    fig.text(.5,.015,'One synthetic tissue. Spatial maps and regional profiles: all cells; numerical niche metrics: held-out spatial test.\nFour explicit planted programs coexist with basal background; boundary neighborhoods can mix programs. No composition loss.',ha='center',fontsize=7)
    (sources/'summary_figure_metadata.json').write_text(json.dumps({'n_planted_MIs':len(true_h),'n_regions':4,'regional_profiles':'Observed regional CCC saved for audit; c shows predeclared MI dictionary and physical-neighborhood composition weighted by known MI activity','feature_selection':'Three predefined directed LR features per MI shown in c; all six per MI in e; no observed labels used to define dictionaries','program_match':'Four predeclared MI dictionaries matched to H only','dictionary_values':'Native loadings on one shared scale','removed_panels':['b','d'],'niche_metric_split':'heldout_test'},indent=2)+'\n')
    return r.save(fig,'simulation_design_and_recovery_summary')
