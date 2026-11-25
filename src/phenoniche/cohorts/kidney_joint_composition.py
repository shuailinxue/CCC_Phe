"""Post-fit composition checks for the existing joint kidney niche labels."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from matplotlib.lines import Line2D


def selected_joint_composition(report, niches=(1, 6, 10)):
    """Compare center types and nearest-five healthy physical-neighborhood mixtures.

    Matching uses every retained healthy center, regardless of its niche label.
    Repeated reference neighbors retain their matching multiplicity. This is a
    descriptive two-section comparison, not a disease association test.
    """
    cells=report.cells
    valid=report.valid
    comp=np.load(report.input/'composition.npy',mmap_mode='r')
    if comp.shape!=(len(cells),len(report.types)):
        raise ValueError('Local composition rows/columns must align with report cells/types')
    healthy=np.flatnonzero(valid&(cells.condition.to_numpy()==report.control))
    tree=cKDTree(np.sqrt(np.maximum(comp[healthy],0)))
    mixtures=[]; summaries=[]
    for niche in niches:
        masks={g:valid&(cells.condition.to_numpy()==g)&(cells.niche.to_numpy()==niche)
               for g in (report.control,report.case)}
        for g,mask in masks.items():
            fractions=cells.loc[mask,'cell_type'].value_counts(normalize=True).reindex(report.types,fill_value=0)
            mixtures.append(dict(panel='Center cell types',niche=f'N{niche}',group=g,
                                 centers=int(mask.sum()),**fractions.to_dict()))
        case=np.flatnonzero(masks[report.case])
        if not len(case):
            raise ValueError(f'N{niche} has no retained case centers')
        distances,indices=tree.query(np.sqrt(np.maximum(comp[case],0)),k=5)
        matched=healthy[indices]
        hellinger=distances.mean(axis=1)/np.sqrt(2)
        for group,mean in [('DKD neighborhoods',comp[case].mean(axis=0)),
                           ('Nearest HC neighborhoods',comp[matched].mean(axis=(0,1)))]:
            mixtures.append(dict(panel='Physical-neighborhood composition',niche=f'N{niche}',
                                 group=group,centers=len(case),**dict(zip(report.types,mean))))
        counts={g:int(m.sum()) for g,m in masks.items()}
        nearest_niches=cells.iloc[matched.ravel()].niche.value_counts(normalize=True)
        summaries.append({'Niche':f'N{niche}',
            'HC centers':counts[report.control],'DKD centers':counts[report.case],
            'HC fraction (%)':100*counts[report.control]/len(healthy),
            'DKD fraction (%)':100*counts[report.case]/int((valid&(cells.condition.to_numpy()==report.case)).sum()),
            'Median composition distance':float(np.median(hellinger)),
            'DKD centers with close HC composition (%)':float(100*(hellinger<=.15).mean()),
            'Nearest HC niche labels':'; '.join(f'N{int(n)}: {v:.0%}' for n,v in nearest_niches.head(3).items())})
        pd.DataFrame({'case_row':case,'mean_5_HC_Hellinger_distance':hellinger,
                      'close_HC_composition':hellinger<=.15,
                      **{f'HC_row_{i+1}':matched[:,i] for i in range(5)}}).to_csv(
            report.analysis/f'joint_N{niche}_healthy_composition_matches.csv',index=False)
    source=pd.DataFrame(mixtures)
    source.to_csv(report.analysis/'joint_selected_cell_composition.csv',index=False)
    summary=pd.DataFrame(summaries)
    summary.to_csv(report.analysis/'joint_selected_healthy_composition_summary.csv',index=False)
    # Preserve fine author cell types; collapse only the low-abundance plot columns.
    selected=source[report.types].max().nlargest(10).index.tolist()
    plot=source[selected].copy()
    plot['Other']=source[[t for t in report.types if t not in selected]].sum(axis=1)
    colors=[report.type_colors[report.types.index(t)] for t in selected]+['#D5D8DB']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.9))
    fig.subplots_adjust(left=.16,right=.99,bottom=.25,top=.88,wspace=.5)
    for ax,panel,title in zip(axes,['Center cell types','Physical-neighborhood composition'],
            ['a  Cell types within each joint niche','b  Local composition and nearest healthy reference']):
        rows=source.index[source.panel==panel].to_numpy()
        y=np.arange(len(rows));left=np.zeros(len(rows))
        for cell_type,color in zip(plot.columns,colors):
            values=plot.loc[rows,cell_type].to_numpy()*100
            ax.barh(y,values,left=left,height=.7,color=color,label=cell_type)
            for pos,value,start in zip(y,values,left):
                if value>=12:ax.text(start+value/2,pos,f'{value:.0f}%',ha='center',va='center',fontsize=8)
            left+=values
        labels=[]
        for row in source.loc[rows].itertuples():
            if panel=='Center cell types':labels.append(f'{row.niche} · {row.group} (n={row.centers:,})')
            else:labels.append(f'{row.niche} · '+('DKD' if row.group=='DKD neighborhoods' else 'Nearest HC'))
        ax.set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Cell fraction (%)',title=title)
        ax.invert_yaxis();ax.tick_params(axis='y',length=0,labelsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=6,fontsize=8)
    fig.text(.5,.015,'Left: observed center cell types. Right: original distance-weighted physical-neighborhood composition.\n'
        'Each DKD center is matched to five HC centers across all HC niches; close match = mean Hellinger distance ≤ 0.15.\n'
        'Nearest HC references can still be dissimilar; see the accompanying match-distance table. One section per condition.',
        ha='center',fontsize=7)
    return report.save(fig,'joint_N1_N6_N10_cell_composition'),summary


def healthy_alternative_regions(report, niches=(6,10), tolerance=.15):
    """Locate HC neighborhoods similar to case niches but assigned other labels."""
    cells=report.cells;labels=cells.niche.to_numpy();groups=cells.condition.to_numpy()
    comp=np.load(report.input/'composition.npy',mmap_mode='r')
    members=np.load(report.input/'members.npy',mmap_mode='r')
    xy=cells[['x','y']].to_numpy(float)
    hc=np.flatnonzero(report.valid&(groups==report.control))
    dk=np.flatnonzero(report.valid&(groups==report.case))
    fig,axes=plt.subplots(len(niches),2,figsize=(9,4*len(niches)),squeeze=False)
    fig.subplots_adjust(left=.02,right=.99,top=.94,bottom=.14,wspace=.06,hspace=.17)
    summaries=[]
    for col,niche in enumerate(niches):
        case=dk[labels[dk]==niche];eligible=hc[labels[hc]!=niche]
        tree=cKDTree(np.sqrt(np.maximum(comp[case],0)))
        d,j=tree.query(np.sqrt(np.maximum(comp[eligible],0)),k=min(5,len(case)))
        distance=d.mean(axis=1)/np.sqrt(2)
        matched=eligible[distance<=tolerance]
        # Connectivity uses original physical graph edges, restricted to matched
        # centers and the same 60-µm edge cap used in the existing region analysis.
        position=np.full(len(cells),-1,int);position[matched]=np.arange(len(matched))
        nbr=members[matched];keep=position[nbr]>=0
        keep&=np.linalg.norm(xy[matched,None]-xy[nbr],axis=-1)<=60
        first=np.broadcast_to(np.arange(len(matched))[:,None],nbr.shape)[keep]
        graph=coo_matrix((np.ones(len(first)),(first,position[nbr[keep]])),
                         shape=(len(matched),len(matched))).tocsr()
        if len(matched):
            _,component=connected_components(graph,directed=False)
            sizes=np.bincount(component);component_size=sizes[component]
        else:
            component=np.array([],int);sizes=np.array([],int);component_size=np.array([],int)
        references=cells.iloc[matched].niche.value_counts()
        summaries.append({'Niche':f'N{niche}','DKD中心数':len(case),
            'HC其他niche中组成相近中心数':len(matched),
            '最大相连区域中心数':int(sizes.max()) if len(sizes) else 0,
            '至少60中心的区域数':int((sizes>=60).sum()),
            'HC相近区域原niche':'; '.join(f'N{int(n)}: {count}' for n,count in references.items())})
        mask=np.zeros(len(cells),bool);mask[matched]=True
        for row,ids in enumerate((dk,hc)):
            ax=axes[col,row]
            ax.scatter(xy[ids,0],xy[ids,1],s=.5,c='#E1E3E5',linewidths=0,rasterized=True)
            if row==0:
                ax.scatter(xy[case,0],xy[case,1],s=1,c=[report.joint_colors[niche-1]],linewidths=0,rasterized=True)
                ax.set_title(f'DKD · N{niche} (n={len(case):,})')
            else:
                for other in sorted(set(labels[matched])):
                    hit=matched[labels[matched]==other]
                    ax.scatter(xy[hit,0],xy[hit,1],s=1,c=[report.joint_colors[other-1]],linewidths=0,rasterized=True)
                ax.set_title(f'HC · similar composition, excluding N{niche}\nn={len(matched):,}; largest region={int(sizes.max()) if len(sizes) else 0:,}',fontsize=9)
            ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
            lower=xy[ids].min(axis=0);span=np.ptp(xy[ids],axis=0)
            bx=lower[0]+.04*span[0];by=lower[1]+.96*span[1]
            ax.plot([bx,bx+100],[by,by],color='#555555',lw=1.4)
            ax.text(bx+50,by+.025*span[1],'100 µm',ha='center',va='top',fontsize=7)
        export=pd.DataFrame({'HC_row':eligible,'original_joint_niche':labels[eligible],
             'mean_5_DKD_Hellinger_distance':distance,'similar_composition':distance<=tolerance})
        export['component']=-1;export['component_size']=0
        export.loc[export.similar_composition,'component']=component
        export.loc[export.similar_composition,'component_size']=component_size
        export.to_csv(report.analysis/f'joint_N{niche}_HC_other_niche_regions.csv',index=False)
    handles=[Line2D([],[],marker='o',ls='',color=color,label=f'N{i+1}',markersize=5)
             for i,color in enumerate(report.joint_colors)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=10,fontsize=8)
    fig.text(.5,.015,'HC highlights retain their original joint niche colors; HC centers already assigned to the queried niche are excluded.\n'
        f'Similarity: mean Hellinger distance to five nearest DKD niche neighborhoods ≤ {tolerance}. Original distance-weighted compositions.\n'
        'All matching centers are shown; region sizes use original graph edges ≤60 µm. Composition similarity does not establish a CCC difference.',
        ha='center',fontsize=7)
    table=pd.DataFrame(summaries)
    table.to_csv(report.analysis/'joint_HC_other_niche_regions_summary.csv',index=False)
    stem='joint_'+'_'.join(f'N{n}' for n in niches)+'_HC_other_niche_regions'
    return report.save(fig,stem),table
