"""Pre-model selection: shared skin architecture with a localized disease change.

This audit never uses learned GNN niches. All sections and alternative pairs
remain in the exported tables; tissue geometry is never warped for display.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,anndata as ad,torch
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from phenoniche.cohorts.dermatomyositis import audit_sections
from phenoniche.cohorts.colitis import tissue_components
from phenoniche.gnn.data import neighborhoods,pair_composition
from phenoniche.gnn.model import physical_weights

OUT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/dermatomyositis_shared_structure/diagnostics')


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    a=ad.read_h5ad('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/dermatomyositis_5k/all_sections.h5ad')
    quality,_=audit_sections(a);types=sorted(a.obs.cell_type.unique());codes=pd.Categorical(a.obs.cell_type,categories=types).codes
    rng=np.random.default_rng(52173);sections={};records=[]
    for (patient,image),ids in a.obs.groupby(['patient','image'],observed=True).indices.items():
        ids=np.asarray(ids);xy=np.asarray(a.obsm['spatial'][ids]);m,w=neighborhoods(xy)
        comp=np.empty((len(ids),len(types)),np.float32)
        for start in range(0,len(ids),512):
            member=m[start:start+512]
            pw=physical_weights(torch.as_tensor(xy[member],device='cuda:2'),torch.as_tensor(w[start:start+len(member)],device='cuda:2'),20.,torch.as_tensor(member,device='cuda:2'))
            comp[start:start+len(member)]=pair_composition(torch.as_tensor(codes[ids][member],device='cuda:2'),pw,len(types)).cpu().numpy()
        core,keep=tissue_components(xy,np.repeat(image,len(ids)),100,100)
        sizes=pd.Series(core[keep]).value_counts();largest=core==sizes.index[0];v=xy[largest]
        eigen=np.linalg.eigvalsh(np.cov(v.T));aspect=float(np.sqrt(eigen[-1]/max(eigen[0],1e-8)))
        hull=ConvexHull(v);circularity=float(4*np.pi*hull.volume/hull.area**2)
        local=(a.obs.inflammation_score.to_numpy()[ids][m]*w).sum(1)/np.maximum(w.sum(1),1e-8)
        sections[image]={'patient':patient,'ids':ids,'composition':comp,'profile':np.bincount(codes[ids],minlength=len(types))/len(ids),'local':local}
        records.append({'image':image,'patient':patient,'largest_fragment_fraction':float(largest.mean()),
                        'largest_fragment_aspect':aspect,'largest_fragment_circularity':circularity})
    # Composition archetypes are only a pre-model tissue-matching summary.
    sample=np.concatenate([np.sqrt(v['composition'][rng.choice(len(v['ids']),min(500,len(v['ids'])),replace=False)]) for v in sections.values()])
    km=KMeans(n_clusters=8,n_init=10,random_state=52173).fit(sample)
    for image,v in sections.items():
        label=km.predict(np.sqrt(v['composition']));v['archetypes']=np.bincount(label,minlength=8)/len(label)
    table=quality.merge(pd.DataFrame(records),on=['patient','image'],validate='one_to_one')
    clinical=pd.read_csv('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/dermatomyositis_5k/published_sample_metadata.csv').drop_duplicates('patient_id').rename(columns={'patient_id':'patient'})
    table=table.merge(clinical[['patient','site','source','cancer']],on='patient',how='left');table.to_csv(OUT/'all_section_structure.csv',index=False)
    pairs=[]
    for h in table[table.condition=='HC'].itertuples():
        hv=sections[h.image]
        for d in table[table.condition=='DM'].itertuples():
            dv=sections[d.image];overlap=float(np.minimum(hv['profile'],dv['profile']).sum())
            shared=float(np.minimum(hv['archetypes'],dv['archetypes']).sum())
            record={'healthy_patient':h.patient,'healthy_image':h.image,'DM_patient':d.patient,'DM_image':d.image,
                'cell_type_overlap':overlap,'local_composition_overlap':shared,
                'composition_JS':float(jensenshannon(hv['profile'],dv['profile'])**2),
                'RNA_depth_log2_ratio':float(np.log2(d.median_counts/h.median_counts)),
                'DM_largest_fragment_aspect':d.largest_fragment_aspect,
                'DM_largest_fragment_circularity':d.largest_fragment_circularity,
                'DM_largest_fragment_fraction':d.largest_fragment_fraction,
                'local_inflammation_ratio':d.inflammation_local_p90/max(h.inflammation_local_p90,1e-6),
                'same_source':h.source==d.source,'healthy_site':h.site,'DM_site':d.site}
            record['eligible']=(h.quality_eligible and d.quality_eligible and d.cells>=5000 and
                d.largest_fragment_aspect<=3.5 and overlap>=.7 and shared>=.55 and
                abs(record['RNA_depth_log2_ratio'])<=1 and record['local_inflammation_ratio']>=2)
            record['selection_score']=.5*shared+.3*overlap+.2*d.largest_fragment_circularity
            pairs.append(record)
    pairs=pd.DataFrame(pairs).sort_values(['eligible','selection_score'],ascending=False)
    pairs.to_csv(OUT/'all_candidate_pairs.csv',index=False)
    (OUT/'selection_rules.json').write_text(json.dumps({'uses_GNN_results':False,'composition_archetypes_for_tissue_matching_only':8,
        'eligible':'>=5000 DM cells; aspect<=3.5; global type overlap>=0.7; local composition overlap>=0.55; RNA median within 2-fold; observed local inflammation>=2-fold',
        'score':'.5 local composition overlap + .3 global type overlap + .2 circularity'},indent=2))
    selected=table[table.condition.isin(['HC','DM'])].sort_values(['condition','patient','image'])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,5,figsize=(18,11));palette=np.array(plt.get_cmap('tab20').colors)
    for ax,r in zip(axes.flat,selected.itertuples()):
        v=sections[r.image];xy=a.obsm['spatial'][v['ids']]
        ax.scatter(xy[:,0],xy[:,1],c=palette[codes[v['ids']]],s=.5,linewidths=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.set_title(f'{r.patient.replace("patient_","")} · {r.image} · {len(v["ids"]):,}\nRNA {r.median_counts:.0f}; aspect {r.largest_fragment_aspect:.1f}',fontsize=9)
    handles=[Line2D([],[],marker='o',ls='',color=palette[i],label=t,markersize=4) for i,t in enumerate(types)]
    fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=7);fig.subplots_adjust(bottom=.12,hspace=.25,wspace=.12)
    fig.savefig(OUT/'all_skin_sections.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(pairs.head(12).to_string(index=False),flush=True)


if __name__=='__main__':main()
