"""Diagnose existing slice separation without any new model training."""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.simulation.baselines import dictionary_program_projection


def main():
    source=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/dermatomyositis_5k_pair')
    out=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/dermatomyositis_shared_structure/diagnostics')
    cells=pd.read_csv(source/'assignments.csv.gz');p=np.load(source/'gnn/P.npy');a=np.log1p(np.load(source/'gnn/activity.npy'))
    comp=np.load(source/'input/composition.npy');rng=np.random.default_rng(52173)
    fit=rng.choice(len(cells),min(12000,len(cells)),replace=False);records=[]
    def record(name,labels):
        f=pd.crosstab(cells.condition,labels,normalize='index');overlap=float(np.minimum(f.loc['HC'],f.loc['DM']).sum())
        records.append({'representation':name,'fixed_K':10,'shared_niche_mass':overlap,
            'niches_at_least_1pct_in_both':int(((f.loc['HC']>=.01)&(f.loc['DM']>=.01)).sum())})
    record('Saved original GNN + spatial pooling',cells.niche)
    for name,x in [('Composition only',np.sqrt(comp)),('GNN program mixture only',np.sqrt(p)),
                   ('GNN program mixture + activity',np.column_stack([np.sqrt(p),.25*(a-np.median(a))/max(a.std(),.1)]))]:
        model=KMeans(n_clusters=10,n_init=10,random_state=52173).fit(x[fit]);record(name,model.predict(x))
    h=np.load(source/'gnn/H.npy');scale=np.load(source/'input/preprocessing.npz')['feature_scale']
    profile=p@dictionary_program_projection(h,scale)
    profile/=max(float(np.sqrt(np.mean(np.sum((profile[fit]-profile[fit].mean(0))**2,axis=1)))),1e-12)
    for name,x in [('CCC dictionary profile only',profile),('CCC dictionary profile + activity',np.column_stack([profile,.25*(a-np.median(a))/max(a.std(),.1)]))]:
        model=KMeans(n_clusters=10,n_init=10,random_state=52173).fit(x[fit]);record(name,model.predict(x))
    pd.DataFrame(records).to_csv(out/'initial_pair_separation_probes.csv',index=False);print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main()
