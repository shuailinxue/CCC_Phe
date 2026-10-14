"""Downstream audit of expression/cell-state information; labels never enter the GNN."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from diagnose_gnn_dictionary import load_data
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.data import expression_block
from phenoniche.gnn.model import model_from_config
from phenoniche.ccc.lr_atlas import canonicalize_complex


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    payload=torch.load(source/'starting_checkpoint.pt',map_location='cpu',weights_only=False);cfg=replace(GNNConfig(**payload['config']),device='cuda:2')
    torch.set_num_threads(8);data=load_data(cfg);rng=np.random.default_rng(70330)
    train_cells=np.unique(data.members[data.rows[0]]);valid_cells=np.unique(data.members[data.rows[1]])
    if np.intersect1d(train_cells,valid_cells).size:raise ValueError('Shared physical cells between splits')
    tr=rng.choice(train_cells,min(20000,len(train_cells)),replace=False);va=rng.choice(valid_cells,min(10000,len(valid_cells)),replace=False)
    ytr=data.types[tr];yva=data.types[va];np.savez(out/'cell_rows.npz',train=tr,validation=va)
    selected=np.r_[tr,va];x=expression_block(data.expression,selected,data.encoder_indices,data.totals)
    x=(x-data.gene_mean)/data.gene_scale
    model=model_from_config(data,cfg).to(cfg.device);model.load_state_dict(payload['model_state']);model.eval();encoded=[]
    inputs=np.column_stack((x,np.log1p(data.totals[selected])/10)).astype(np.float32)
    with torch.no_grad():
        for start in range(0,len(inputs),2048):encoded.append(model.cell_encoder(torch.as_tensor(inputs[start:start+2048],device=cfg.device)).cpu().numpy())
    h=np.concatenate(encoded);del model
    lr_genes={g for side in ('ligand','receptor') for value in data.features[side] for g in canonicalize_complex(value)}
    lr_columns=np.array([i for i,index in enumerate(data.encoder_indices) if data.genes[index] in lr_genes])
    records=[]
    def evaluate(name,features):
        scaler=StandardScaler().fit(features[:len(tr)]);features=scaler.transform(features)
        clf=LogisticRegression(max_iter=400,C=1.,class_weight='balanced').fit(features[:len(tr)],ytr)
        pred=clf.predict(features[len(tr):]);result={'representation':name,'accuracy':accuracy_score(yva,pred),'balanced_accuracy':balanced_accuracy_score(yva,pred),'macro_F1':f1_score(yva,pred,average='macro')}
        records.append(result);print(result,flush=True)
        pd.DataFrame(confusion_matrix(yva,pred,labels=np.arange(len(data.categories))),index=data.categories,columns=data.categories).to_csv(out/(name+'_confusion.csv'))
        pd.DataFrame(records).to_csv(out/'cell_state_metrics.csv',index=False)
    evaluate('GNN_cell_encoder',h);evaluate('LR_genes',x[:,lr_columns]);evaluate('all_selected_genes',x)
    # Unsupervised PCA uses only training expression; all-panel comparison is
    # downstream information audit, not a new GNN input or cell-type loss.
    full=expression_block(data.expression,selected,np.arange(len(data.genes)),data.totals)
    scaler=StandardScaler().fit(full[:len(tr)]);full=scaler.transform(full)
    pca=PCA(n_components=64,svd_solver='randomized',random_state=cfg.seed).fit(full[:len(tr)])
    evaluate('panel_expression_PCA64',pca.transform(full))
    (out/'summary.json').write_text(json.dumps({'training_cells':len(tr),'validation_cells':len(va),'physical_cell_overlap':0,
        'test_set_used':False,'purpose':'Downstream label predictability audit only; these classifiers and labels are NOT GNN inputs or losses.',
        'provisional_labels':True,'metrics':records},indent=2)+'\n')


if __name__=='__main__':main()
