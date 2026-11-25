"""One NicheCompass fit per brain sample; reuse completed STARmap model.

MERFISH uses one 100-epoch fit; STARmap reuses the prior 400-epoch fit.
KMeans and a predeclared K=7 Leiden search are both downstream analyses.
"""
import os,sys,json,time,random,pickle,traceback
os.environ.setdefault('OMP_NUM_THREADS','3');os.environ.setdefault('OPENBLAS_NUM_THREADS','3');os.environ.setdefault('MKL_NUM_THREADS','3')
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,torch
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,confusion_matrix,f1_score
from scipy.optimize import linear_sum_assignment
import scanpy as sc
W=Path('/data1/xueshuailin/CCC_Phe/workspace');D=W/'data/harmonics_fig2';O=W/'outputs/brain_nichecompass_single_run';ds=sys.argv[1];assert ds in ('starmap','merfish');out=O/ds;out.mkdir(parents=True,exist_ok=True);seed=112;torch.set_num_threads(3);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);start=time.monotonic()
try:
 if (out/'complete.json').exists():print('Already completed; refusing duplicate fit',flush=True);sys.exit(0)
 path=D/('starmap/analysis_input.h5ad' if ds=='starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad');raw=ad.read_h5ad(path);xy=np.asarray(raw.obsm['spatial']).copy();ids=raw.obs_names.to_numpy(dtype=str);print('START',ds,'n',len(ids),'GPU',torch.cuda.get_device_name(0),flush=True)
 if ds=='starmap':
  prior=W/'outputs/brain_harmonics_nichecompass_fusion/starmap';h=np.load(prior/'nichecompass_h.npy');old=np.load(prior/'predictions.npz',allow_pickle=True);assert np.array_equal(ids,old['cell_ids'].astype(str));epochs=400;training='reused existing fit';print('Reuse saved 400-epoch STARmap h',h.shape,flush=True)
 else:
  from nichecompass.models import NicheCompass
  from nichecompass.utils import add_gps_from_gp_dict_to_adata
  counts=sparse.csr_matrix(raw.layers['counts'],dtype=np.float32);a=ad.AnnData(X=counts.copy(),obs=pd.DataFrame(index=raw.obs_names.copy()),var=pd.DataFrame(index=raw.var_names.copy()));a.layers['counts']=counts;a.obsm['spatial']=xy
  gp=pickle.load((W/'data/priors/nichecompass_original_mouse/public_mouse_combined.pkl').open('rb'));near=cKDTree(xy).query(xy,k=5,workers=3)[1];g=sparse.csr_matrix((np.ones(len(a)*4,dtype='float32'),(np.repeat(np.arange(len(a)),4),near[:,1:].ravel())),shape=(len(a),len(a)));a.obsp['spatial_connectivities']=g.maximum(g.T).tocsr();add_gps_from_gp_dict_to_adata(gp,a,genes_uppercase=True,min_genes_per_gp=1,min_source_genes_per_gp=0,min_target_genes_per_gp=0,filter_genes_not_in_masks=False);print('Prior and graph ready; training 100 epochs',flush=True)
  torch.cuda.set_device(0);model=NicheCompass(a,counts_key='counts',adj_key='spatial_connectivities',n_addon_gp=10,active_gp_thresh_ratio=.05,gene_expr_recon_dist='nb',n_fc_layers_encoder=1,conv_layer_encoder='gcnconv',n_hidden_encoder=None,log_variational=True,node_label_method='one-hop-norm',use_cuda_if_available=True,seed=seed)
  model.train(n_epochs=100,n_epochs_all_gps=25,n_epochs_no_cat_covariates_contrastive=5,lr=.001,lambda_edge_recon=5000000.,lambda_gene_expr_recon=3000.,lambda_cat_covariates_contrastive=0.,contrastive_logits_pos_ratio=0.,contrastive_logits_neg_ratio=0.,lambda_group_lasso=0.,lambda_l1_masked=0.,lambda_l1_addon=0.,edge_batch_size=256,node_batch_size=None,n_sampled_neighbors=-1,seed=seed,verbose=True,use_cuda_if_available=True)
  h=model.get_latent_representation(adata=a,counts_key='counts',only_active_gps=True,node_batch_size=512,dtype=np.float32);np.save(out/'h.npy',h);epochs=100;training='new short-budget fit';print('TRAIN COMPLETE',h.shape,flush=True)
 assert len(h)==len(ids);km=KMeans(n_clusters=7,n_init=10,random_state=40700).fit_predict(h);z=ad.AnnData(np.zeros((len(h),1),dtype=np.float32));z.obsm['X_h']=h;sc.pp.neighbors(z,use_rep='X_h',random_state=seed);leiden=None;resolution=None
 for res in [.1,.05,.15,.025,.075,.125,.175,.2,.225,.25,.3,.35,.4,.45,.5,.6,.7,.8,1.]:
  sc.tl.leiden(z,resolution=res,random_state=seed,key_added='leiden');lab=z.obs.leiden.astype(str)
  if lab.nunique()==7:leiden=pd.Categorical(lab).codes;resolution=res;break
 np.savez_compressed(out/'predictions.npz',kmeans=km,leiden=np.array([],dtype=int) if leiden is None else leiden,cell_ids=ids)
 # Annotations used only after predictions are frozen.
 if ds=='starmap':
  ref=ad.read_h5ad(D/'starmap/stDyer_mouse_cortex.h5ad');dist,ix=cKDTree(np.c_[ref.obs.X,-ref.obs.Y]).query(xy);valid=dist<.01;truth=pd.Categorical(ref.obs.label.iloc[ix].astype(str).str.upper()).codes
 else:valid=np.ones(len(ids),bool);truth=pd.Categorical(raw.obs.benchmark_region.astype(str)).codes
 def evaluate(lab):
  cm=confusion_matrix(truth[valid],lab[valid],labels=np.arange(7));r,c=linear_sum_assignment(-cm);mp=dict(zip(c,r));match=np.array([mp.get(x,-1) for x in lab[valid]]);return {'ARI':float(adjusted_rand_score(truth[valid],lab[valid])),'macro_F1':float(f1_score(truth[valid],match,average='macro'))}
 result={'dataset':ds,'seed':seed,'epochs_budget':epochs,'training':training,'K':7,'n':len(ids),'n_evaluated':int(valid.sum()),'h_dim':h.shape[1],'KMeans':evaluate(km),'Leiden':None if leiden is None else evaluate(leiden),'leiden_resolution':resolution,'leiden_grid':'predeclared first seven-class match independent of truth','prior':'native mouse NicheCompass GP; spatial KNN4 symmetric','truth':'evaluation only','note':'single run; reduced MERFISH budget; historical V3 comparators used mouse-to-human ortholog inputs','seconds':round(time.monotonic()-start,2)};(out/'complete.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result),flush=True)
except Exception:(out/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
