"""Reproduce author single-sample benchmarking defaults, simulation run 1.
No planted niche labels or gp_data are used in fitting. Public current database
versions may differ from the paper's archived 2023 files; see metadata."""
from pathlib import Path
import os,sys,json,traceback,time,pickle
import numpy as np,pandas as pd,anndata as ad,torch
R=Path('/home/xueshuailin/CCC_Phe');W=Path('/data1/xueshuailin/CCC_Phe/workspace');O=W/'outputs/nichecompass_simulation/nichecompass_paper_run1';O.mkdir(parents=True,exist_ok=True);P=W/'data/priors/nichecompass_original_mouse';P.mkdir(parents=True,exist_ok=True);os.chdir(P);torch.set_num_threads(4)
from nichecompass.models import NicheCompass
from nichecompass.utils import extract_gp_dict_from_omnipath_lr_interactions,extract_gp_dict_from_nichenet_lrt_interactions,extract_gp_dict_from_mebocost_ms_interactions,filter_and_combine_gp_dict_gps,add_gps_from_gp_dict_to_adata
from scipy.spatial import cKDTree
from scipy import sparse
try:
 prior=P/'public_mouse_combined.pkl'
 if prior.exists():gp=pickle.load(prior.open('rb'))
 else:
  orth=W/'tmp/nichecompass-reproducibility/datasets/ga_data/human_mouse_gene_orthologs.csv';met=W/'tmp/nichecompass-reproducibility/datasets/gp_data/metabolite_enzyme_sensor_gps';print('Preparing official OmniPath mouse programs',flush=True)
  omni=extract_gp_dict_from_omnipath_lr_interactions(species='mouse',min_curation_effort=0,load_from_disk=False,save_to_disk=True,lr_network_file_path=str(P/'omnipath_lr_network.csv'),gene_orthologs_mapping_file_path=str(orth),plot_gp_gene_count_distributions=False)
  print('Preparing official mouse NicheNet v2 programs',flush=True)
  niche=extract_gp_dict_from_nichenet_lrt_interactions(species='mouse',version='v2',max_n_target_genes_per_gp=250,load_from_disk=False,save_to_disk=True,lr_network_file_path=str(P/'nichenet_lr_mouse.csv'),ligand_target_matrix_file_path=str(P/'nichenet_ligand_target_mouse.csv'),plot_gp_gene_count_distributions=False)
  print('Preparing official MEBOCOST mouse programs',flush=True);metab=extract_gp_dict_from_mebocost_ms_interactions(dir_path=str(met),species='mouse',plot_gp_gene_count_distributions=False);gp=dict(omni);gp.update(niche);gp.update(metab);gp=filter_and_combine_gp_dict_gps(gp_dict=gp,gp_filter_mode='subset',combine_overlap_gps=True,overlap_thresh_source_genes=.9,overlap_thresh_target_genes=.9,overlap_thresh_genes=.9,verbose=False);pickle.dump(gp,prior.open('wb'));print('Prepared public GP count',len(gp),flush=True)
 a=ad.read_h5ad(W/'data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad');truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {i}' for i in range(1,9)]).codes;del a.uns['gp_data'];a.obs.drop(columns=['niche_types'],inplace=True);xy=np.asarray(a.obsm['spatial']);_,ind=cKDTree(xy).query(xy,k=5,workers=4);n=len(a);g=sparse.csr_matrix((np.ones(n*4,dtype='float32'),(np.repeat(np.arange(n),4),ind[:,1:].ravel())),shape=(n,n));a.obsp['spatial_connectivities']=g.maximum(g.T).tocsr();add_gps_from_gp_dict_to_adata(gp,a,genes_uppercase=True,min_genes_per_gp=1,min_source_genes_per_gp=0,min_target_genes_per_gp=0,filter_genes_not_in_masks=False);print('Public GP mask',len(a.uns['nichecompass_gp_names']),flush=True)
 torch.cuda.set_device(1);model=NicheCompass(a,counts_key='counts',adj_key='spatial_connectivities',n_addon_gp=10,active_gp_thresh_ratio=.05,gene_expr_recon_dist='nb',n_fc_layers_encoder=1,conv_layer_encoder='gcnconv',n_hidden_encoder=None,log_variational=True,node_label_method='one-hop-norm',use_cuda_if_available=True,seed=0);print('Paper run1 GCN k4 seed0 400 epochs, native batch/validation',flush=True);start=time.monotonic();model.train(n_epochs=400,n_epochs_all_gps=25,n_epochs_no_cat_covariates_contrastive=5,lr=.001,lambda_edge_recon=5000000.,lambda_gene_expr_recon=3000.,lambda_cat_covariates_contrastive=0.,contrastive_logits_pos_ratio=0.,contrastive_logits_neg_ratio=0.,lambda_group_lasso=0.,lambda_l1_masked=0.,lambda_l1_addon=0.,edge_batch_size=256,node_batch_size=None,n_sampled_neighbors=-1,seed=0,verbose=True,use_cuda_if_available=True);model.save(str(O/'model'),overwrite=True,save_adata=False);h=model.get_latent_representation(adata=a,counts_key='counts',only_active_gps=True,node_batch_size=512,dtype=np.float32);np.save(O/'embedding.npy',h);print('Training complete, Leiden clustering',flush=True)
 import scanpy as sc
 from sklearn.metrics import adjusted_rand_score,confusion_matrix
 z=ad.AnnData(np.zeros((n,1),dtype='float32'));z.obsm['X_h']=h;sc.pp.neighbors(z,use_rep='X_h',random_state=0)
 # Predetermined resolution sweep; first solution with eight clusters. No truth-based ARI selection.
 candidates=[.1,.05,.15,.025,.075,.125,.175,.2,.225,.25,.3,.35,.4,.45,.5,.6,.7,.8,1.];found=None
 for resolution in candidates:
  sc.tl.leiden(z,resolution=resolution,random_state=0,key_added='leiden');labs=z.obs.leiden.astype(str);k=labs.nunique();print('resolution',resolution,'clusters',k,flush=True)
  if k==8:found=(resolution,pd.Categorical(labs).codes);break
 if found is None:raise RuntimeError('No 8-cluster Leiden solution on predefined resolutions; saved embedding')
 resolution,labels=found;np.save(O/'labels_K8.npy',labels);ari=float(adjusted_rand_score(truth,labels));(O/'result.json').write_text(json.dumps({'ARI':ari,'k':8,'leiden_resolution':resolution,'spatial_k':4,'epochs':400,'seed':0,'encoder':'gcnconv','prior':'mouse OmniPath + NicheNet v2 + MEBOCOST official extraction, current versions not archived','truth_used_only_evaluation':True,'runtime_seconds':time.monotonic()-start},indent=2));print('FINAL ARI',ari,flush=True)
 import nbformat
 from nbclient import NotebookClient
 p=R/'NicheCompass_Simulation_V2.ipynb';nb=nbformat.read(p,4);source=f'''# NicheCompass 作者设置复现：鼠源先验、KNN4、Leiden K=8
%matplotlib inline
from pathlib import Path
import numpy as np,pandas as pd,anndata as ad,matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix,adjusted_rand_score
from scipy.optimize import linear_sum_assignment
out=Path('{O}');a=ad.read_h5ad('{W}/data/nichecompass_simulation/sim1_1105genes_10000locs_strongincrements.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=pd.Categorical(a.obs.niche_types,categories=[f'Niche {{i}}' for i in range(1,9)]).codes;a.file.close();pred=np.load(out/'labels_K8.npy');r,c=linear_sum_assignment(-confusion_matrix(truth,pred,labels=range(8)));mapping=dict(zip(c,r));pred=np.array([mapping[i] for i in pred]);colors=plt.get_cmap('tab10')(np.arange(8));order=np.random.default_rng(2026).permutation(len(truth));fig,axes=plt.subplots(1,2,figsize=(8,7),layout='constrained')
for ax,labels,title in [(axes[0],truth,'Ground truth'),(axes[1],pred,f'NicheCompass · Leiden K=8\\nARI = {{adjusted_rand_score(truth,pred):.3f}}')]:
 ax.scatter(xy[order,0],xy[order,1],c=colors[labels[order]],s=4,lw=0,rasterized=True);ax.set_aspect('equal');ax.axis('off');ax.set_title(title,fontsize=11)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=colors[i],label=f'Niche {{i+1}}',markersize=5) for i in range(8)],loc='outside lower center',ncol=4,frameon=False,fontsize=8);plt.show()
''';one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)],metadata=nb.metadata);NotebookClient(one,timeout=300,kernel_name='cccphe').execute(cwd=str(R));assert any('image/png' in o.get('data',{}) for o in one.cells[0].outputs);latest=nbformat.read(p,4);latest.cells.append(one.cells[0]);nbformat.write(latest,p);print('Notebook updated',flush=True)
except BaseException:(O/'failure.txt').write_text(traceback.format_exc());traceback.print_exc();raise
