"""Recolor saved benchmark visualizations using post-fit truth matching."""
from pathlib import Path
import sys,json,base64
import numpy as np,anndata as ad,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from nbclient import NotebookClient
R=Path('/home/xueshuailin/CCC_Phe');sys.path.insert(0,str(R/'scripts'))
from simulation_plot_colors import TRUTH_COLORS,matched_palette
W=Path('/data1/xueshuailin/CCC_Phe/workspace');S=W/'data/tumor_immune_interface_v2_epithelial_majority';B=W/'outputs/tumor_immune_interface';O=B/'v2_baseline_k5';a=ad.read_h5ad(S/'tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close();order=np.random.default_rng(2026).permutation(len(truth));fg=truth!=0
methods=[('GraphST','graphst'),('STAGATE','stagate'),('CellCharter','cellcharter'),('scNiche','scniche'),('NicheCompass','nichecompass'),('SpaNiche','spaniche'),('BANKSY','banksy')]
def draw(ax,v,pal,title):
 ax.scatter(xy[order,0],xy[order,1],s=4.3,c=np.asarray(pal)[v[order]],lw=0,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.set_title(title,fontsize=11)
fig,axes=plt.subplots(2,4,figsize=(19,10),layout='constrained');fig.suptitle('Baseline comparison · K = 5\nColors matched to truth for display only',fontsize=15);draw(axes.flat[0],truth,TRUTH_COLORS,'Ground truth')
for ax,(name,key) in zip(axes.flat[1:],methods):
 v=np.load(O/key/'labels.npy');draw(ax,v,matched_palette(truth,v,TRUTH_COLORS),f'{name} · ARI {adjusted_rand_score(truth,v):.3f} · tumor {adjusted_rand_score(truth[fg],v[fg]):.3f}')
fig.savefig(O/'comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained');draw(axes[0],truth,TRUTH_COLORS,'Ground truth · K=5');axes[1].axis('off');axes[1].set_title('STCase · rerun required');axes[1].text(.5,.5,'Invalid adaptation; rerun required',ha='center',transform=axes[1].transAxes);v=np.load(O/'commuspace/labels.npy');draw(axes[2],v,matched_palette(truth,v,TRUTH_COLORS),f'CommuSpace · ARI={adjusted_rand_score(truth,v):.3f}');fig.suptitle('Colors matched to truth for display only',fontsize=12);fig.savefig(O/'communication_baselines.png',dpi=160,bbox_inches='tight');plt.close(fig)
# Ablation output already uses one-to-one matching; update its palette to match the same truth.
from matplotlib.lines import Line2D
cases=[('baseline','All three · PCA16'),('program','Program only'),('response','Response only'),('ccc_pca','CCC only · PCA16'),('no_program','Without program'),('no_response','Without response'),('no_ccc','Without CCC'),('raw_combined','All three · raw CCC'),('raw_only','CCC only · raw'),('raw_all_only','CCC only · all raw')]
for out in [B/'v2_three_view_ablation',B/'v2_cellphonedb_v5/ablation']:
 fig,axes=plt.subplots(3,4,figsize=(19,13),layout='constrained');axes=axes.ravel();draw(axes[0],truth,TRUTH_COLORS,'Ground truth')
 for ax,(key,title) in zip(axes[1:],cases):
  p=out/(key+'.npy');meta=out/(key+'.json')
  if not p.exists() or not meta.exists():ax.axis('off');continue
  v=np.load(p);m=json.loads(meta.read_text());draw(ax,v,matched_palette(truth,v,TRUTH_COLORS),f'{title}\nGlobal ARI {m["global_ARI"]:.3f} · A/B ARI {m["AB_ARI"]:.3f}')
 for ax in axes[len(cases)+1:]:ax.axis('off')
 fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=c,label=n,markersize=6) for n,c in zip(['Background','Tumor core','Interface A','Interface B','Infiltrated tumor'],TRUTH_COLORS)],loc='outside lower center',ncol=5,frameon=False);fig.suptitle('Three-view ablation · identical coarse class and K=2\nColors matched to truth for display only',fontsize=14);fig.savefig(out/'comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
for filename in ['Multifocal_Communication_Simulation.ipynb','Multifocal_CellPhoneDB_V5_Simulation.ipynb']:
 path=R/filename;nb=nbformat.read(path,4);cell=nb.cells[2];s=cell.source
 s=s.replace('from sklearn.metrics import adjusted_rand_score','from sklearn.metrics import adjusted_rand_score\nimport sys\nsys.path.insert(0, \'/home/xueshuailin/CCC_Phe/scripts\')\nfrom simulation_plot_colors import matched_palette, TRUTH_COLORS, COARSE_COLORS')
 s=s.replace("coarse_colors = ['#6daf66', '#e89349', '#a87bc1', '#317db4']","# A/B are one reference category only for coloring the K=4 panel.\ncoarse_truth = np.array([0, 1, 2, 2, 3])[truth]\ncoarse_colors = matched_palette(coarse_truth, coarse, COARSE_COLORS)")
 s=s.replace("fine_colors = coarse_colors + ['#228c99', '#e07b55']","fine_colors = matched_palette(truth, fine, TRUTH_COLORS)")
 s=s.replace("'Composition · K=4'","'Composition · K=4 (A/B merged)'");s=s.replace("Tumor-region ARI={foreground_ari:.3f}'","Tumor-region ARI={foreground_ari:.3f} · colors matched'")
 cell.source=s
 inds=[0,2]
 if filename=='Multifocal_Communication_Simulation.ipynb':
  c=nb.cells[3];s=c.source;s=s.replace("colors = ['#317db4', '#e89349', '#6daf66', '#a87bc1', '#cd7a70']","# Each prediction is matched to truth for colors only.\ncolors_composition = matched_palette(truth, ablation['composition'], TRUTH_COLORS)\ncolors_communication = matched_palette(truth, ablation['communication'], TRUTH_COLORS)")
  s=s.replace("[f'C{k+1}' for k in range(5)], colors,","[f'C{k+1}' for k in range(5)], colors_composition,").replace("[f'N{k+1}' for k in range(5)], colors,","[f'N{k+1}' for k in range(5)], colors_communication,")
  c.source=s;inds.append(3)
 sub=nbformat.v4.new_notebook(cells=[nb.cells[i] for i in inds],metadata=nb.metadata);NotebookClient(sub,timeout=600,kernel_name='cccphe',resources={'metadata':{'path':str(R)}}).execute()
 for i,c in zip(inds,sub.cells):nb.cells[i]=c
 import re
 for c in nb.cells:
  m=re.search(r"filename=['\"]([^'\"]+)['\"]",c.source)
  if not m:continue
  p=Path(m.group(1))
  if p.exists() and p.name in {'comparison.png','communication_baselines.png'}:c.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(p.read_bytes()).decode()},metadata={})]
 nbformat.write(nb,path);print('updated',path,flush=True)
