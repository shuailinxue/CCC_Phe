"""Refresh saved scatter outputs from existing labels; no analysis retraining."""
from pathlib import Path
import base64,json,shutil
import anndata as ad, numpy as np,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from nbclient import NotebookClient
R=Path('/home/xueshuailin/CCC_Phe');W=Path('/data1/xueshuailin/CCC_Phe/workspace');B=W/'outputs/tumor_immune_interface';A=B/'v2_baseline_k5';D=W/'data/tumor_immune_interface_v2_epithelial_majority';a=ad.read_h5ad(D/'tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close();order=np.random.default_rng(2026).permutation(len(truth));colors=np.array(['#d4d8dd','#e78087','#da9b3b','#8960ac','#3b9298']);predcolors=np.array(['#2780b8','#ff9b41','#49a662','#9b70b9','#cf7583']);names=['Background','Tumor core','Interface A','Interface B','Infiltrated tumor']
def draw(ax,v,pal,title):
 ax.scatter(xy[order,0],xy[order,1],s=4.3,c=pal[v[order]],lw=0,rasterized=True);ax.axis('off');ax.set_aspect('equal');ax.set_title(title,fontsize=10)
shutil.copyfile(A/'comparison_display_preview.png',A/'comparison.png')
# Original communication baseline remains unchanged; its invalid STCase adaptation stays clearly labelled.
fig,axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained');draw(axes[0],truth,colors,'Ground truth · K=5');axes[1].axis('off');axes[1].set_title('STCase · rerun required');axes[1].text(.5,.5,'Invalid adaptation; rerun required',ha='center',transform=axes[1].transAxes)
p=np.load(A/'commuspace/labels.npy');draw(axes[2],p,predcolors,f'CommuSpace · ARI={adjusted_rand_score(truth,p):.3f}');fig.savefig(A/'communication_baselines.png',dpi=160,bbox_inches='tight');plt.close(fig)
cases=[('baseline','All three · PCA16'),('program','Program only'),('response','Response only'),('ccc_pca','CCC only · PCA16'),('no_program','Without program'),('no_response','Without response'),('no_ccc','Without CCC'),('raw_combined','All three · raw CCC'),('raw_only','CCC only · raw'),('raw_all_only','CCC only · all raw')]
for out in [B/'v2_three_view_ablation',B/'v2_cellphonedb_v5/ablation']:
 fig,axes=plt.subplots(3,4,figsize=(19,13),layout='constrained');axes=axes.ravel();draw(axes[0],truth,colors,'Ground truth')
 for ax,(key,title) in zip(axes[1:],cases):
  p=out/(key+'.npy'); meta=out/(key+'.json')
  if not p.exists() or not meta.exists():ax.axis('off');continue
  v=np.load(p);u=np.unique(v);counts=np.array([[np.sum((truth==k)&(v==x)) for x in u] for k in range(5)]);rows,cols=linear_sum_assignment(-counts);lookup={u[j]:int(i) for i,j in zip(rows,cols)};v=np.array([lookup[x] for x in v]);m=json.loads(meta.read_text());draw(ax,v,colors,f'{title}\nGlobal ARI {m["global_ARI"]:.3f} · A/B ARI {m["AB_ARI"]:.3f}')
 for ax in axes[len(cases)+1:]:ax.axis('off')
 fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=c,label=n,markersize=6) for n,c in zip(names,colors)],loc='outside lower center',ncol=5,frameon=False);fig.suptitle('Three-view ablation · identical coarse class and K=2\nColor matching for display only',fontsize=14);fig.savefig(out/'comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
for filename in ['Multifocal_Communication_Simulation.ipynb','Multifocal_CellPhoneDB_V5_Simulation.ipynb']:
 path=R/filename;nb=nbformat.read(path,4);c0=nb.cells[0];start=c0.source.index('def spatial_panel(');end=c0.source.index('\nfig, axes',start)
 c0.source=c0.source[:start]+'''def spatial_panel(ax, labels, names, colors, title, sizes=4.3):
    order = np.random.default_rng(2026).permutation(len(labels))
    palette = np.asarray(colors)
    if labels.dtype.kind in 'OUS':
        lookup = {name: k for k, name in enumerate(names)}
        ids = np.asarray([lookup[x] for x in labels])
    else:
        ids = labels.astype(int)
    ax.scatter(xy[order, 0], xy[order, 1], s=4.3, c=palette[ids[order]], linewidths=0, rasterized=True)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=13)
    handles = [Line2D([0], [0], marker='o', linestyle='', color=c, markersize=6, label=n)
               for n, c in zip(names, colors)]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.025),
              ncol=3, frameon=False, fontsize=8)
''' + c0.source[end:]
 # Remaining panel previously highlighted a final category; use same fixed order across panels.
 c=nb.cells[2];s=c.source;start=s.index('for k, (name, color) in enumerate(zip(fine_names, fine_colors)):');end=s.index("axes[2].set_aspect",start)
 s=s[:start]+'''order = np.random.default_rng(2026).permutation(len(fine))
axes[2].scatter(xy[order, 0], xy[order, 1], s=4.3, c=np.asarray(fine_colors)[fine[order]], linewidths=0, rasterized=True)
'''+s[end:];c.source=s
 inds=[0,2]+([3] if 'Communication_Simulation' in filename and 'CellPhoneDB' not in filename else [])
 sub=nbformat.v4.new_notebook(cells=[nb.cells[i] for i in inds],metadata=nb.metadata);NotebookClient(sub,timeout=600,kernel_name='cccphe',resources={'metadata':{'path':str(R)}}).execute()
 for ix,cell in zip(inds,sub.cells):nb.cells[ix]=cell
 # Refresh embedded images from their unchanged source paths.
 for cell in nb.cells:
  src=cell.source
  if 'display(Image(filename=' not in src:continue
  import re
  m=re.search(r"filename=['\"]([^'\"]+)['\"]",src)
  if not m:continue
  img=Path(m.group(1))
  if img.exists() and img.name in {'comparison.png','communication_baselines.png'}:
   cell.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(img.read_bytes()).decode()},metadata={})]
 nbformat.write(nb,path);print('updated',path,flush=True)
