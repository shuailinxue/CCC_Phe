"""Keep the two communication baselines in one separate notebook cell."""
import base64,fcntl,json,time
from pathlib import Path
import anndata as ad,numpy as np,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5';nbpath=Path('/home/xueshuailin/CCC_Phe/Multifocal_Communication_Simulation.ipynb')
def render():
 with (out/'display.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  a=ad.read_h5ad(root/'data/tumor_immune_interface_v2_epithelial_majority/tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close()
  fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');pal=['#d4d8dd','#e78087','#da9b3b','#8960ac','#3b9298'];predpal=['#2780b8','#ff9b41','#49a662','#9b70b9','#cf7583']
  for ax in axes:ax.set_aspect('equal');ax.axis('off')
  for k,c in enumerate(pal):m=truth==k;axes[0].scatter(*xy[m].T,s=1.3,c=c,lw=0,rasterized=True)
  axes[0].set_title('Ground truth · K=5')
  for ax,(name,key) in zip(axes[1:],[('STCase · rerun required','stcase'),('CommuSpace','commuspace')]):
   p=out/key/'labels.npy';info=out/key/'complete.json'
   if key != 'stcase' and p.exists() and info.exists() and not (out/key/'consensus_pending').exists():
    labels=np.load(p);ari=adjusted_rand_score(truth,labels)
    for k,c in enumerate(predpal):m=labels==k;ax.scatter(*xy[m].T,s=1.3,c=c,lw=0,rasterized=True)
    ax.set_title(f'{name} · ARI={ari:.3f}')
   else:ax.text(.5,.5,'Invalid adaptation; rerun required' if key == 'stcase' else 'Running',transform=ax.transAxes,ha='center');ax.set_title(name)
  img=out/'communication_baselines.png';fig.savefig(img,dpi=140,bbox_inches='tight');plt.close(fig)
  nb=nbformat.read(nbpath,4);marker='通讯方法 baseline：STCase 与 CommuSpace';src=f"# {marker}（K=5；STCase 先前改造无效、待重跑；CommuSpace 原通讯张量 + 共识聚类）\nfrom IPython.display import display, Image\ndisplay(Image(filename='{img}', width=1400))\n";output=nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(img.read_bytes()).decode()},metadata={});cell=nbformat.v4.new_code_cell(src,outputs=[output]);idx=[i for i,c in enumerate(nb.cells) if marker in c.source]
  if idx:nb.cells[idx[0]]=cell
  else:nb.cells.append(cell)
  nbformat.validate(nb);nbformat.write(nb,nbpath)
if __name__=='__main__':
 previous=None
 for _ in range(2160):
  state=tuple((out/x/'complete.json').exists() for x in ('stcase','commuspace'))
  if state!=previous:render();print('render',state,flush=True);previous=state
  if all(state):break
  time.sleep(20)
