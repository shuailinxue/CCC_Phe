"""Run seven existing native baselines on current simulation; update one notebook cell."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess,sys,os,json,fcntl,base64,time
import anndata as ad,numpy as np,nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1];NB=ROOT/'Multifocal_Communication_Simulation.ipynb';S=Path('/data1/xueshuailin/CCC_Phe/workspace/data/tumor_immune_interface_v2_epithelial_majority');O=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/tumor_immune_interface/v2_baseline_k5');O.mkdir(parents=True,exist_ok=True)
METHODS={'GraphST':'graphst','STAGATE':'stagate','CellCharter (scVI)':'cellcharter','scNiche':'scniche','NicheCompass':'nichecompass','SpaNiche':'spaniche','BANKSY':'banksy'}
GPU=['GraphST','STAGATE','CellCharter (scVI)','scNiche','NicheCompass'];CPU=['BANKSY','SpaNiche']
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PHENONICHE_BASELINE_GPU='1',PHENONICHE_BASELINE_K='5',PHENONICHE_BASELINE_DATA=str(S),PHENONICHE_BASELINE_OUTPUT=str(O),PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLBACKEND='Agg')

def render():
 with (O/'display.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);a=ad.read_h5ad(S/'tissue.h5ad',backed='r');xy=np.asarray(a.obsm['spatial']);truth=np.array([0,1,2,3,4,4])[a.obs.design_region_id.to_numpy()];a.file.close();fg=truth!=0;fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained');axes=axes.ravel();colors=['#d4d8dd','#e78087','#da9b3b','#8960ac','#3b9298'];names=['Background','Tumor core','Interface A','Interface B','Infiltrated tumor'];summary=[]
  for ax in axes:ax.set_aspect('equal');ax.axis('off')
  for k,col in enumerate(colors):
   ids=truth==k;axes[0].scatter(*xy[ids].T,s=1.4,c=col,lw=0,rasterized=True)
  axes[0].set_title('Ground truth · 5 niches')
  pal=['#2780b8','#ff9b41','#49a662','#9b70b9','#cf7583']
  for ax,(name,key) in zip(axes[1:],METHODS.items()):
   d=O/key;done=d/'complete.json';lab=d/'labels.npy'
   if done.exists() and lab.exists():
    v=np.load(lab);assert len(v)==len(truth);g=float(adjusted_rand_score(truth,v));t=float(adjusted_rand_score(truth[fg],v[fg]));summary.append(dict(method=name,status='complete',global_ARI=g,tumor_ARI=t));
    for k,col in enumerate(pal):
     ids=v==k;ax.scatter(*xy[ids].T,s=1.4,c=col,lw=0,rasterized=True)
    ax.set_title(f'{name} · ARI={g:.3f} / tumor={t:.3f}')
   else:
    status='failed' if (d/'failed.json').exists() else 'running / queued';summary.append(dict(method=name,status=status));ax.text(.5,.5,status,ha='center',va='center',transform=ax.transAxes,color='#777');ax.set_title(name)
  fig.suptitle('Baselines · same simulated tissue · K=5',fontsize=15);png=O/'comparison.png';fig.savefig(png,dpi=150,bbox_inches='tight');plt.close(fig);(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  nb=nbformat.read(NB,as_version=4);marker='七种 baseline 对照（K=5）';source="""# 七种 baseline 对照（K=5）：使用已保存结果，不在 notebook 中重复训练。
from IPython.display import display, Image
display(Image(filename='/data1/xueshuailin/CCC_Phe/workspace/outputs/tumor_immune_interface/v2_baseline_k5/comparison.png', width=1600))
""";output=nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(png.read_bytes()).decode()},metadata={});cell=nbformat.v4.new_code_cell(source,outputs=[output],execution_count=None);ix=[i for i,c in enumerate(nb.cells) if marker in c.source]
  if ix:nb.cells[ix[0]]=cell
  else:nb.cells.append(cell)
  nbformat.validate(nb);nbformat.write(nb,NB)

def run_queue(names):
 for name in names:
  d=O/METHODS[name];d.mkdir(exist_ok=True);(d/'running.json').write_text(json.dumps({'started':time.time()}));render()
  with (d/'run.log').open('a',buffering=1) as log:
   r=subprocess.run([sys.executable,str(ROOT/'scripts/run_interface_baselines.py'),'--action','worker','--method',name],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
  (d/'running.json').unlink(missing_ok=True)
  if r.returncode:(d/'failed.json').write_text(json.dumps({'exit_code':r.returncode,'log':str(d/'run.log')}))
  else:(d/'failed.json').unlink(missing_ok=True)
  render()

def main():
 with (O/'controller.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);render();(O/'controller.json').write_text(json.dumps({'pid':os.getpid(),'gpu':1,'methods':list(METHODS),'K':5}))
  with ThreadPoolExecutor(max_workers=2) as pool:
   jobs=[pool.submit(run_queue,GPU),pool.submit(run_queue,CPU)]
   for task in jobs:task.result()
  (O/'finished.json').write_text(json.dumps({'summary':json.loads((O/'summary.json').read_text())},indent=2));print('Baseline queues finished',flush=True)
if __name__=='__main__':main()
