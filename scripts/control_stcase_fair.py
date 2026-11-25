"""Wait for CCC preparation, then train and publish type-specific comparison."""
import os,sys,time,subprocess,fcntl
from pathlib import Path
root=Path('/data1/xueshuailin/CCC_Phe/workspace');out=root/'outputs/tumor_immune_interface/v2_baseline_k5/stcase_fair';project=Path('/home/xueshuailin/CCC_Phe');pid=int(sys.argv[1])
while not (out/'ccc_complete.json').exists():
 if not Path(f'/proc/{pid}').exists():raise SystemExit('CCC preparation ended without completion; inspect its logs')
 time.sleep(5)
print('CCC prepared; starting STCase type-specific models',flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
with (out/'training.log').open('a',buffering=1) as f:
 p=subprocess.run([sys.executable,'-u',str(project/'scripts/train_stcase_fair_comparison.py')],cwd=project,env=env,stdout=f,stderr=subprocess.STDOUT)
if p.returncode:raise SystemExit(f'STCase training failed with exit {p.returncode}; see {out / "training.log"}')
import nbformat,pandas as pd
metrics=pd.read_csv(out/'per_cell_type_metrics.csv');nbpath=project/'Multifocal_Communication_Simulation.ipynb'
with (out.parent/'display.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);nb=nbformat.read(nbpath,4)
 for c in nb.cells:
  if 'STCase 公平对照：同一细胞类型' in c.source:
   c.outputs=[nbformat.v4.new_output('display_data',data={'text/plain':metrics.to_string(index=False),'text/html':metrics.to_html(index=False)},metadata={})]
 nbformat.validate(nb);nbformat.write(nb,nbpath)
print('Published per-cell-type comparison in notebook.',flush=True)
