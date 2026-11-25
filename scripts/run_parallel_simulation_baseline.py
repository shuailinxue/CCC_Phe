"""Run one pending baseline on a specified physical GPU, updating the display."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_current_simulation_baselines import ROOT,S,O,METHODS,render
name=sys.argv[1];gpu=sys.argv[2]
if name not in METHODS or gpu not in ('0','1','2'):raise ValueError('Unknown method or GPU')
d=O/METHODS[name];d.mkdir(parents=True,exist_ok=True);(d/'running.json').write_text(json.dumps({'started':time.time(),'gpu':int(gpu)}))
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PHENONICHE_BASELINE_GPU=gpu,PHENONICHE_BASELINE_K='5',PHENONICHE_BASELINE_DATA=str(S),PHENONICHE_BASELINE_OUTPUT=str(O),PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
with (d/'run.log').open('a',buffering=1) as log:r=subprocess.run([sys.executable,str(ROOT/'scripts/run_interface_baselines.py'),'--action','worker','--method',name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
(d/'running.json').unlink(missing_ok=True)
if r.returncode:(d/'failed.json').write_text(json.dumps({'exit_code':r.returncode,'log':str(d/'run.log')}))
else:(d/'failed.json').unlink(missing_ok=True)
render();print(name,'exit',r.returncode,flush=True);sys.exit(r.returncode)
