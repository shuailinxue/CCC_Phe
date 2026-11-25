"""Publish native consensus only after its running worker has finished."""
import json,os,sys,time
from pathlib import Path
out=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/tumor_immune_interface/v2_baseline_k5/commuspace')
pid=int(sys.argv[1])
while Path(f'/proc/{pid}').exists():time.sleep(5)
info=json.loads((out/'complete.json').read_text())
if not info.get('clustering','').startswith('native CommuSpace'):
 print('Consensus did not complete; old K-means result remains hidden.',flush=True);sys.exit(1)
(out/'labels_kmeans.npy').unlink(missing_ok=True)
(out/'consensus_pending').unlink(missing_ok=True)
from render_communication_baselines import render
render();print('Published native CommuSpace consensus only.',flush=True)
