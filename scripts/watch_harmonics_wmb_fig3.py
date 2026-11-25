"""Wait for the existing WMB worker; refresh existing notebook on success."""
import os,time,sys,subprocess
from pathlib import Path
pid=int(sys.argv[1]); root=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/harmonics_wmb/fig3_slice008_v3'); print('Watching PID',pid,flush=True)
while not (root/'complete.json').exists():
 if (root/'failure.txt').exists():print('Worker reported failure',flush=True);sys.exit(1)
 try:os.kill(pid,0)
 except ProcessLookupError:print('Worker exited before completion',flush=True);sys.exit(1)
 time.sleep(30)
subprocess.run([sys.executable,'/home/xueshuailin/CCC_Phe/scripts/refresh_harmonics_wmb_notebook.py'],check=True)
