"""Wait for the current fixed-K ST1 run, then evaluate granularity once."""
from pathlib import Path
import os,sys,time
O=Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/nsclc_zhencaiwei/ST1/v2')
PID=int(sys.argv[1])
def running():
 try:os.kill(PID,0);return True
 except ProcessLookupError:return False
while True:
 if (O/'pipeline_complete.json').exists():
  print('ST1 fixed-K run complete; starting granularity analysis',flush=True)
  os.execv(sys.executable,[sys.executable,'-u','scripts/evaluate_cellist_st1_granularity.py'])
 if not running():
  print('ST1 fixed-K process ended without completion; granularity analysis not started',flush=True)
  sys.exit(1)
 time.sleep(20)
