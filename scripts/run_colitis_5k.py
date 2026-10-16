from pathlib import Path
from phenoniche.cohorts.colitis import run
if __name__=='__main__':run(Path(__file__).resolve().parents[1]/'config/colitis_5k.json')
