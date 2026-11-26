"""Post-fit evaluation. Only this command takes author labels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from phenoniche.v1.comparison import compare

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",nargs=2,action="append",metavar=("NAME","DIR"),required=True)
    p.add_argument("--input",required=True)
    p.add_argument("--label-key",required=True)
    p.add_argument("--classes",nargs="+",required=True)
    p.add_argument("--root",required=True)
    p.add_argument("--coarse")
    p.add_argument("--v0",help="Existing V0 NPZ with cell_ids and labels on exactly the evaluation subset")
    a = p.parse_args()
    print(compare(dict(a.run),a.input,a.label_key,a.classes,a.root,a.coarse,a.v0))
