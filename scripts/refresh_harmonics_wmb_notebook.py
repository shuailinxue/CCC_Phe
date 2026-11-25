"""Refresh outputs only for still-existing, unchanged notebook cells."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=Path('/home/xueshuailin/CCC_Phe/Harmonics_WMB_Fig3_V3.ipynb')
if not p.exists():raise FileNotFoundError('Notebook removed; do not recreate it')
old=nbformat.read(p,4);NotebookClient(old,timeout=600,kernel_name='cccphe').execute(cwd=str(p.parent));new=nbformat.read(p,4);executed={c.id:c for c in old.cells if c.cell_type=='code'}
for c in new.cells:
 if c.id in executed and c.source==executed[c.id].source:
  c.outputs=executed[c.id].outputs;c.execution_count=executed[c.id].execution_count
nbformat.write(new,p);print('WMB notebook outputs refreshed',flush=True)
