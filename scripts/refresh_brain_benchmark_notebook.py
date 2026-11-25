from pathlib import Path
import nbformat
from nbclient import NotebookClient
p=Path('/home/xueshuailin/CCC_Phe/Brain_V3_scNiche_Harmonics_Benchmark.ipynb');nb=nbformat.read(p,as_version=4);NotebookClient(nb,timeout=300,kernel_name='cccphe',allow_errors=False).execute(cwd=str(p.parent));nbformat.write(nb,p);print('Notebook outputs updated',flush=True)
