from pathlib import Path
import subprocess,sys,nbformat
from nbclient import NotebookClient
root=Path('/home/xueshuailin/CCC_Phe');subprocess.run([sys.executable,str(root/'scripts/run_cpdb_global_communication_ablation.py')],cwd=root,check=True)
p=root/'Multifocal_Communication_Simulation.ipynb';n=nbformat.read(p,4);work=nbformat.v4.new_notebook(cells=[n.cells[0],n.cells[3]],metadata=n.metadata);NotebookClient(work,timeout=600,kernel_name='cccphe',resources={'metadata':{'path':str(root)}}).execute();n=nbformat.read(p,4);n.cells[3]=work.cells[1];nbformat.write(n,p);print('Notebook global ablation updated',flush=True)
