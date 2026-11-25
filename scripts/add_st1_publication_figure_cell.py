"""Append the completed ST1 border figure without regenerating other cells."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path('/home/xueshuailin/CCC_Phe')
NB=ROOT/'Cellist_ST1_V2_Communication_Niches.ipynb'
MARKER='# ST1 上下边缘：空间分块统计与主图'
code='''# ST1 上下边缘：空间分块统计与主图
from pathlib import Path
import pandas as pd
from IPython.display import display, Image
OUT=Path('/data1/xueshuailin/CCC_Phe/workspace/artifacts/st1_upper_lower_publication')
display(Image(filename=str(OUT/'main_figure.png'), width=1400))
display(pd.read_csv(OUT/'niche_composition_stats.csv').round(4))
print((OUT/'interpretation.txt').read_text())
print('完整通讯结果、边缘宽度与分块大小敏感性分析已另存为 CSV；同切片分块检验为探索性分析。')
'''
nb=nbformat.read(NB,4)
existing=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith(MARKER)]
new=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(code)],metadata=nb.metadata)
NotebookClient(new,timeout=300,kernel_name='cccphe',allow_errors=False).execute(cwd=str(ROOT))
new.cells[0].execution_count=None
if existing:nb.cells[existing[0]]=new.cells[0]
else:nb.cells.append(new.cells[0])
nbformat.write(nb,NB)
print('Updated',NB)
