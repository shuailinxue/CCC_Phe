"""Append one completed analysis cell without restoring deleted notebook cells."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path('/home/xueshuailin/CCC_Phe')
NB=ROOT/'Cellist_ST1_V2_Communication_Niches.ipynb'
MARKER='# ST1 组成匹配与通讯额外信息'
CODE=r'''# ST1 组成匹配与通讯额外信息
from pathlib import Path
import json, pandas as pd
from IPython.display import display, Image, Markdown
OUT=Path('/data1/xueshuailin/CCC_Phe/workspace/artifacts/st1_c4_independent_communication')
display(Image(filename=str(OUT/'figure_c4_communication.png'),width=1500))
display(Markdown((OUT/'analysis_report.md').read_text()))
display(Image(filename=str(OUT/'mechanism_sensitivity.png'),width=1300))
display(pd.read_csv(OUT/'region_adjustment_sensitivity.csv').round(3))
print('完整的组成匹配、通路与 LR 统计、留出响应基因消融，以及模型输入和空间分块诊断均已保存为 CSV。')
'''
nb=nbformat.read(NB,4)
old=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith(MARKER)]
one=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(CODE)],metadata=nb.metadata)
NotebookClient(one,timeout=300,kernel_name='cccphe',allow_errors=False).execute(cwd=str(ROOT));one.cells[0].execution_count=None
if old:nb.cells[old[0]]=one.cells[0]
else:nb.cells.append(one.cells[0])
nbformat.write(nb,NB);print('Updated',NB)
