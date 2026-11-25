"""Replay preserved notebook displays without executing retired model code."""
import json
from pathlib import Path
from ..storage import STORAGE_ROOT

DISPLAY_ROOT=Path(STORAGE_ROOT)/'Backup/project_artifacts/notebook_displays'


def show_saved_display(notebook, cell_id):
    from IPython.display import display
    path=DISPLAY_ROOT/(Path(notebook).stem+'.json')
    value=json.loads(path.read_text())
    if value['notebook']!=notebook:
        raise ValueError('Historical display notebook mismatch')
    for output in value['cells'][str(cell_id)]:
        kind=output['output_type']
        if kind in ('display_data','execute_result'):
            display(output['data'],raw=True,metadata=output.get('metadata',{}))
        elif kind=='stream':
            print(output.get('text',''),end='')
        elif kind=='error':
            print('历史执行错误：'+output.get('ename','')+': '+output.get('evalue',''))
