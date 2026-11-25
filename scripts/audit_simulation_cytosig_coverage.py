"""Measure simulated assay-panel overlap with the public CytoSig signature."""
from pathlib import Path
import hashlib
import io
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from phenoniche.shared.io import write_json

WORK=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
SOURCE=WORK/'data/tumor_immune_interface/tissue.h5ad'
OUT=WORK/'outputs/tumor_immune_interface/ccc_nmf_program_pilot'


def run():
    api='https://api.github.com/repos/data2intelligence/CytoSig/commits/master'
    revision=requests.get(api,timeout=30);revision.raise_for_status();commit=revision.json()['sha']
    url=f'https://raw.githubusercontent.com/data2intelligence/CytoSig/{commit}/CytoSig/signature.centroid'
    response=requests.get(url,timeout=30);response.raise_for_status();payload=response.content
    signature=pd.read_csv(io.BytesIO(payload),sep='\t',index_col=0)
    a=ad.read_h5ad(SOURCE,backed='r');panel=set(a.var_names.astype(str).str.upper());a.file.close()
    names=signature.index.astype(str).str.upper();present=names.isin(panel);rows=[]
    for column in signature.columns:
        values=np.abs(signature[column].to_numpy(dtype=float));top=np.argsort(-values)[:100]
        rows.append(dict(program=column,covered_in_top100=int(present[top].sum()),
                         full_abs_weight_fraction=float(values[present].sum()/max(values.sum(),1e-12))))
    frame=pd.DataFrame(rows);OUT.mkdir(parents=True,exist_ok=True);frame.to_csv(OUT/'cytosig_simulation_panel_coverage.csv',index=False)
    info=dict(source=url,commit=commit,sha256=hashlib.sha256(payload).hexdigest(),signature_genes=len(signature),
              signature_programs=len(signature.columns),simulation_panel_genes=len(panel),overlap_genes=int(present.sum()),
              coverage_only=True,caveat='Gene presence and weight coverage do not establish reliable response prediction; expression and panel-specific calibration are needed.')
    write_json(OUT/'cytosig_simulation_panel_coverage.json',info)
    print(json.dumps(info,indent=2));print(frame[frame.program.str.contains('IFN|TGFB',case=False)].to_string(index=False))


if __name__=='__main__':run()
