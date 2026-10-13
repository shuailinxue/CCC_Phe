"""Reuse exact simulated inputs and frozen controls across CCC-only experiments."""
from pathlib import Path
import hashlib,json,shutil
from .workflow import protect


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def reuse_inputs(source,config):
    source=Path(source);output=protect(config)
    names=('before_injection.h5ad','injected.h5ad','lr.csv','seed_reference.h5ad','expression_fidelity.json','input/ccc.npy')
    identity={}
    for name in names:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source/name,target)
        expected=digest(source/name)
        if digest(target)!=expected:raise ValueError('Experiment data changed: '+name)
        identity[name]=expected
    for name in ('Z.npy','P.npy','H.npy','activity.npy','complete.json'):
        target=output/'baselines/ccc_ae'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(source/'baselines/ccc_ae'/name,target)
    (output/'shared_input_manifest.json').write_text(json.dumps({'source':str(source),'shared_data_sha256':identity,'controls':'AE uses original train-only RMS normalization and frozen outputs'},indent=2)+'\n')
    return output
