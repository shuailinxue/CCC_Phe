"""Pinned scCube preparation without downgrading the shared cccphe environment."""
from pathlib import Path
import json,hashlib,importlib.util,subprocess,sys,urllib.request
from ..storage import STORAGE_ROOT
from .config import ROOT
from .generation import check_environment

def prepare_environment():
    # Normal Notebook execution is offline when the prepared package is present.
    requirements=[]
    for module,spec in [('scanpy','scanpy>=1.10'),('ot','POT>=0.9'),('tqdm','tqdm>=4.66')]:
        if importlib.util.find_spec(module) is None:requirements.append(spec)
    if requirements:subprocess.run([sys.executable,'-m','pip','install',*requirements],check=True)
    audit=json.loads((ROOT/'data/scCube_source_audit.json').read_text())
    valid=False
    if importlib.util.find_spec('scCube') is not None:
        import scCube
        package=Path(scCube.__file__).parent
        valid=all((package/Path(name).name).is_file() and hashlib.sha256((package/Path(name).name).read_bytes()).hexdigest()==digest for name,digest in audit['files'].items())
    if not valid:
        source=STORAGE_ROOT/'artifacts/dependencies/scCube_pinned';source.mkdir(exist_ok=True,parents=True)
        for name in [*audit['files'],'setup.py']:
            path=source/name;path.parent.mkdir(exist_ok=True,parents=True)
            if name in audit['files'] and path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==audit['files'][name]:continue
            url=f"https://raw.githubusercontent.com/ZJUFanLab/scCube/{audit['commit']}/{name}"
            with urllib.request.urlopen(url,timeout=45) as response:content=response.read()
            if name in audit['files'] and hashlib.sha256(content).hexdigest()!=audit['files'][name]:raise RuntimeError('scCube source checksum mismatch')
            path.write_bytes(content)
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--force-reinstall',str(source)],check=True)
        from importlib import invalidate_caches
        invalidate_caches()
        # If an incompatible scCube was imported for verification, ensure the
        # next import loads the newly installed pinned module.
        for name in list(sys.modules):
            if name=='scCube' or name.startswith('scCube.'):del sys.modules[name]
    return check_environment()
