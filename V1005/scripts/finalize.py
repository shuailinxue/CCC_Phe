"""Require successful tests and both complete runs before notebook execution (cross-dataset matching is paused)."""
import os,sys,json,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['NUMBA_CACHE_DIR']=str(root/'outputs/cache/numba')

def main():
    for ds in ['hbc1','xenium5k']:
        if not (root/'outputs'/ds/'complete.json').exists():raise RuntimeError(f'{ds}: full experiment incomplete')
    result=subprocess.run([sys.executable,'-m','pytest',str(root/'tests'),'-q','-o',f'cache_dir={root}/outputs/pytest_cache','--basetemp='+str(root/'outputs/pytest_tmp')],capture_output=True,text=True)
    (root/'outputs/pytest.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Tests failed; notebook not executed')
    from phenoniche.v1005.data import input_audit
    for ds in ['hbc1','xenium5k']:input_audit(ds)
    from phenoniche.v1005.evaluation import representation_diagnostics,finalize_table_labels
    for ds in ['hbc1','xenium5k']:
        representation_diagnostics(ds);finalize_table_labels(ds)
    import importlib.metadata as metadata
    versions={name:metadata.version(name) for name in ['numpy','scipy','pandas','torch','scikit-learn','anndata','igraph','leidenalg','pynndescent','umap-learn','matplotlib','nbclient']}
    (root/'outputs/environment_versions.json').write_text(json.dumps(versions,indent=2))
    import nbformat
    from nbclient import NotebookClient
    npath=root/'V1005_breast_niche_walkthrough.ipynb';nb=nbformat.read(npath,as_version=4)
    client=NotebookClient(nb,timeout=1800,resources={'metadata':{'path':str(root)}},kernel_name='python3')
    try:client.execute()
    finally:nbformat.write(nb,npath)
    before=json.loads((root/'outputs/protected_versions_before.json').read_text());after={}
    for version in ['V1000','V1001','V1002','V1003','V1004']:
        for p in (root.parent/version).rglob('*'):
            if p.is_file() and not any(x in p.parts for x in ['__pycache__','.pytest_cache']):
                st=p.stat();after[str(p.relative_to(root.parent))]={'size':st.st_size,'mtime_ns':st.st_mtime_ns}
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    (root/'outputs/protected_versions_check.json').write_text(json.dumps({'unchanged':not changed,'changed':changed},indent=2))
    if changed:raise RuntimeError('Protected files changed; inspect protected_versions_check.json')
    (root/'outputs/final_status.json').write_text(json.dumps({'pytest':'passed','notebook':'executed','protected_versions':'unchanged','cross_dataset':'deferred_by_user'},indent=2))
    print('Tests passed; notebook executed; protected versions unchanged.',flush=True)
if __name__=='__main__':main()
