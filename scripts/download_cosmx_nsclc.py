"""Fetch the full eight-section public NSCLC AnnData to NAS, with MD5 audit."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import time
import requests

ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/cosmx_nsclc')
ARTICLE='https://api.figshare.com/v2/articles/25976224'
NAME='cosmx_human_nsclc_clustered.h5ad'


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'download.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metadata=requests.get(ARTICLE,timeout=60);metadata.raise_for_status();metadata=metadata.json()
        (ROOT/'source_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        item=next(f for f in metadata['files'] if f['name']==NAME)
        path=ROOT/NAME;partial=ROOT/(NAME+'.partial');total=int(item['size'])
        def report(**kw):
            record=dict(file=NAME,total_bytes=total,pid=os.getpid(),**kw)
            (ROOT/'download_progress.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
        if not (path.exists() and path.stat().st_size==total):
            for attempt in range(8):
                try:
                    done=partial.stat().st_size if partial.exists() else 0
                    if done==total:break
                    url=item['download_url']+f'?download_attempt={int(time.time())}_{attempt}'
                    headers={'Range':f'bytes={done}-'} if done else {}
                    with requests.get(url,headers=headers,stream=True,timeout=(30,180)) as response:
                        response.raise_for_status()
                        if done and response.status_code!=206:done=0
                        if done and not response.headers.get('Content-Range','').startswith(f'bytes {done}-'):
                            raise IOError('Resume server range mismatch')
                        report(stage='downloading',downloaded_bytes=done)
                        with partial.open('ab' if done else 'wb') as output:
                            last=time.monotonic()
                            for block in response.iter_content(4<<20):
                                output.write(block);done+=len(block)
                                if time.monotonic()-last>10:
                                    report(stage='downloading',downloaded_bytes=done);last=time.monotonic()
                    if done!=total:raise IOError(f'Incomplete file: {done}/{total}')
                    break
                except Exception as exc:
                    report(stage='retry',attempt=attempt,error=str(exc))
                    if attempt==7:raise
                    time.sleep(min(2**attempt,30))
            partial.replace(path)
        md5=hashlib.md5();sha=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8<<20),b''):md5.update(block);sha.update(block)
        if md5.hexdigest()!=item['computed_md5']:raise ValueError('Downloaded NSCLC MD5 mismatch')
        report(stage='complete',downloaded_bytes=path.stat().st_size,md5=md5.hexdigest(),sha256=sha.hexdigest(),
            source=ARTICLE,source_doi=metadata['doi'],license=metadata['license'],
            usage='Expression counts, physical coordinates and author cell annotations only; existing spatial clusters/embeddings excluded from learning')
    return path


if __name__=='__main__':run()
