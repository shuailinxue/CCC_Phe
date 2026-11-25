"""Resume and validate the author-shared COAD Xenium transcriptome archive."""
import fcntl
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

ROOT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/spatch_coad_xenium5k')
SHARE = 'Pdrp2CtWvxT'


def download(root=ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'transcriptome.zip'
    def status(**values):
        p = root / 'download_status.json'
        t = p.with_suffix('.tmp'); t.write_text(json.dumps(values, indent=2)+'\n'); t.replace(p)
    with (root / 'download.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        done = root / 'archive_complete.json'
        if done.exists() and target.exists():
            record = json.loads(done.read_text())
            if target.stat().st_size == record['bytes']:
                print('Read verified author archive:', target, flush=True)
                return target
            raise ValueError('Archive no longer matches its completion record')
        base = 'https://bj21400.api.aliyunfile.com'
        s = requests.Session()
        response = s.post(base+'/v2/share_link/get_share_token',
            json={'share_id': SHARE, 'ignoreError': True}, timeout=45)
        response.raise_for_status(); token = response.json()['share_token']
        response = s.post(base+'/v2/file/list', headers={'x-share-token': token},
            json={'limit': 100, 'marker': '', 'share_id': SHARE, 'parent_file_id': 'root',
                  'fields': '*', 'url_expire_sec': 7200}, timeout=45)
        response.raise_for_status()
        choices = [f for f in response.json()['items'] if f['name'] == 'transcriptome.zip']
        if len(choices) != 1:
            raise ValueError('Expected a single author transcriptome archive')
        item = choices[0]; size = int(item['size'])
        metadata = {k: item.get(k) for k in ('name', 'size', 'file_id', 'content_hash', 'content_hash_name')}
        (root / 'source_manifest.json').write_text(json.dumps(dict(share_id=SHARE,
            portal='https://spatch.pku-genomics.org/#/dataset/xenium', dataset_id=3,
            table='xenium_coad', item=metadata), indent=2)+'\n')
        partial = root / 'transcriptome.zip.partial'
        # Refresh the public download URL after failures; completed byte ranges
        # remain on NAS and are never reported as a completed dataset.
        for attempt in range(6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise ValueError('Partial archive exceeds published size')
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            try:
                with s.get(item['download_url'], headers=headers, stream=True, timeout=(30, 90)) as reply:
                    reply.raise_for_status()
                    if offset and reply.status_code != 206:
                        raise ValueError('Server did not honor resume Range; partial file preserved')
                    if offset and not reply.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Invalid download Content-Range')
                    last = offset
                    with partial.open('ab' if offset else 'wb') as handle:
                        for block in reply.iter_content(4 << 20):
                            if block:
                                handle.write(block); offset += len(block)
                                if offset-last >= 20 << 20:
                                    status(status='downloading', bytes=offset, total_bytes=size)
                                    print(f'Download {offset/size:.1%}: {offset:,}/{size:,} bytes', flush=True)
                                    last = offset
                break
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(2)
        if not partial.exists() or partial.stat().st_size != size:
            raise ValueError('Downloaded archive size does not match the author listing')
        status(status='validating_archive', total_bytes=size)
        with zipfile.ZipFile(partial) as archive:
            failed = archive.testzip()
            if failed:
                raise ValueError(f'ZIP CRC failed: {failed}')
            entries = [dict(name=i.filename, bytes=i.file_size, crc32=i.CRC) for i in archive.infolist()]
        sha = hashlib.sha256()
        with partial.open('rb') as handle:
            for block in iter(lambda: handle.read(4 << 20), b''):
                sha.update(block)
        partial.replace(target)
        record = dict(bytes=size, sha256=sha.hexdigest(), zip_crc_verified=True, entries=entries,
                      source=metadata)
        done.write_text(json.dumps(record, indent=2)+'\n')
        status(status='complete', **record)
        print('Archive verified:', target, flush=True)
        print('Entries:', entries, flush=True)
        return target


if __name__ == '__main__':
    download()
