"""Fetch the public 11-patient CosMx 6K dataset, with verified NAS storage."""
from pathlib import Path
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone

DEST = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/cosmx_early_crc_nature2026')
RECORD = 'https://zenodo.org/api/records/20136513'
ARCHIVE = 'Nanostring_CosMx_exp2.zip'
EXPECTED_MD5 = '8a128cdff99e61873a4b6405925f45f7'
EXPECTED_BYTES = 2337829570
SUPPLEMENT = ('https://media.springernature.com/original/springer-static/esm/'
    'art%3A10.1038%2Fs41586-026-10344-7/MediaObjects/41586_2026_10344_MOESM3_ESM.xlsx')


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def checksum(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fetch(url, path):
    subprocess.run([shutil.which('aria2c') or 'aria2c', '--continue=true',
        '--allow-overwrite=false', '--auto-file-renaming=false', '--max-connection-per-server=4',
        '--split=4', '--min-split-size=16M', '--max-tries=6', '--retry-wait=15',
        '--connect-timeout=30', '--timeout=60', '--summary-interval=30',
        '--console-log-level=warn', '--enable-color=false', '--file-allocation=none',
        '--dir=' + str(path.parent), '--out=' + path.name, url], check=True)


def run():
    for name in ('raw', 'metadata', 'logs'):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    with (DEST / 'logs/download.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (DEST / 'complete.json').exists():
            print('Verified download already complete:', DEST, flush=True)
            return
        status_path = DEST / 'logs/status.json'
        def status(stage, **kwargs):
            save(status_path, dict(status=stage, pid=os.getpid(),
                updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs))
            print(stage, kwargs, flush=True)
        try:
            status('fetching_manifest')
            with urllib.request.urlopen(RECORD, timeout=30) as response:
                record = json.load(response)
            save(DEST / 'metadata/zenodo_record.json', record)
            files = {f['key']: f for f in record['files']}
            info = files[ARCHIVE]
            if info['size'] != EXPECTED_BYTES or info['checksum'] != 'md5:' + EXPECTED_MD5:
                raise RuntimeError('Published archive identity changed; refuse an unreviewed replacement')
            target = DEST / 'raw' / ARCHIVE
            status('downloading', expected_bytes=EXPECTED_BYTES, file=str(target))
            if not target.exists() or target.stat().st_size != EXPECTED_BYTES or target.with_suffix('.zip.aria2').exists():
                fetch(info['links']['self'], target)
            status('verifying_md5')
            observed = checksum(target, 'md5')
            if target.stat().st_size != EXPECTED_BYTES or observed != EXPECTED_MD5:
                raise RuntimeError('Downloaded archive size or MD5 does not match Zenodo')
            readme = DEST / 'metadata/author_README.md'
            if not readme.exists():
                fetch(files['README.md']['links']['self'], readme)
            if 'md5:' + checksum(readme, 'md5') != files['README.md']['checksum']:
                raise RuntimeError('Author README checksum mismatch')
            supplement = DEST / 'metadata/Supplementary_Tables_1_8.xlsx'
            if not supplement.exists():
                fetch(SUPPLEMENT, supplement)
            status('extracting_and_checking_zip_crc')
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    output = (DEST / 'raw' / member.filename).resolve()
                    if not output.is_relative_to((DEST / 'raw').resolve()):
                        raise RuntimeError('Unsafe archive member path')
                    # Reading each complete entry during extraction also verifies its CRC.
                    archive.extract(member, DEST / 'raw')
                entries = [dict(name=m.filename, bytes=m.file_size, crc32=m.CRC)
                           for m in archive.infolist() if not m.is_dir()]
            patients = DEST / 'raw/Nanostring_CosMx_exp2/tables/cosmx_exp2_patient_metadata.tsv'
            with patients.open() as source:
                rows = list(csv.DictReader(source, delimiter='\t'))
            if len(rows) != 11 or len({r['patient_ID'] for r in rows}) != 11:
                raise RuntimeError('Unexpected patient metadata contents')
            from openpyxl import load_workbook
            workbook = load_workbook(supplement, read_only=True, data_only=True)
            sheet = workbook['Table 1 Patients']
            records = list(sheet.iter_rows(values_only=True))
            header = list(records[0]); aliases = header.index('Alternative patient IDs')
            clinical = {}
            for row in records[1:]:
                for alias in str(row[aliases] or '').split(','):
                    clinical[alias.strip()] = dict(zip(header, row))
            with (DEST / 'metadata/patients_11_clinical.tsv').open('w') as output:
                writer = csv.DictWriter(output, fieldnames=['patient_ID', 'slideName', 'tumorStage'] + header,
                                        delimiter='\t')
                writer.writeheader()
                for row in rows:
                    if row['patient_ID'] not in clinical:
                        raise RuntimeError('No exact clinical-table patient ID match')
                    writer.writerow(dict(row, **clinical[row['patient_ID']]))
            workbook.close()
            manifest = dict(status='complete', source_record='https://zenodo.org/records/20136513',
                paper_doi='10.1038/s41586-026-10344-7', archive=str(target), bytes=EXPECTED_BYTES,
                md5=observed, zip_crc_verified=True, files=entries, patients=11,
                supplement_url=SUPPLEMENT, supplement_sha256=checksum(supplement, 'sha256'),
                completed_utc=datetime.now(timezone.utc).isoformat(), analysis_started=False)
            save(DEST / 'complete.json', manifest)
            status('complete', path=str(DEST), md5_verified=True, zip_crc_verified=True, patients=11)
        except BaseException as exc:
            status('failed', error=str(exc))
            raise


if __name__ == '__main__':
    run()
