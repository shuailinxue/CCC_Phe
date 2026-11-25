"""Fetch exactly the two samples displayed in the Harmonics PF figure."""
import zipfile
from pathlib import Path
from fetch_harmonics_pf_author_subset import Remote

root = Path('/data1/xueshuailin/CCC_Phe/workspace/data/harmonics_pf/figure_samples')
root.mkdir(parents=True, exist_ok=True)
url = 'https://zenodo.org/api/records/16794669/files/Data.zip/content'
with zipfile.ZipFile(Remote(url)) as z:
    for sample in ('VUHD049', 'TILD117LA'):
        member = f'Data/Spatial/Transcriptomics/Xenium_PF_Vannan2025/processed/{sample}.h5ad'
        dest = root / f'{sample}.h5ad'
        if dest.exists() and dest.stat().st_size == z.getinfo(member).file_size:
            print('Verified existing', dest, flush=True)
            continue
        tmp = dest.with_suffix('.h5ad.part')
        print('Downloading', sample, flush=True)
        with z.open(member) as source, tmp.open('wb') as target:
            while block := source.read(4 * 1024 * 1024):
                target.write(block)
        if tmp.stat().st_size != z.getinfo(member).file_size:
            raise ValueError('Downloaded file size mismatch')
        tmp.replace(dest)
        print('Downloaded', dest, dest.stat().st_size, flush=True)
