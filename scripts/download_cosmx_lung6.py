"""Download the original public, single-cell CosMx Lung6 flat files."""
from pathlib import Path
import hashlib,json
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/cosmx_lung6')
BASE='https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI/Lung6/'
FILES={'Lung6_exprMat_file.csv':185066835,'Lung6_metadata_file.csv':10831069,'Lung6_fov_positions_file.csv':1163,'SMI-ReadMe.html':3455929}

def fetch(item):
 name,size=item;path=ROOT/name
 if path.exists() and path.stat().st_size==size:return name
 temp=path.with_suffix(path.suffix+'.incomplete')
 response=requests.get(BASE+name,stream=True,timeout=(30,180));response.raise_for_status()
 with temp.open('wb') as f:
  for block in response.iter_content(4<<20):f.write(block)
 if temp.stat().st_size!=size:raise ValueError('Unexpected download size: '+name)
 temp.replace(path);print('Downloaded',name,size,flush=True);return name

if __name__=='__main__':
 ROOT.mkdir(parents=True,exist_ok=True)
 with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,FILES.items()))
 records=[]
 for name,size in FILES.items():
  h=hashlib.sha256()
  with (ROOT/name).open('rb') as f:
   for block in iter(lambda:f.read(4<<20),b''):h.update(block)
  records.append(dict(name=name,bytes=size,url=BASE+name,sha256=h.hexdigest()))
 (ROOT/'download_manifest.json').write_text(json.dumps({'source':'He et al. Nature Biotechnology 2022, DOI10.1038/s41587-022-01483-z','sample':'Lung6','selection':'Single real-cell section used in the NicheCompass case; chosen before model fitting; original counts and annotations only','files':records},indent=2)+'\n')
 print('Lung6 download verified',flush=True)
