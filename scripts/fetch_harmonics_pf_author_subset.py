"""Fetch named members of the public author archive using HTTP byte ranges."""
import io,zipfile,requests
from pathlib import Path
ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/harmonics_pf/author');ROOT.mkdir(parents=True,exist_ok=True)
class Remote(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;r=requests.get(url,headers={'Range':'bytes=-1'},timeout=30);r.raise_for_status();self.size=int(r.headers['Content-Range'].split('/')[-1])
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,x,whence=0):self.pos=x if whence==0 else self.pos+x if whence==1 else self.size+x;return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  if n==0:return b''
  r=requests.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=180);r.raise_for_status()
  if r.status_code!=206:raise ValueError('server did not honor range request')
  data=r.content;self.pos+=len(data);return data
archives=[('16728860','Harmonics_reproducibility.zip',['Harmonics_reproducibility/Xenium_PF_Vannan2025/checkdataset.ipynb','Harmonics_reproducibility/Xenium_PF_Vannan2025/comparison.ipynb','Harmonics_reproducibility/Xenium_PF_Vannan2025/runNicheCompass.ipynb']),('16794669','Data.zip',['Data/Spatial/Transcriptomics/Xenium_PF_Vannan2025/processed/slice_name_list.txt'])]
for record,name,members in archives:
 z=zipfile.ZipFile(Remote(f'https://zenodo.org/api/records/{record}/files/{name}/content'))
 for member in members:
  dest=ROOT/Path(member).name
  if dest.exists():print('cached',dest,flush=True);continue
  print('fetch',member,flush=True);content=z.read(member);dest.write_bytes(content);print('saved',dest,len(content),flush=True)
