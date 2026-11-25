"""Read a public HDF5 object's needed byte ranges into a direct NAS cache."""
import io,hashlib,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CachedRangeReader(io.RawIOBase):
    def __init__(self,url,size,cache,block_size=4*1024**2,prefetch_blocks=1):
        super().__init__();self.url=url;self.size=int(size);self.position=0;self.block_size=block_size
        key=hashlib.sha256(url.encode()).hexdigest()[:16]
        if block_size!=4*1024**2:key+=f'-block{block_size}'
        self.cache=Path(cache)/key;self.cache.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.fetched_bytes=0
        self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
        self._memory=OrderedDict()
        self.prefetch_blocks=max(1,min(int(prefetch_blocks),4));self._pending={}
        self._pool=ThreadPoolExecutor(max_workers=self.prefetch_blocks) if self.prefetch_blocks>1 else None
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if target<0:raise ValueError('Negative seek')
        self.position=target;return target
    def _block(self,index):
        if index in self._memory:
            data=self._memory.pop(index);self._memory[index]=data;return data
        start=index*self.block_size;stop=min(start+self.block_size,self.size)-1;path=self.cache/f'{index:06d}.bin'
        if path.exists():
            data=path.read_bytes()
            if len(data)!=stop-start+1:raise IOError('Corrupt cached HTTP range')
            return self._remember(index,data)
        if self._pool is not None:
            # Public, immutable byte ranges only. At most four simultaneous
            # requests; HDF5 itself remains on the calling thread.
            for old in list(self._pending):
                if old!=index and self._pending[old].done():del self._pending[old]
            for next_index in range(index,min(index+self.prefetch_blocks,(self.size+self.block_size-1)//self.block_size)):
                if next_index not in self._pending and not (self.cache/f'{next_index:06d}.bin').exists():
                    self._pending[next_index]=self._pool.submit(self._download,next_index)
            data=self._pending.pop(index).result()
        else:data=self._download(index)
        return self._remember(index,data)
    def _download(self,index):
        start=index*self.block_size;stop=min(start+self.block_size,self.size)-1;path=self.cache/f'{index:06d}.bin'
        # Unique public query avoids intermediaries serving a different range.
        url=self.url+('&' if '?' in self.url else '?')+f'byte_range_start={start}'
        for attempt in range(3):
            response=None
            try:
                request_url=url+(f'&range_attempt={attempt}' if attempt else '')
                response=self.session.get(request_url,headers={'Range':f'bytes={start}-{stop}'},stream=True,timeout=(20,120))
                if response.status_code!=206:raise IOError(f'Server did not honor public Range request: {response.status_code}')
                expected=f'bytes {start}-{stop}/{self.size}'
                if response.headers.get('Content-Range')!=expected:raise IOError('Unexpected HTTP Content-Range')
                data=b''.join(response.iter_content(1024**2))
                if len(data)!=stop-start+1:raise IOError('Truncated HTTP range')
                break
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
            finally:
                if response is not None:response.close()
        temporary=path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.incomplete');temporary.write_bytes(data);temporary.replace(path);self.fetched_bytes+=len(data);return data
    def _remember(self,index,data):
        self._memory[index]=data
        while len(self._memory)>8:self._memory.popitem(last=False)
        return data
    def read(self,size=-1):
        if size is None or size<0:size=self.size-self.position
        stop=min(self.position+size,self.size);pieces=[]
        while self.position<stop:
            index=self.position//self.block_size;block=self._block(index);offset=self.position%self.block_size
            take=min(stop-self.position,len(block)-offset);pieces.append(block[offset:offset+take]);self.position+=take
        return b''.join(pieces)
    def readinto(self,buffer):
        data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)
    def close(self):
        if not self.closed:
            if self._pool is not None:self._pool.shutdown(wait=True,cancel_futures=True)
            self.session.close()
        super().close()
