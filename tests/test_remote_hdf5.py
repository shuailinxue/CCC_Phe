"""Exact byte-range integrity; a server ignoring Range must never be cached."""
import pytest
from phenoniche.cohorts.remote_hdf5 import CachedRangeReader


@pytest.mark.parametrize('prefetch',[1,4])
def test_cached_seek_and_read_match_original_bytes(tmp_path,monkeypatch,prefetch):
    original=bytes(range(256))*5;calls=[]
    class Response:
        status_code=206
        def __init__(self,start,stop):
            self.value=original[start:stop+1];self.headers={'Content-Range':f'bytes {start}-{stop}/{len(original)}'}
        def iter_content(self,size):yield self.value
        def close(self):pass
    def request(url,headers,**kwargs):
        start,stop=map(int,headers['Range'].split('=')[1].split('-'));calls.append((start,stop));return Response(start,stop)
    with CachedRangeReader('https://example.org/public.h5',len(original),tmp_path,block_size=64,prefetch_blocks=prefetch) as reader:
        monkeypatch.setattr(reader.session,'get',request)
        reader.seek(55);assert reader.read(200)==original[55:255]
        reader.seek(-17,2);buffer=bytearray(30);assert reader.readinto(buffer)==17 and buffer[:17]==original[-17:]
        reader.seek(55);assert reader.read(200)==original[55:255]
    assert len(calls)==len(set(calls))
    with CachedRangeReader('https://example.org/public.h5',len(original),tmp_path,block_size=64) as reader:
        monkeypatch.setattr(reader.session,'get',lambda *args,**kwargs:pytest.fail('Cached range fetched again'))
        reader.seek(55);assert reader.read(200)==original[55:255]


def test_server_ignoring_range_is_rejected(tmp_path,monkeypatch):
    class Response:
        status_code=200
        def close(self):pass
    with CachedRangeReader('https://example.org/broken.h5',1000,tmp_path,block_size=64) as reader:
        monkeypatch.setattr(reader.session,'get',lambda *args,**kwargs:Response())
        with pytest.raises(IOError,match='honor'):reader.read(3)
        assert not list(reader.cache.glob('*.bin'))
