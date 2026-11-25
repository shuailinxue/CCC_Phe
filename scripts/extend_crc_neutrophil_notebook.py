"""Validate the sole V1 CRC notebook; never append retired model cells."""
from pathlib import Path
import nbformat
PROJECT=Path(__file__).resolve().parents[1]
BOOK=PROJECT/'Xenium_CRC_Neutrophil_Atlas.ipynb'
def build():
    book=nbformat.read(BOOK,as_version=4)
    if book.metadata.get('crc_model_backend')!='gnn':raise ValueError('V1 GNN notebook required')
    nbformat.validate(book)
    return BOOK
if __name__=='__main__':print(build())
