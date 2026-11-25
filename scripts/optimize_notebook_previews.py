"""Shrink saved notebook PNG outputs and configure bounded future previews.

Run from the project root. Does not run any cells or modify source figure files.
"""
import base64
from io import BytesIO
from pathlib import Path
import re
import nbformat
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
MAX_SIZE=(1600,1600)
MAX_WIDTH=1200
MPL_MARKER='# Notebook preview DPI: screen-sized output'
MPL_CONFIG=MPL_MARKER+"\nimport matplotlib as _notebook_mpl\n_notebook_mpl.rcParams.update({'figure.dpi': 72, 'savefig.dpi': 72})\n"
IMG_MARKER='# Bounded inline image preview; original figure file is preserved'
IMG_CONFIG=IMG_MARKER+"\nimport sys as _notebook_sys\nif '/home/xueshuailin/CCC_Phe/src' not in _notebook_sys.path:\n    _notebook_sys.path.insert(0, '/home/xueshuailin/CCC_Phe/src')\nfrom phenoniche.notebook_display import notebook_image as _notebook_image\n"


def main():
    for path in sorted(ROOT.glob('*.ipynb')):
        nb=nbformat.read(path,4);before=path.stat().st_size;changed=False;resized=0;count=0
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            source=cell.source
            if re.search(r'(?<![\w.])Image\(',source):
                source=re.sub(r'(?<![\w.])Image\(', '_notebook_image(', source)
                if IMG_MARKER not in source:source=IMG_CONFIG+source
            if ('matplotlib' in source or 'plt.' in source or re.search(r'\bshow\(',source)) and MPL_MARKER not in source:
                source=MPL_CONFIG+source
            if source!=cell.source:cell.source=source;changed=True
            for output in cell.get('outputs',[]):
                data=output.get('data',{});encoded=data.get('image/png')
                if not encoded:continue
                count+=1;raw=base64.b64decode(encoded)
                with Image.open(BytesIO(raw)) as image:
                    old_size=image.size
                    if image.width>MAX_SIZE[0] or image.height>MAX_SIZE[1]:
                        image.thumbnail(MAX_SIZE,Image.Resampling.LANCZOS)
                        buf=BytesIO();image.save(buf,format='PNG',optimize=True)
                        data['image/png']=base64.b64encode(buf.getvalue()).decode('ascii')
                        resized+=1;changed=True
                    width,height=image.size
                metadata=output.setdefault('metadata',{}).setdefault('image/png',{})
                old_width=metadata.get('width',width)
                display_width=min(MAX_WIDTH,width,old_width)
                if metadata.get('width')!=display_width:
                    if 'height' in metadata:
                        metadata['height']=round(display_width*height/width)
                    metadata['width']=display_width;changed=True
        if changed:nbformat.write(nb,path)
        print(f'{path.name}: {count} PNGs; resized {resized}; {before/1048576:.2f} -> {path.stat().st_size/1048576:.2f} MiB',flush=True)


if __name__=='__main__':main()
