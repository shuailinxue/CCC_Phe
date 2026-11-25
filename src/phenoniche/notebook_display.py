"""Memory-conscious inline previews without modifying source figure files."""
from io import BytesIO
from pathlib import Path

from IPython.display import Image as IPythonImage
from PIL import Image


MAX_PREVIEW_SIZE = (1600, 1600)
MAX_DISPLAY_WIDTH = 1200


def notebook_image(*, filename, width=None):
    """Read a figure into a bounded PNG preview for notebook display.

    The original PNG/PDF result is untouched. For small PNGs the original bytes
    are reused; large images are downsampled in memory while preserving aspect.
    """
    path = Path(filename)
    with Image.open(path) as img:
        if img.width <= MAX_PREVIEW_SIZE[0] and img.height <= MAX_PREVIEW_SIZE[1] and img.format == 'PNG':
            payload = path.read_bytes()
            preview_width = img.width
        else:
            img.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
            preview_width = img.width
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            payload = buffer.getvalue()
    display_width = min(int(width or preview_width), preview_width, MAX_DISPLAY_WIDTH)
    return IPythonImage(data=payload, format='png', width=display_width)
