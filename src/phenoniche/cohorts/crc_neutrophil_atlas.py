"""Display the published CRC Xenium cell labels without fitting any model.

Marteau et al., Cancer Cell (2026), DOI 10.1016/j.ccell.2025.12.003.
BioImage Archive S-BIAD2208. Every panel is a separate author specimen.
"""
from __future__ import annotations

from collections import OrderedDict
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
import requests

ROOT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/crc_neutrophil_xenium')
OUTPUT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium')
DATA_URL = 'https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/208/S-BIAD2208/Files/xenium/processed/crca_xenium.h5ad'
DATA_BYTES = 27337886004


class RemoteH5(io.RawIOBase):
    """Read HDF5 metadata/observations by validated HTTP ranges, in memory.

    This does not fetch expression datasets. The complete expression object is
    downloaded separately. A byte budget prevents accidentally reading that object.
    """
    def __init__(self, url=DATA_URL, size=DATA_BYTES, block_size=2 << 20):
        self.url, self.size, self.block_size = url, size, block_size
        self.position = self.fetched_bytes = 0
        self.cache = OrderedDict()
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative file position')
        return self.position

    def _block(self, number):
        if number in self.cache:
            self.cache.move_to_end(number)
            return self.cache[number]
        if self.fetched_bytes > 512 << 20:
            raise IOError('Remote metadata read exceeded its 512-MB budget')
        start = number * self.block_size
        end = min(start + self.block_size, self.size) - 1
        for attempt in range(5):
            try:
                response = self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 120))
                response.raise_for_status()
                if (response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}'
                        or len(response.content) != end - start + 1):
                    raise IOError('Unexpected HTTP range for author metadata')
                block = response.content
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
        self.fetched_bytes += len(block)
        self.cache[number] = block
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return block

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        result = bytearray()
        while size > 0:
            page = self.position // self.block_size
            block = self._block(page)
            offset = self.position - page * self.block_size
            length = min(size, len(block) - offset)
            if length <= 0:
                break
            result.extend(block[offset:offset + length])
            self.position += length
            size -= length
        return bytes(result)

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)


def _column(group, name):
    field = group[name]
    if isinstance(field, h5py.Group):
        if 'codes' not in field or 'categories' not in field:
            raise ValueError(f'Unsupported author column: {name}')
        codes = field['codes'][:]
        values = field['categories']
        categories = values.asstr()[:] if values.dtype.kind in 'OSU' else values[:]
        return pd.Categorical.from_codes(codes, categories=categories)
    if field.dtype.kind in 'OSU':
        return pd.Categorical(field.asstr()[:])
    return field[:]


def prepare_display_table(allow_remote=False):
    """Cache only author labels and aligned coordinates; retain all author rows."""
    cache = ROOT / 'processed/display_cells.parquet'
    audit_path = ROOT / 'metadata/display_audit.json'
    if cache.exists() and audit_path.exists():
        return cache
    source = ROOT / 'crca_xenium.h5ad'
    remote = None
    if source.exists():
        handle = source
    elif allow_remote:
        remote = RemoteH5()
        handle = remote
    else:
        raise FileNotFoundError('Author metadata are not ready yet; the download controller prepares this atlas automatically.')
    with h5py.File(handle, 'r') as h:
        obs = h['obs']
        available = list(obs)
        print('Inspect author fields:', available, flush=True)
        required = ['celltype', 'patient_id', 'tissue_region']
        if not all(name in obs for name in required):
            raise ValueError('Author H5AD does not have all required cell/specimen annotation fields')
        name_key = 'name' if 'name' in obs else 'sample_id' if 'sample_id' in obs else None
        if name_key is None:
            raise ValueError('No independent specimen ID in the author object')
        columns = {name: _column(obs, name) for name in required}
        columns['specimen_id'] = _column(obs, name_key)
        xy = h['obsm/spatial'][:]
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError('Expected two author physical coordinate columns')
        frame = pd.DataFrame(columns)
        if len(frame) != len(xy):
            raise ValueError('Author labels and coordinates have different row counts')
        if not np.isfinite(xy).all():
            raise ValueError('Author coordinates contain missing/nonfinite values')
        frame['source_row'] = np.arange(len(frame), dtype=np.int64)
        frame['x_um'], frame['y_um'] = xy[:, 0], xy[:, 1]
        frame['celltype'] = frame.celltype.cat.add_categories(['Unassigned']) if 'Unassigned' not in frame.celltype.cat.categories else frame.celltype
        frame['celltype'] = frame.celltype.fillna('Unassigned')
        if frame[['patient_id', 'tissue_region', 'specimen_id']].isna().any().any():
            raise ValueError('Missing author patient/specimen/region identities')
        check = frame.groupby('specimen_id', observed=True)[['patient_id', 'tissue_region']].nunique()
        if (check > 1).any().any():
            raise ValueError('A specimen ID spans different patients or tissue regions')
        index_key = obs.attrs.get('_index', '_index')
        cells = len(frame)
        if len(obs[index_key]) != cells:
            raise ValueError('Author index, labels and coordinates are not row-aligned')
        expression_present = 'counts' in h['layers']
        expression_shape = h['layers/counts'].attrs.get('shape', None) if expression_present else None
        if expression_shape is not None and int(expression_shape[0]) != cells:
            raise ValueError('Author native counts have a different cell axis')
        if not expression_present:
            raise ValueError('Author native count layer is missing')
        audit = dict(source=DATA_URL, source_expected_bytes=DATA_BYTES, cells=cells,
                     patients=int(frame.patient_id.nunique()), specimens=int(frame.specimen_id.nunique()),
                     specimen_key=name_key, available_obs_columns=available,
                     expression_shape=list(map(int, expression_shape)) if expression_shape is not None else None,
                     author_celltypes={str(k): int(v) for k, v in frame.celltype.value_counts().items()},
                     tissue_region_counts={str(k): int(v) for k, v in frame.tissue_region.value_counts().items()},
                     annotation_source='Published author obs/celltype; no inferred labels',
                     coordinates='Author obsm/spatial in micrometers; display offsets preserved; separate specimens never connected',
                     remote_metadata_bytes=remote.fetched_bytes if remote else 0,
                     full_expression_download_complete=source.exists(),
                     no_cells_sampled_or_filtered=True,
                     model_training=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix('.parquet.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(cache)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    table = frame.groupby(['specimen_id', 'patient_id', 'tissue_region'], observed=True).size().reset_index(name='cells')
    table.to_csv(ROOT / 'metadata/specimens.csv', index=False)
    print(f'Author atlas metadata ready: {cells:,} cells, {audit["patients"]} patients, {audit["specimens"]} specimens', flush=True)
    return cache


COLORS = {
    'Cancer cell': '#E77786', 'Epithelial cell': '#E9ADA2', 'T cell': '#20A5A0',
    'B cell': '#77A94E', 'Plasma cell': '#4D7D38', 'Macrophage': '#8C79B3',
    'Dendritic cell': '#B784D2', 'Neutrophil': '#BF8CA7', 'Fibroblast': '#CA9D40',
    'Endothelial cell': '#83AEC9', 'Smooth muscle cell': '#AAA17E',
    'Schwann cell': '#BC7445', 'Mast cell': '#AE4F80', 'Unassigned': '#D9D9D9',
}


def spatial_cell_maps(point_size=1.5, columns=4, specimens_per_page=12, force=False):
    """Show every independent specimen with consistent author cell-type colors."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import hashlib

    if columns < 1 or specimens_per_page < 1 or point_size <= 0:
        raise ValueError('Point size, columns and specimens per page must be positive')
    cache = prepare_display_table()
    frame = pd.read_parquet(cache)
    region_order = {'core': 0, 'margin': 1, 'normal': 2}
    specimens = frame[['patient_id', 'tissue_region', 'specimen_id']].drop_duplicates().copy()
    for key in specimens:
        specimens[key] = specimens[key].astype(str)
    specimens['region_order'] = specimens.tissue_region.map(region_order).fillna(3)
    specimens = specimens.sort_values(['patient_id', 'region_order', 'specimen_id'])
    entries = specimens.to_dict('records')
    observed = sorted(frame.celltype.unique().astype(str))
    extra = [name for name in observed if name not in COLORS]
    palette = dict(COLORS)
    for i, name in enumerate(extra):
        palette[name] = matplotlib.colors.to_hex(plt.get_cmap('tab20')(i % 20))
    legend_names = [name for name in palette if name in observed]
    settings = dict(point_size=point_size, columns=columns, specimens_per_page=specimens_per_page,
                    cells=len(frame), specimen_ids=[e['specimen_id'] for e in entries],
                    source_cache_bytes=cache.stat().st_size, palette=palette)
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUTPUT / 'atlas_settings.json'
    count = (len(entries) + specimens_per_page - 1) // specimens_per_page
    paths = [OUTPUT / f'cell_types_{i + 1:02d}.png' for i in range(count)]
    if not force and receipt_path.exists() and all(p.exists() for p in paths):
        if json.loads(receipt_path.read_text()).get('hash') == identity:
            return paths
    grouped = frame.groupby('specimen_id', observed=True).indices
    for page in range(count):
        selected = entries[page * specimens_per_page:(page + 1) * specimens_per_page]
        nrows = (len(selected) + columns - 1) // columns
        fig, axes = plt.subplots(nrows, columns, figsize=(columns * 5.1, nrows * 5.0), squeeze=False)
        for ax, item in zip(axes.ravel(), selected):
            cells = frame.iloc[grouped[item['specimen_id']]]
            x, y = cells.x_um.to_numpy().copy(), cells.y_um.to_numpy().copy()
            x -= x.min()
            y -= y.min()
            order = np.random.default_rng(40700).permutation(len(cells))
            color = cells.celltype.astype(str).map(palette).to_numpy()
            ax.scatter(x[order], y[order], c=color[order], s=point_size, linewidths=0,
                       alpha=1, rasterized=True)
            dx, dy = max(np.ptp(x), 1), max(np.ptp(y), 1)
            scale = 1000 if dx > 6000 else 500 if dx > 2000 else 200 if dx > 700 else 100
            bx, by = .05 * dx, 1.025 * dy
            ax.plot([bx, bx + scale], [by, by], color='#333333', lw=2)
            ax.text(bx + scale / 2, by + .025 * dy, f'{scale} µm', ha='center', va='top', fontsize=9)
            ax.set_xlim(-.025 * dx, 1.025 * dx)
            ax.set_ylim(1.11 * dy, -.025 * dy)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"Patient {item['patient_id'].upper()} · {item['tissue_region'].capitalize()}\n"
                         f"{item['specimen_id']} · {len(cells):,} cells", fontsize=12, pad=8)
        for ax in axes.ravel()[len(selected):]:
            ax.axis('off')
        handles = [Line2D([], [], marker='o', color='none', markerfacecolor=palette[name],
                          markeredgewidth=0, markersize=7, label=name) for name in legend_names]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncol=7,
                   frameon=False, fontsize=10, columnspacing=1.1, handletextpad=.35)
        fig.suptitle(f'CRC Xenium · author cell-type annotations\n'
                     f'{page * specimens_per_page + 1}–{page * specimens_per_page + len(selected)} / '
                     f'{len(entries)} specimens · page {page + 1}/{count}', fontsize=18, y=.993)
        fig.text(.5, .003, 'One panel per independent specimen; all author cells shown. '
                 'Core, margin and normal are separately sampled tissue regions.',
                 ha='center', fontsize=10, color='#555555')
        fig.subplots_adjust(left=.015, right=.985, bottom=.095, top=.91, hspace=.22, wspace=.11)
        fig.savefig(paths[page], dpi=170, facecolor='white')
        fig.savefig(paths[page].with_suffix('.pdf'), dpi=170, facecolor='white')
        plt.close(fig)
    receipt_path.write_text(json.dumps(dict(hash=identity, **settings), indent=2) + '\n')
    return paths
