"""Replot the public SPATCH COAD Xenium cell map; visualization only."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import requests

WORK = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace')
DATA = WORK / 'data/spatch_coad_preview'
OUTPUT = WORK / 'outputs/spatch_coad_preview'
BASE = 'https://spatch.pku-genomics.org/spatch'
PARAMS = dict(datasetId=3, tableName='xenium_coad')


def source(name):
    path = DATA / f'{name}.json'
    if not path.exists():
        reply = requests.post(BASE + '/visualization/' + name, data=PARAMS, timeout=90)
        reply.raise_for_status()
        value = reply.json()
        if value.get('status') != 600200:
            raise RuntimeError(f'Public visualization endpoint failed: {name}')
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(value))
        temp.replace(path)
    return json.loads(path.read_text())['data']


def run():
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Figure contract: show the complete author-displayed cell map for dataset
    # review. No model, niche labels, selected region, expression or smoothing.
    contract = dict(
        conclusion='Original spatial organization of all author-displayed COAD Xenium cells',
        evidence='Public portal coordinate arrays and original cell annotations',
        archetype='quantitative grid, one spatial panel', backend='Python/matplotlib',
        exports=['PNG', 'PDF', 'source CSV', 'provenance JSON'],
        integrity='No downsampling, coordinate transformation, filtering or spatial smoothing; Y displayed downward as in author portal',
        statistics='Descriptive cell counts only; no hypothesis tests',
        limitation='Portal Epithelial class is not separated into malignant and normal epithelium; confidence and RNA quality not independently audited',
        source='https://spatch.pku-genomics.org/#/dataset/xenium',
        paper='https://doi.org/10.1038/s41467-025-64292-3',
        dataset=PARAMS, assay='Xenium 5K', expected_cells=365395,
        row_identity='Portal array index, NOT a physical cell barcode; do not align expression using this index without raw-object audit',
        rendering_order='Fixed random permutation of all points to avoid annotation-specific overplotting',
    )
    (OUTPUT / 'figure_contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    names = ['showThreeX', 'showThreeY', 'showThreeCeId', 'getCellType']
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = dict(zip(names, pool.map(source, names)))
    x = np.asarray(values['showThreeX']['vis']['x'], np.float64)
    y = np.asarray(values['showThreeY']['vis']['y'], np.float64)
    ids = np.asarray(values['showThreeCeId']['vis']['ceId'], int)
    cells = values['getCellType']['cells']
    categories = {int(c['id']): c['name'] for c in cells}
    if len(x) != 365395 or len(y) != len(x) or len(ids) != len(x):
        raise ValueError('Portal arrays do not match the declared whole-section cell count')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite coordinates')
    for cell in cells:
        if np.count_nonzero(ids == int(cell['id'])) != int(cell['count']):
            raise ValueError('Portal point labels and category counts disagree')
    labels = np.asarray([categories[i] for i in ids])
    frame = pd.DataFrame(dict(portal_row_index=np.arange(len(x)), x_native=x,
                              y_native=y, cell_type=labels))
    frame.to_csv(DATA / 'portal_cells.csv.gz', index=False)
    counts = pd.Series(labels).value_counts()
    pd.DataFrame(dict(cells=counts, percent=100*counts/len(x))).to_csv(OUTPUT / 'cell_type_counts.csv')
    colors = dict(Epithelial='#DE737B', Fibroblast='#C39B49', SMC='#B6AA80',
        Endothelial='#8AADC7', CD4T='#31A59C', CD8T='#257DB2', Tprolif='#205E74',
        B='#78A94D', Plasma='#4F7E37', Macrophage='#8E79AE', Monocyte='#B6A1CB',
        Neutrophil='#C591A9', Mast='#B65F91', cDC2='#DD9361', mregDC='#AE6037', pDC='#E6BB88')
    if set(labels) - colors.keys():
        raise ValueError('An unexpected annotation requires an explicit legend entry')
    order = np.random.default_rng(40700).permutation(len(x))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(13.5, 9.3), layout='constrained')
    ax.scatter(x[order], y[order], c=[colors[v] for v in labels[order]],
               s=.65, marker='o', linewidths=0, edgecolors='none', rasterized=True)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_axis_off()
    ax.set_title('Colon adenocarcinoma · Xenium 5K\n365,395 cells · author cell-type annotations',
                 fontsize=15, pad=18)
    legend_order = ['Epithelial', 'Fibroblast', 'SMC', 'Endothelial', 'CD4T', 'CD8T', 'Tprolif',
                    'B', 'Plasma', 'Macrophage', 'Monocyte', 'Neutrophil', 'Mast', 'cDC2', 'mregDC', 'pDC']
    def percent(k):
        value = 100*counts[k]/len(x)
        return '<0.1%' if value < .1 else f'{value:.1f}%'
    handles = [Line2D([], [], marker='o', linestyle='', color=colors[k], markersize=5.5,
                      label=f'{k}  {percent(k)}') for k in legend_order]
    ax.legend(handles=handles, title='Author cell types', frameon=False,
               loc='center left', bbox_to_anchor=(1.01, .5), labelspacing=.85, fontsize=10)
    # These are Xenium centroid coordinates from the native cell-resolution
    # portal table. Preserve the same Cartesian proportions and image Y direction.
    # Do not add a physical scale bar until raw coordinate units are independently
    # confirmed from the downloadable object; this preview is for morphology review.
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT / f'COAD_Xenium5K_cell_types.{ext}', dpi=250, bbox_inches='tight')
    plt.close(fig)
    contract['observed_cells'] = len(x)
    contract['observed_types'] = len(categories)
    contract['coordinate_ranges_native'] = dict(x=[float(x.min()), float(x.max())],
                                                y=[float(y.min()), float(y.max())])
    contract['source_sha256'] = {n: hashlib.sha256((DATA/f'{n}.json').read_bytes()).hexdigest() for n in names}
    (OUTPUT / 'provenance.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(counts.to_string())
    print(OUTPUT / 'COAD_Xenium5K_cell_types.png')


if __name__ == '__main__':
    run()
