"""Author cell-type spatial maps for all25 SIMVI melanoma patients (metadata only)."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem

DATA = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/simvi_melanoma_cosmx')
OUT = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/simvi_melanoma_cosmx/cell_types')
COLORS = {
    'Tumor_a': '#b2182b', 'Tumor_b': '#d6604d', 'Tumor_c': '#ef8a62',
    'Tumor_d': '#f4a582', 'Tumor_e': '#ce4b78', 'Tumor_f': '#e9a6ba',
    'T CD8 memory': '#2285b8', 'T CD8 naive': '#76b7de', 'Treg': '#424b9b',
    'B-cell': '#599c39', 'plasmablast': '#99c56a', 'NK': '#18a59d',
    'macrophage': '#b68629', 'pDC': '#8c59a4', 'mast': '#75503c',
    'fibroblast': '#a2a2ad', 'endothelial': '#4cacc3',
}


def main():
    # Figure contract: show observed tissue organization for every patient, not
    # treatment effects. Preserve all author cell labels and all cell positions.
    # Independently crop each panel; preserve aspect ratio, no spatial smoothing.
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(DATA / 'Melanoma_5612.h5ad', 'r') as f:
        frame = pd.DataFrame({key: read_elem(f['obs'][key]) for key in ('CPID_str', 'SPID', 'celltype', 'fov')})
        xy = np.asarray(read_elem(f['obsm']['spatial']))
    if len(frame) != 56761 or xy.shape != (len(frame), 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid author metadata or spatial coordinates')
    frame['patient'] = frame.CPID_str.astype(str)
    frame['type'] = frame.celltype.astype(str)
    if set(frame.type) != set(COLORS):
        raise ValueError('Author cell labels changed')
    if (frame.groupby('patient').SPID.nunique() != 1).any():
        raise ValueError('Multiple specimens per patient: plot separate specimens')
    patients = sorted(frame.patient.unique(), key=int)
    panels = [(patient, fov) for patient in patients
              for fov in sorted(frame.loc[frame.patient == patient, 'fov'].unique(), key=int)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42})
    handles = [Line2D([], [], marker='o', linestyle='', markersize=7, color=color, label=kind)
               for kind, color in COLORS.items()]
    receipts = []
    for page, selected in enumerate((panels[:13], panels[13:]), 1):
        rows, columns = (len(selected) + 4) // 5, 5
        fig, axes = plt.subplots(rows, columns, figsize=(20, rows * 3.9 + 2.1), squeeze=False)
        fig.subplots_adjust(left=.015, right=.985, top=.915, bottom=1.85 / (rows * 3.9 + 2.1), hspace=.22, wspace=.08)
        fig.suptitle('SIMVI melanoma | author cell types', fontsize=22, fontweight='bold', y=.985)
        for ax, (patient, fov) in zip(axes.flat, selected):
            idx = np.flatnonzero((frame.patient.to_numpy() == patient) & (frame.fov.to_numpy() == fov))
            points = xy[idx]
            center = (points.min(0) + points.max(0)) / 2
            points = points - center
            span = max(float(np.ptp(points, axis=0).max()), 1.) * 1.08
            labels = frame.type.iloc[idx].to_numpy()
            # One uniformly shuffled draw order; immune cells are not overlaid
            # preferentially over tumor cells.
            order = np.random.default_rng(40700).permutation(len(idx))
            ax.scatter(points[order, 0], points[order, 1], s=9.5,
                       c=[COLORS[label] for label in labels[order]], linewidths=0, alpha=.95, rasterized=True)
            ax.set(xlim=(-span / 2, span / 2), ylim=(span / 2, -span / 2), aspect='equal')
            extra = f' / FOV{fov}' if frame.loc[frame.patient == patient, 'fov'].nunique() > 1 else ''
            ax.set_title(f'Patient {patient}{extra}  |  {len(idx):,} cells', fontsize=13, pad=7)
            ax.axis('off')
        for ax in list(axes.flat)[len(selected):]:
            ax.axis('off')
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .025), ncol=6,
                   frameon=False, fontsize=12, columnspacing=1.8, handletextpad=.45, labelspacing=.8)
        path = OUT / f'author_cell_types_page{page}.png'
        fig.savefig(path, dpi=170, facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), dpi=170, facecolor='white')
        plt.close(fig)
        receipts.append({'image': str(path), 'patient_fov_panels': [[patient, int(fov)] for patient, fov in selected]})
        print(path, flush=True)
    audit = {'source': str(DATA / 'Melanoma_5612.h5ad'), 'cells': len(frame), 'patients': len(patients),
             'author_labels_retained': True, 'type_colors': COLORS, 'sampling': 'none',
             'coordinates': 'author obsm/spatial; each patient/FOV plotted separately; panels independently cropped; aspect preserved; y inverted for image display',
             'point_size': 9.5, 'clinical_outcomes_used': False, 'model_fitting': False, 'pages': receipts}
    (OUT / 'figure_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    frame.groupby(['patient', 'type']).size().unstack(fill_value=0).to_csv(OUT / 'patient_cell_type_counts.csv')


if __name__ == '__main__':
    main()
