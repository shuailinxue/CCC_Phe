"""Select one TNBC tissue sample per patient without clinical endpoint access.

H&E is assessed at native resolution in spatially spread, spot-covered fields.
Marker activity is a coverage proxy, never a measured cell-type proportion.
All weights are fixed before inspecting patient-level clinical associations.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from skimage.color import rgb2hed

from phenoniche.phenomap_adapter.genes import MARKERS, gene_parts

Image.MAX_IMAGE_PIXELS = None
POLICY = {
    'version': 1, 'seed': 52173, 'native_fields': 4, 'field_pixels': 512,
    'minimum_spots': 100, 'minimum_UMI': 500, 'minimum_detected_genes': 200,
    'minimum_good_spot_fraction': 0.5,
    'weights': {'he': .40, 'spot': .25, 'structure': .20, 'representative': .15},
    'he_near_best_tolerance': .15,
    'clinical_endpoints_used': False,
    'interpretation': 'Image/spot QC and marker-activity coverage; no validated histological diagnosis or deconvolved proportions',
}


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def patient_id(sample):
    match = re.fullmatch(r'TNBC(\d+)_CN\d+_[A-Z]\d+', sample)
    if match is None:
        raise ValueError(f'Unexpected sample identifier: {sample}')
    return int(match.group(1))


def spread_indices(xy, n=4):
    """Deterministic farthest-point sampling, seeded at the spatial medoid."""
    xy = np.asarray(xy, float)
    if not len(xy):
        raise ValueError('No spot coordinates')
    first = int(np.argmin(((xy - np.median(xy, axis=0))**2).sum(1)))
    ids = [first]
    distance = ((xy-xy[first])**2).sum(1)
    for _ in range(min(n, len(xy))-1):
        distance[ids] = -1
        index = int(np.argmax(distance))
        ids.append(index)
        distance = np.minimum(distance, ((xy-xy[index])**2).sum(1))
    return ids


def field_metrics(rgb):
    rgb = np.asarray(rgb, np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hed = rgb2hed(rgb).astype(np.float32)
    h = np.maximum(hed[..., 0], 0)
    tissue = (rgb.mean(2) < 238) & (rgb.min(2) < 220)
    frac = float(tissue.mean())
    contrast = float(np.percentile(h[tissue], 90)-np.percentile(h[tissue], 10)) if tissue.any() else 0.
    # Normalize focus for stain amplitude, restricting to tissue to avoid blank edges.
    lap = cv2.Laplacian(h, cv2.CV_32F)
    focus = float(np.mean(lap[tissue]**2)/(contrast**2+1e-6)) if tissue.any() else 0.
    saturation = float(((rgb < 8).all(2)).mean())
    return {'tissue_fraction': frac, 'nuclear_contrast': contrast,
            'focus': focus, 'black_fraction': saturation,
            'usable': bool(frac >= .15 and contrast >= .008 and saturation < .2)}


def native_crop(image_path, x, y, size, djpeg):
    with Image.open(image_path) as image:
        w, h = image.size
    x0 = int(np.clip(round(x-size/2), 0, max(0, w-size)))//16*16
    y0 = int(np.clip(round(y-size/2), 0, max(0, h-size)))//16*16
    result = subprocess.run([str(djpeg), '-crop', f'{size}x{size}+{x0}+{y0}', str(image_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    crop = np.asarray(Image.open(io.BytesIO(result.stdout)).convert('RGB'))
    if crop.shape[:2] != (size, size):
        raise ValueError(f'Unexpected JPEG crop shape: {crop.shape}')
    return crop, [x0, y0, size, size]


def audit_one(source, out, genes, lr_pairs, djpeg):
    """Read one original spot matrix and four native H&E fields; resumable."""
    cv2.setNumThreads(1)
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = [source/'cnts.csv', source/'locs.csv', source/'tissue_image.jpg']
    signature = {p.name: {'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime_ns}
                 for p in files if p.exists()}
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    signature.update({'policy': POLICY, 'code': code_hash})
    if (out/'metrics.json').exists():
        data = json.loads((out/'metrics.json').read_text())
        if data['signature'] == signature:
            return data['metrics']
    base = {'sample': source.name, 'patient_id': patient_id(source.name)}
    missing = [p.name for p in files if not p.exists()]
    if missing:
        base.update({'input_ready': False, 'error': 'Missing '+','.join(missing)})
        atomic_json(out/'metrics.json', {'signature': signature, 'metrics': base})
        return base
    loc = pd.read_csv(source/'locs.csv', index_col=0)
    with (source/'cnts.csv').open() as stream:
        columns = next(csv.reader(stream))[1:]
    # Read raw counts, not reconstructed expression; no clinical metadata is opened.
    count = pd.read_csv(source/'cnts.csv', index_col=0,
                        dtype={g: np.float32 for g in columns})
    count.index = count.index.astype(str)
    loc.index = loc.index.astype(str)
    if count.index.has_duplicates or loc.index.has_duplicates:
        raise ValueError('Duplicate spot identifiers')
    if not loc.index.isin(count.index).all():
        raise ValueError('Spatial spot IDs absent from count matrix')
    count = count.reindex(loc.index)
    values = count.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid raw counts')
    totals = values.sum(1)
    detected = (values > 0).sum(1)
    good = (totals >= POLICY['minimum_UMI']) & (detected >= POLICY['minimum_detected_genes'])
    lookup = {g.upper(): i for i, g in enumerate(count.columns)}
    ids = np.array([lookup[g] for g in genes])
    panel = values[:, ids]
    normalized = panel / np.maximum(totals[:, None], 1)*1e4
    logs = np.log1p(normalized)
    activities = np.column_stack([logs[:, [genes.index(g) for g in markers if g in genes]].mean(1)
                                  for markers in MARKERS.values()])
    signature_expression = np.log1p(panel.mean(0)/max(float(totals.mean()), 1)*1e4)
    measured = set(g for g, fraction in zip(genes, (panel > 0).mean(0)) if fraction >= .10)
    complete = sum(all(g in measured for g in parts) for parts in lr_pairs)
    xy = loc[['x', 'y']].to_numpy(float)
    pool = np.flatnonzero(good) if good.sum() >= 4 else np.arange(len(xy))
    # Sample center and spatial extremes inside actual spot support.
    selected = pool[spread_indices(xy[pool], POLICY['native_fields'])]
    fields, bounds, stats = [], [], []
    for index in selected:
        crop, box = native_crop(source/'tissue_image.jpg', *xy[index], POLICY['field_pixels'], djpeg)
        fields.append(crop); bounds.append(box); stats.append(field_metrics(crop))
    np.savez_compressed(out/'fields.npz', rgb=np.stack(fields), boxes=bounds)
    np.savez_compressed(out/'spot_summary.npz', expression=signature_expression.astype(np.float32),
                        activities=activities.astype(np.float32), xy=xy.astype(np.float32),
                        good=good, genes=np.asarray(genes), lineage_names=np.asarray(list(MARKERS)))
    with Image.open(source/'tissue_image.jpg') as image:
        image.draft('RGB', (1024, 1024)); image.thumbnail((700, 700))
        image.convert('RGB').save(out/'thumbnail.jpg', quality=88)
    base.update({'input_ready': True, 'error': '', 'spots': len(loc), 'good_spots': int(good.sum()),
                 'good_spot_fraction': float(good.mean()), 'median_UMI': float(np.median(totals)),
                 'median_detected_genes': float(np.median(detected)),
                 'LR_pairs_detected_at_10pct': int(complete),
                 'LR_gene_detection_fraction': float((panel > 0).mean()),
                 'he_focus': float(np.median([r['focus'] for r in stats])),
                 'he_nuclear_contrast': float(np.median([r['nuclear_contrast'] for r in stats])),
                 'he_usable_fraction': float(np.mean([r['usable'] for r in stats])),
                 'he_black_fraction': float(np.mean([r['black_fraction'] for r in stats])),
                 'field_stats': stats})
    atomic_json(out/'metrics.json', {'signature': signature, 'metrics': base})
    return base


def segmentation_fields(mask, rgb):
    """Plausible nuclei and contrast, not a validated segmentation accuracy score."""
    count = np.bincount(mask.ravel())
    valid_ids = np.flatnonzero((count >= 20) & (count <= 5000))
    valid_ids = valid_ids[valid_ids > 0]
    foreground = np.isin(mask, valid_ids)
    h = np.maximum(rgb2hed(rgb)[..., 0], 0)
    ring = binary_dilation(foreground, iterations=3) & ~foreground
    # Ring contrast rewards visible nuclear contrast rather than dark/noisy slides.
    contrast = max(0., float(np.median(h[foreground])-np.median(h[ring]))) if foreground.any() and ring.any() else 0.
    return {'plausible_nuclei': int(len(valid_ids)), 'foreground_fraction': float(foreground.mean()),
            'nuclear_ring_contrast': contrast,
            'supported': bool(len(valid_ids) >= 3 and .002 < foreground.mean() < .8 and contrast > .002)}


def rank_samples(frame, cache_root):
    """Freeze quality-aware patient choices; never inspect survival or phenotype."""
    frame = frame.copy()
    ready = frame.input_ready.fillna(False).astype(bool)
    valid = frame.loc[ready].copy()
    activities = []
    for row in valid.itertuples():
        with np.load(Path(cache_root)/row.sample/'spot_summary.npz') as data:
            activities.append(data['activities'])
    population = np.concatenate(activities)
    mean, sd = population.mean(0), np.maximum(population.std(0), .15)
    distributions, signatures = {}, {}
    from scipy.special import softmax
    for row, activity in zip(valid.itertuples(), activities):
        z = (activity-mean)/sd
        probability = softmax(z*1.5, axis=1)
        profile = probability.mean(0)
        distributions[row.sample] = profile
        with np.load(Path(cache_root)/row.sample/'spot_summary.npz') as data:
            signatures[row.sample] = data['expression']
            xy, good = data['xy'], data['good']
        dominant = z.argmax(1)
        neighbor = cKDTree(xy).query(xy, k=min(7, len(xy)))[1][:, 1:]
        if neighbor.ndim != 2 or not neighbor.size:
            raise ValueError('Too few spots for spatial coverage')
        different = dominant[:, None] != dominant[neighbor]
        # Count coherent heterotypic interfaces: reject isolated random mixtures.
        within = (dominant[:, None] == dominant[neighbor]).mean(1)
        interface = float((different * (within[:, None] >= .33) * (within[neighbor] >= .33)).mean())
        entropy = float(-(profile*np.log(profile+1e-10)).sum()/np.log(len(profile)))
        coherent_types = sum(np.sum((dominant == i) & (within >= .5) & good) >= max(5, .02*len(xy)) for i in range(6))
        ix = valid.index[valid['sample'] == row.sample][0]
        valid.loc[ix, ['marker_activity_entropy','coherent_lineages','coherent_interface_fraction']] = [entropy, coherent_types, interface]
    expression = np.stack(list(signatures.values()))
    expr_sd = np.maximum(expression.std(0), .15)
    for patient, group in valid.groupby('patient_id'):
        samples = group['sample'].tolist()
        for sample in samples:
            # Patient profile weights regions by good-spot support, avoiding extreme cores.
            weights = np.array([max(float(group.loc[group['sample'] == s, 'good_spots'].iloc[0]), 1) for s in samples])
            profile = np.average([distributions[s] for s in samples], axis=0, weights=weights)
            expression_profile = np.average([signatures[s] for s in samples], axis=0, weights=weights)
            distance = float(jensenshannon(distributions[sample], profile) + .15*np.sqrt(np.mean(((signatures[sample]-expression_profile)/expr_sd)**2)))
            valid.loc[valid['sample'] == sample, 'patient_profile_distance'] = distance
    rank = lambda c: valid[c].rank(pct=True, method='average')
    valid['he_score'] = .30*rank('he_focus') + .25*rank('he_nuclear_contrast') + .15*rank('he_usable_fraction') + .15*rank('segmentation_supported_fraction') + .15*rank('segmentation_ring_contrast')
    valid['spot_score'] = .25*rank('good_spots') + .20*rank('good_spot_fraction') + .20*rank('median_UMI') + .15*rank('median_detected_genes') + .20*rank('LR_pairs_detected_at_10pct')
    valid['structure_score'] = .35*rank('coherent_lineages') + .35*rank('coherent_interface_fraction') + .30*rank('marker_activity_entropy')
    valid['representative_score'] = 1-rank('patient_profile_distance')
    valid['selection_score'] = sum(POLICY['weights'][k]*valid[v] for k,v in [('he','he_score'),('spot','spot_score'),('structure','structure_score'),('representative','representative_score')])
    valid['quality_pass'] = (valid.good_spots >= POLICY['minimum_spots']) & (valid.good_spot_fraction >= .5) & (valid.he_usable_fraction >= .5) & (valid.segmentation_supported_fraction >= .5)
    valid['selected'] = False
    valid['selection_note'] = ''
    for patient, group in valid.groupby('patient_id'):
        eligible = group[group.quality_pass]
        note = 'QC-pass; H&E near-best; maximum fixed composite score'
        if eligible.empty:
            eligible = group
            note = 'Review required: all available patient samples fail at least one absolute QC threshold; best available retained'
        else:
            eligible = eligible[eligible.he_score >= eligible.he_score.max()-POLICY['he_near_best_tolerance']]
        winner = eligible.sort_values(['selection_score','sample'], ascending=[False, True]).index[0]
        valid.loc[winner, ['selected','selection_note']] = [True, note]
    invalid = frame.loc[~ready].copy()
    invalid['selected'], invalid['selection_note'] = False, 'Unavailable input'
    return pd.concat([valid,invalid],ignore_index=True).sort_values(['patient_id','sample'])


def contact_pages(frame, cache_root, out):
    """QC contact sheets, not manuscript figures; retain native-field evidence."""
    out = Path(out); out.mkdir(exist_ok=True)
    patients = sorted(frame.patient_id.unique())
    for page in range(0, len(patients), 8):
        chosen = patients[page:page+8]
        canvas = Image.new('RGB', (1800, len(chosen)*330), 'white')
        draw = ImageDraw.Draw(canvas)
        for row_index, patient in enumerate(chosen):
            for col, row in enumerate(frame[frame.patient_id == patient].itertuples()):
                x, y = col*600, row_index*330
                selected = bool(row.selected)
                if selected:
                    draw.rectangle((x+1,y+1,x+598,y+328),outline='#218739',width=5)
                text = f'{row.sample}'+(' SELECTED' if selected else '')
                if row.input_ready:
                    text += f' | HE {row.he_score:.2f} | spot {row.spot_score:.2f}'
                    thumb = Image.open(Path(cache_root)/row.sample/'thumbnail.jpg'); thumb.thumbnail((280,260))
                    canvas.paste(thumb,(x+8,y+40))
                    with np.load(Path(cache_root)/row.sample/'fields.npz') as data:
                        for k,crop in enumerate(data['rgb']):
                            image=Image.fromarray(crop); image.thumbnail((140,140))
                            canvas.paste(image,(x+300+(k%2)*145,y+35+(k//2)*145))
                draw.text((x+10,y+12),text,fill='black')
        canvas.save(out/f'patient_QC_{page//8+1:02d}.jpg',quality=92)
