#!/usr/bin/env python
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
md("# Xenium Prime 5K Breast — V1006 representation-first CCC niches\n\nOnly directed CCC enters training. Cell type is used only for interpretation."),
code("""from pathlib import Path
import sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path('/home/xueshuailin/CCC_Phe/V1006')
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path: sys.path.insert(0, str(SOURCE))
from phenoniche.v1006.pipeline import ensure_results
from phenoniche.v1006.data import load_prepared_input
from phenoniche.v1006.reporting import (training_curves, spatial_overview, niche_celltype_enrichment,
    spatial_each_niche, niche_size_plot, representative_h_heatmap)
artifacts = ensure_results()
summary, output = artifacts['summary'], artifacts['output']
"""),
md("## 1. Prime 5K data summary"),
code("""pd.Series({'cells': summary['n_cells'], 'original CCC features': summary['n_input_features'],
           'retained CCC features': summary['n_retained_features'], 'latent dimension': 32,
           'final niches': 8}, name='V1006')"""),
md("## 2. Model summary"),
code("""print('CCC F → 256 → 64 → ReLU Z32 → nonlinear AE decoder')
print('Z32 → fixed train-derived PCA whitening → 8 learnable cosine prototypes → soft assignment Q8')
print('Niche CCC profiles are soft-weighted means of the observed directed CCC vectors.')"""),
md("## 3. Training loss"),
code("""training_curves(output); plt.show()
pd.Series({'Stage 1 best epoch': summary['stage1_best_epoch'],
           'Stage 1 validation reconstruction': summary['stage1_validation_reconstruction'],
           'Stage 2 selected epoch': summary['stage2_best_epoch'],
           'Stage 2 validation AE': summary['stage2_validation_reconstruction'],
           'Stage 2 validation prototype loss': summary['stage2_validation_cluster_loss']})"""),
md("## 4. Final 8-niche spatial map\n\nV1006 has one CCC-derived final assignment. It therefore shows cell types and V1006 niches, without inventing Composition-only or shared-W panels."),
code("""spatial_overview(artifacts); plt.show()"""),
md("## 5. Niche size and proportion"),
code("""niche_size_plot(artifacts); plt.show()
display(artifacts['counts'])
confidence = artifacts['assignments']['assignment_confidence_relative']
fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.hist(confidence, bins=40, color='#4c78a8')
ax.set(xlabel='Relative assignment confidence', ylabel='Cells', title='Final niche assignment confidence')
plt.tight_layout(); plt.show()
pd.Series(confidence).describe(percentiles=[.1,.25,.5,.75,.9]).to_frame('confidence')"""),
md("## 6. Cell-type / niche summary\n\nCell type is used here only after training."),
code("""_, enrichment_table = niche_celltype_enrichment(artifacts); plt.show()
print('Stars: one-sided Mann–Whitney U, * p≤0.05, ** p≤0.001, *** p≤0.0001; dot size: across-niche entropy.')"""),
code("""spatial_each_niche(artifacts); plt.show()"""),
md("## 7. Representative CCC"),
code("""for niche in range(1, 9):
    print(f'\\nNiche {niche}')
    display(artifacts['top_ccc'].query('niche == @niche')[['rank','ccc','weight']].head(15).reset_index(drop=True))
print('Top sender → receiver')
display(pd.read_csv(output/'top_sender_receiver.csv').groupby('niche', group_keys=False).head(5))
print('Top ligand–receptor')
display(pd.read_csv(output/'top_lr.csv').groupby('niche', group_keys=False).head(5))"""),
md("## 8. Empirical niche CCC-profile heatmap"),
code("""prepared = load_prepared_input(artifacts['config'])
representative_h_heatmap(artifacts['H8'], prepared.features); plt.show()"""),
md("## 9. Prototype diagnostics"),
code("""display(artifacts['counts'])
prototype_similarity = pd.read_csv(output/'prototype_cosine_similarity.csv', index_col=0)
fig, ax = plt.subplots(figsize=(5,4));
import seaborn as sns
sns.heatmap(prototype_similarity, vmin=-1, vmax=1, center=0, cmap='RdBu_r', square=True, ax=ax)
ax.set(title='Prototype cosine similarity', xlabel='Prototype', ylabel='Prototype'); plt.tight_layout(); plt.show()"""),
md("## 10. Final summary"),
code("""top = artifacts['top_ccc'].query('rank == 1').set_index('niche')['ccc'].to_dict()
print(f\"All 8 niches used: {not summary['niche_collapse']}\")
print(f\"Severe niche imbalance: {summary['severe_imbalance']}\")
print(f\"Largest niche fraction: {max(summary['niche_cell_fractions']):.3%}\")
print(f\"Mean assignment confidence: {summary['mean_assignment_confidence']:.3f}\")
print(f\"Maximum off-diagonal prototype cosine: {summary['prototype_max_offdiagonal_cosine']:.3f}\")
display(pd.Series(top, name='Top representative CCC').rename_axis('Niche').to_frame())""")]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "cccphe", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "XeniumPrime5K_Breast_V1006_niche_walkthrough.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
