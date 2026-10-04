#!/usr/bin/env python
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def md(text): return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
def code(text): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = [
md("# Xenium Prime 5K Breast — V1006 CCC niches\n\nThe trained AE, P32/Z32 and H32 are reused. Final K is selected from cell-level program usage without retraining."),
code("""from pathlib import Path
import sys, numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path('/home/xueshuailin/CCC_Phe/V1006')
SOURCE = ROOT / 'src'
if str(SOURCE) not in sys.path: sys.path.insert(0, str(SOURCE))
from phenoniche.v1006.pipeline import ensure_results
from phenoniche.v1006.data import load_prepared_input
from phenoniche.v1006.reporting import (training_curves, model_selection_plot, spatial_overview,
    niche_celltype_enrichment, spatial_each_niche, niche_size_plot, representative_h_heatmap)
artifacts = ensure_results()
summary, output, selected_K = artifacts['summary'], artifacts['output'], artifacts['selected_K']
"""),
md("## 1. Data and trained model"),
code("""pd.Series({'cells': summary['n_cells'], 'original CCC features': summary['n_input_features'],
           'retained CCC features': summary['n_retained_features'], 'latent programs': 32,
           'selected final K': selected_K, 'V1002-comparable K': 8}, name='V1006')"""),
code("""training_curves(output); plt.show()
pd.Series({'Stage 1 validation reconstruction': summary['stage1_validation_reconstruction'],
           'Stage 2 validation nonlinear AE': summary['stage2_validation_ae'],
           'Stage 2 validation Z32 H32': summary['stage2_validation_linear']})"""),
md("## 2. Automatic K selection (4–15)\n\nClustering uses cell-level sqrt(P32) plus log CCC activity. Original Q8_direct is retained as the K=8 comparator."),
code("""model_selection_plot(artifacts); plt.show()
columns = ['K','silhouette','adjacent_K_stability','max_niche_cell_fraction',
           'tiny_niche_fraction','mean_assignment_confidence','mean_H_program_separation',
           'selection_score','eligible','selected']
display(artifacts['model_selection'][columns].round(4))
print(f"Selected K={selected_K}: {summary['selection_reason']}")"""),
md("## 3. Selected-K spatial map"),
code("""spatial_overview(artifacts); plt.show()"""),
md("## 4. Niche size and assignment confidence"),
code("""niche_size_plot(artifacts); plt.show()
display(artifacts['counts'])
confidence = artifacts['assignments']['assignment_confidence_relative']
fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.hist(confidence, bins=40, color='#4c78a8')
ax.set(xlabel='Assignment confidence', ylabel='Cells', title=f'Selected K={selected_K} confidence')
plt.tight_layout(); plt.show()
pd.Series(confidence).describe(percentiles=[.1,.25,.5,.75,.9]).to_frame('confidence')"""),
md("## 5. Cell types and individual niche maps\n\nCell type is used only for post-training interpretation."),
code("""_, enrichment_table = niche_celltype_enrichment(artifacts); plt.show()
print('Stars: one-sided Mann–Whitney U; dot size: across-niche entropy.')"""),
code("""spatial_each_niche(artifacts); plt.show()"""),
md("## 6. Representative CCC programs"),
code("""for niche in range(1, selected_K + 1):
    print(f'\\nNiche {niche}')
    display(artifacts['top_ccc'].query('niche == @niche')[['rank','ccc','weight']].head(15).reset_index(drop=True))
selected_dir = output / 'selected_model'
print('Top sender → receiver')
display(pd.read_csv(selected_dir/'top_sender_receiver.csv').groupby('niche', group_keys=False).head(5))
print('Top ligand–receptor')
display(pd.read_csv(selected_dir/'top_lr.csv').groupby('niche', group_keys=False).head(5))"""),
md("## 7. Selected niche × representative CCC heatmap"),
code("""prepared = load_prepared_input(artifacts['config'])
representative_h_heatmap(artifacts['H'], prepared.features); plt.show()"""),
md("## 8. Latent-program contribution to selected niches"),
code("""compact = (artifacts['mapping'].groupby('final_niche_id')
           .agg(latent_programs=('program_id', lambda x: ', '.join(map(str, x))),
                total_exposure=('mean_exposure','sum')).reset_index())
display(compact)"""),
md("## 9. Selected K versus original Q8_direct comparator"),
code("""display(pd.DataFrame({
    'niches': [8, selected_K],
    'largest_niche_fraction': [summary['K8_max_niche_cell_fraction'], summary['selected_max_niche_cell_fraction']],
    'tiny_niche_fraction': [summary['K8_tiny_niche_fraction'], summary['selected_tiny_niche_fraction']],
    'mean_assignment_confidence': [summary['K8_mean_assignment_confidence'], summary['selected_mean_assignment_confidence']]
}, index=['Original Q8_direct', f'Cell-latent selected K={selected_K}']).round(4))"""),
md("## 10. Summary"),
code("""top = artifacts['top_ccc'].query('rank == 1').set_index('niche')['ccc'].to_dict()
print(f'Selected K: {selected_K}; all selected niches used: {len(artifacts["counts"]) == selected_K and (artifacts["counts"].n_cells > 0).all()}')
print(f'Largest selected niche: {summary["selected_max_niche_cell_fraction"]:.2%}')
print(f'Tiny-niche cell fraction: {summary["selected_tiny_niche_fraction"]:.2%}')
print(f'Mean assignment confidence: {summary["selected_mean_assignment_confidence"]:.3f}')
display(pd.Series(top, name='Top representative CCC').rename_axis('Niche').to_frame())""")]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "cccphe", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "XeniumPrime5K_Breast_V1006_niche_walkthrough.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
