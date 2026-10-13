#!/usr/bin/env python
"""Export native CCC predictions, metrics, and execute the read-only result notebook."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import torch
import nbformat
from nbclient import NotebookClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phenoniche.storage import STORAGE_ROOT

out=STORAGE_ROOT/'outputs/microenvironment_gnn'
prep=np.load(out/'cache/preprocessing.npz');z=np.load(out/'Z.npy',mmap_mode='r');h=np.load(out/'H.npy')
source=np.load(STORAGE_ROOT/'data/ccc/cache/ccc.npy',mmap_mode='r')
features=pd.read_csv(out/'cache/features.csv');mask=prep['feature_mask'];scale=prep['feature_scale']
pred=np.lib.format.open_memmap(out/'CCC_reconstruction.incomplete.npy',mode='w+',dtype=np.float32,shape=(len(z),len(h.T)))
stats={i:{'n':0,'sum':np.zeros(len(h.T)),'square':np.zeros(len(h.T)),'sse':np.zeros(len(h.T)),'huber':0.} for i in range(3)}
for start in range(0,len(z),8192):
    stop=min(start+8192,len(z));pred[start:stop]=np.asarray(z[start:stop])@h
    actual=np.asarray(source[start:stop])[:,prep['columns']];estimate=np.asarray(pred[start:stop])
    for fold,record in stats.items():
        keep=prep['folds'][start:stop]==fold;a=actual[keep].astype(np.float64);p=estimate[keep].astype(np.float64)
        error=np.abs((a[:,mask]-p[:,mask])/scale[mask]);loss=np.where(error<=1,.5*error**2,error-.5)
        record['n']+=len(a);record['sum']+=a.sum(0);record['square']+=(a*a).sum(0);record['sse']+=((a-p)**2).sum(0);record['huber']+=loss.sum()
pred.flush();del pred;(out/'CCC_reconstruction.incomplete.npy').replace(out/'CCC_reconstruction.npy')
metrics={}
for fold,name in enumerate(('train','validation','test')):
    r=stats[fold];sst=r['square']-r['sum']**2/max(r['n'],1);r2=np.divide(r['sse'],sst,out=np.full_like(sst,np.nan),where=sst>1e-12);r2=1-r2
    features[name+'_R2']=r2
    metrics[name]={'n':r['n'],'scaled_huber':r['huber']/max(r['n']*mask.sum(),1),'native_RMSE':float(np.sqrt(r['sse'][mask].sum()/max(r['n']*mask.sum(),1))),'median_feature_R2':float(np.nanmedian(r2[mask]))}
features['effective_mask']=mask;features.to_csv(out/'reconstruction_feature_metrics.csv',index=False)
(out/'reconstruction_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False);nh=np.load(out/'selected_model/H.npy')
for branch,alignment in payload['alignments'].items():
    prior=alignment.prior;load=nh@prior.mapping;records=[]
    for i in range(len(nh)):
        for j in np.argsort(-np.abs(load[i])):
            if load[i,j]==0:continue
            records.append({'niche':i+1,**prior.modules.iloc[j].to_dict(),'prior_loading':float(load[i,j]),'used_in_bio_loss':bool(alignment.valid[j]),'training_residual_slope':float(alignment.slope[j])})
    pd.DataFrame(records).to_csv(out/('niche_'+branch+'.csv'),index=False)

nb=nbformat.v4.new_notebook();cells=[]
def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
def code(s):cells.append(nbformat.v4.new_code_cell(s))
md('# Xenium Prime 5K Breast — GNN communication niches\n\nExecuted real-data analysis of one breast tissue. Each of the 699,110 cell-centered, 30-cell microenvironments has one nonnegative 32-program embedding. This notebook only reads saved results; opening or rerunning it does not train a model. This independent downstream analysis does not enter the clinical CCC/Cox mainline.')
code("%matplotlib inline\nfrom pathlib import Path\nimport sys, json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\nsys.path.insert(0, '/home/xueshuailin/CCC_Phe/src')\nfrom phenoniche.gnn.reporting import GNNReport\nOUTPUT = Path("+repr(str(out))+")\nreport = GNNReport(OUTPUT)\npd.set_option('display.max_colwidth', 150)\npd.set_option('display.max_rows', 200)\ndef show(fig):\n    display(fig)\n    plt.close(fig)")
md('## 1. Data scale, fixed CCC features, and model configuration\n\nBroad cell labels are provisional, replayed from outcome-blind expression clusters and fixed markers. Unassigned cells are retained. All retained directed CCC definitions and feature order are unchanged. Spatial blocks are 400 µm; boundary centers are purged so physical cells never cross training/validation/test folds. Purged centers are mapped only after training. Biological scoring targets are excluded from encoder genes and its library-size input.')
code("display(report.overview())\ndisplay(pd.Series(report.config).to_frame('Configuration'))\ndisplay(report.features.head(12))\nfor branch in ('upstream','downstream'):\n    audit=report.audit[branch]\n    display(pd.Series({k:v for k,v in audit.items() if k not in ('independent_score_genes','skipped_entities','excluded_LR_score_genes','encoder_genes')}).to_frame(branch))")
md('## 2. Training and CCC reconstruction\n\n$Z=\\mathrm{softplus}(a)\\,\\mathrm{softmax}(p)$; $\\hat C=ZH$, with $H\\geq0$ and unit row sums. The objective is masked, feature-scaled CCC Huber error + 0.1 × (upstream + downstream residual alignment) + 0.005 × dictionary/program regularization. Biological alignment is calibrated on training cells after removing cell-type, local-composition, and expression-state baselines. Conflicting auxiliary gradients are projected against the CCC gradient. Biological targets are regulon/response proxies, not causal readouts.')
code("show(report.training_curves())\nshow(report.reconstruction())\ndisplay(pd.DataFrame(report.reconstruction_metrics).T)\ndisplay(pd.Series(json.loads((OUTPUT/'heldout_metrics.json').read_text())).to_frame('Held-out metrics'))")
md('## 3. Usage and directed CCC interpretation of all 32 programs\n\nH entries trace programs to specific sender → receiver → ligand–receptor features. Absolute Z retains intensity; relative mixture preserves composition. Prior-linked regulators and targets are hypotheses supported by the supplied databases; NicheNet response modules do not establish receptor-specific causal effects.')
code("fig,usage=report.program_usage();show(fig);display(usage)\nshow(report.dictionary_heatmap())\ndisplay(pd.read_csv(OUTPUT/'programs/top_ccc.csv').groupby('program').head(5))\nfor branch in ('upstream','downstream'):\n    display(pd.read_csv(OUTPUT/'programs'/f'{branch}.csv').groupby('program').head(3))")
md('## 4. Automatic niche selection, K = 4–15\n\nClustering uses communication mixture and a weighted log-activity component of Z. K is selected on validation blocks using silhouette, clustering-seed stability, and niche balance; test blocks are not used for K selection. Stability here concerns clustering seeds, not independent GNN training seeds.')
code("show(report.k_selection())\ndisplay(report.selection)")
md('## 5. Cell types and communication niches in tissue space\n\nAll cell centers are plotted in original micrometers; no spatial region or phenotype labels are supplied to the model.')
code('show(report.spatial_overview())')
md('## 6. Individual niches, cell-type composition, and enrichment\n\nEnrichment is descriptive log₂ fold enrichment against this tissue, without independent-cell p-values. Both center-cell labels and the original physical-neighborhood composition are displayed.')
code('show(report.each_niche())\nfig,fraction,enrichment=report.composition();show(fig)\ndisplay(fraction)\ndisplay(enrichment)\nshow(report.local_composition())')
md('## 7. Niche × CCC program activity')
code('fig,activities=report.niche_programs();show(fig);display(activities)')
md('## 8. Representative CCC, upstream regulators, and downstream targets\n\n[CollecTRI](https://github.com/saezlab/CollecTRI) signed TF–ligand priors connect sender regulon proxies to H. [NicheNet-v2](https://zenodo.org/records/7074291) connects ligands to measured receiver target modules. Source URLs, raw-file checksums, and panel/feature matching are retained in the audit and prior manifest. Activities use exactly the original directed physical pairs and spatial weighting. Niche CCC/prior profiles use training-block soft program contributions; the activity heatmaps above use mean Z of assigned cells. Prior dictionary loadings below indicate communication-linked support, rather than differential TF activity or proof of signaling. Calibration-valid flags and evidence are available in the complete CSVs. No patient phenotypes are inferred.')
code("overview,tables=report.interpretation();display(overview)\nshow(report.biological_heatmaps())\nfor niche in range(1,report.k+1):\n    print(f'Niche {niche}')\n    for branch,table in tables.items():\n        print(branch)\n        display(table.loc[table.niche==niche].head(5))\ndiagnostics=json.loads((OUTPUT/'diagnostics.json').read_text())\ndisplay(pd.Series(diagnostics).to_frame('Diagnostic'))\nprint('Composition-only niche prediction: {:.1%}; majority-niche baseline: {:.1%}.'.format(diagnostics.get('composition_only_niche_test_accuracy',float('nan')), diagnostics['dominant_niche_test_fraction']))\nprint('Held-out composition-to-Z R²: {:.3f}; spatial-neighbor niche agreement: {:.1%}.'.format(diagnostics['composition_to_Z_test_R2'], diagnostics['test_spatial_neighbor_agreement']))")
md('## 9. Comparison with original CCC-vector AE\n\nOriginal AE results and weights are preserved. These same-tissue maps and assignment correspondence are descriptive: the archived AE used overlapping random folds, so they do not establish comparative generalization. Composition-only prediction, spatial agreement, effective programs, and shuffled-prior/activity diagnostics above reveal important limits. Full training ablations and independent-seed GNN replication remain necessary before attributing benefit to biological losses or claiming reproducible disease niches.')
code("fig,comparison=report.comparison();show(fig);display(comparison)\nprint('All PNG (600 dpi), editable-text PDF, and result tables:', OUTPUT)\nprint('Checkpoint, Z, P, H, native CCC reconstruction, training and K-selection records are saved separately.')")
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'cccphe','language':'python','name':'cccphe'},'language_info':{'name':'python'}}
path=Path(__file__).resolve().parents[1]/'XeniumPrime5K_GNN_niche_analysis.ipynb'
nbformat.write(nb,path)
try:
    NotebookClient(nb,timeout=1800,kernel_name='cccphe',resources={'metadata':{'path':str(path.parent)}}).execute()
finally:
    nbformat.write(nb,path)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.get('outputs',[])) for c in nb.cells if c.cell_type=='code')
images=sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
assert images>=12, f'Expected actual rendered figures; got {images}'
(out/'figures/spatial_niches.png').unlink(missing_ok=True)  # Superseded by the styled PNG/PDF tissue overview.
print(f'Executed cells: {sum(c.cell_type=="code" for c in nb.cells)}; displayed figures: {images}',flush=True)
print(path,flush=True)
