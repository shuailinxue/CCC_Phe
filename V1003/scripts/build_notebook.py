from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
notebook = nbf.v4.new_notebook()
notebook.metadata.kernelspec = {"display_name": "Python (cccphe)", "language": "python", "name": "python3"}
notebook.cells = [
    nbf.v4.new_markdown_cell(
        "# Xenium Prime 5K Breast · V1003\n"
        "CCC-only interpretable autoencoder. K=8, seed=40700. No graph, Leiden, tensor model, bulk, or Cox analysis."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT=Path('/home/xueshuailin/CCC_Phe/V1003')\n"
        "sys.path.insert(0,str(ROOT/'src'))\n"
        "from phenoniche.v1003.pipeline import ensure_results\n"
        "from phenoniche.v1003 import reporting as report\n"
        "import importlib\nreport=importlib.reload(report)\n"
        "summary=ensure_results()  # trains only when a complete cached result is absent\n"
        "artifacts=report.load_artifacts()"
    ),
]
sections = [
    ("1. Prime 5K data summary", "report.dataset_summary(artifacts)"),
    ("2. Cell types and V1003 niche spatial maps", "report.overview_map(artifacts)"),
    ("3. V1003 niche × local cell-type enrichment", "report.niche_celltype_enrichment(artifacts)"),
    ("4. Spatial distribution of each niche", "report.spatial_niches(artifacts)"),
    ("5. Top representative CCC", "report.top_ccc_table(artifacts,top=10)"),
    ("6. Representative H heatmap", "report.dictionary_heatmap(artifacts,per_niche=3)"),
    ("7. Final model and niche summary", "report.compact_results(artifacts)"),
]
for title, code in sections:
    notebook.cells.extend([nbf.v4.new_markdown_cell("## " + title), nbf.v4.new_code_cell(code)])
nbf.write(notebook, ROOT / "XeniumPrime5K_Breast_V1003_niche_walkthrough.ipynb")
