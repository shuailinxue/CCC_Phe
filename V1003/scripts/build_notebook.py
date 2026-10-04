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
        "summary=ensure_results()  # trains only when a complete cached result is absent\n"
        "artifacts=report.load_artifacts()"
    ),
]
sections = [
    ("1. Prime 5K data summary", "report.dataset_summary(artifacts)"),
    ("2. Model and final training result", "report.model_and_training(artifacts)"),
    ("3. Cell types and V1003 niche overview",
     "import importlib\nreport=importlib.reload(report)  # pick up reporting updates in an existing Jupyter kernel\nreport.overview_map(artifacts)"),
    ("4. Spatial distribution of eight niches", "report.spatial_niches(artifacts)"),
    ("5. Niche size and proportion", "report.niche_counts(artifacts)"),
    ("6. Top representative CCC",
     "import importlib\nreport=importlib.reload(report)\nartifacts=report.load_artifacts()\nreport.representative_ccc(artifacts,top=15)"),
    ("7. Representative H heatmap", "report.dictionary_heatmap(artifacts,per_niche=3)"),
    ("8. Assignment confidence and niche usage", "report.confidence_and_usage(artifacts)"),
    ("9. Final summary", "report.final_summary(artifacts)"),
]
for title, code in sections:
    notebook.cells.extend([nbf.v4.new_markdown_cell("## " + title), nbf.v4.new_code_cell(code)])
nbf.write(notebook, ROOT / "XeniumPrime5K_Breast_V1003_niche_walkthrough.ipynb")
