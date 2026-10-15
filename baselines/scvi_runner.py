"""Run official scVI in its compatible environment; no region/type/CCC inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '6')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
import json
import sys
import anndata as ad
import numpy as np
import torch
import scvi


def run(input_path, output, config_path):
    output = Path(output)
    config = json.loads(Path(config_path).read_text())
    torch.set_num_threads(config.get('cpu_threads', 6))
    scvi.settings.seed = config['seed']
    a = ad.read_h5ad(input_path)
    train = np.load(output/'training_cells.npy')
    fitting = a[train].copy()
    # Official CellCharter transcriptomics tutorial: counts, sample batch, ZINB,
    # default SCVI architecture (128 hidden, 10 latent, one layer).
    batch_key = 'sample_id'
    if not set(a.obs.sample_id).issubset(set(fitting.obs.sample_id)):
        # Unseen samples have no estimable batch embedding. A shared batch is
        # required for frozen inference; never fit query expression to adapt it.
        batch_key = None
    scvi.model.SCVI.setup_anndata(fitting, batch_key=batch_key)
    model = scvi.model.SCVI(fitting, n_latent=10, gene_likelihood='zinb')
    gpu = config['device'].startswith('cuda:')
    model.train(max_epochs=config['cellcharter_scvi_epochs'], early_stopping=True,
                accelerator='gpu' if gpu else 'cpu',
                devices=[int(config['device'].split(':')[1])] if gpu else 1,
                enable_progress_bar=False)
    model.save(output/'scvi_model', overwrite=True)
    np.save(output/'scvi_training_latent.npy', model.get_latent_representation().astype(np.float32))
    np.save(output/'scvi_latent.npy', model.get_latent_representation(a).astype(np.float32))
    for key, history in model.history.items():
        history.to_csv(output/('scvi_'+key+'.csv'))
    (output/'scvi_audit.json').write_text(json.dumps({
        'scvi_tools':scvi.__version__, 'torch':torch.__version__,
        'training_cells':len(train), 'input':'raw counts', 'n_latent':10,
        'gene_likelihood':'zinb', 'max_epochs':config['cellcharter_scvi_epochs'],
        'epochs_completed':len(model.history['elbo_train']),
        'batch_key':batch_key,
        'batch_policy':'Use sample covariate only when all query sample IDs occur in training; otherwise shared batch for frozen inference',
        'cell_type_CCC_or_truth_input':False,
        'python':sys.executable}, indent=2)+'\n')


if __name__ == '__main__':
    run(*sys.argv[1:])
