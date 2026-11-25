"""Single-slice dual NMF control; read existing targets, never rerun GNN/CCC."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

from phenoniche.controls.joint_nmf import JointNMFConfig, initialize_factors, fit_joint_nmf, reconstruction_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', default='TNBC1_CN1_C2')
    parser.add_argument('--epochs', type=int, default=800)
    args = parser.parse_args()
    folder = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/tnbc_spot_dual')/args.sample
    output = folder/'joint_nmf_control'
    output.mkdir(exist_ok=True)
    import fcntl
    guard = (output/'worker.lock').open('w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = np.load(folder/'targets.npz')
    x, c = saved['X'], saved['C']
    gnn = np.load(folder/'programs.npz')
    if not np.array_equal(saved['spots'], gnn['spots']):
        raise ValueError('GNN and target spot order differs')
    configuration = json.loads((folder/'config.json').read_text())
    options = dict(programs=configuration['programs'], epochs=args.epochs,
                   regularization_weight=configuration['regularization_weight'], seed=configuration['seed'])
    base = JointNMFConfig(**options)
    initial = initialize_factors(x, c, base)
    scale = initial['feature_scale']
    rows = [dict(method='GNN dual', **reconstruction_metrics(x, c, gnn, scale))]
    for name, weight in [('direct_CCC_only', 0.), ('direct_dual', configuration['composition_weight'])]:
        path = output/name
        path.mkdir(exist_ok=True)
        config = JointNMFConfig(**options, composition_weight=weight)
        result = fit_joint_nmf(x, c, config, initialization=initial)
        np.savez_compressed(path/'factors.npz', **{k: result[k] for k in ('Z', 'P', 'H', 'B')})
        pd.DataFrame(result['history']).to_csv(path/'history.csv', index=False)
        metrics = reconstruction_metrics(x, c, result, scale)
        metrics.update(best_epoch=result['best_epoch'], config=result['config'])
        (path/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        # Same K search and representation as the saved spot GNN, no extra figure.
        p, z = result['P'], result['Z']
        activity = np.log1p(z.sum(1))
        activity = (activity-activity.mean())/max(activity.std(), 1e-6)
        embedding = np.c_[np.sqrt(p), .15*activity]
        scores = []
        with threadpool_limits(limits=4):
            for k in range(2, 13):
                labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(embedding)
                scores.append(dict(k=k, silhouette=float(silhouette_score(embedding, labels))))
        pd.DataFrame(scores).to_csv(path/'K_selection.csv', index=False)
        rows.append(dict(method=name, **{k: metrics[k] for k in rows[0] if k != 'method'},
                         selected_K=max(scores, key=lambda item: item['silhouette'])['k']))
    table = pd.DataFrame(rows)
    table.to_csv(output/'comparison.csv', index=False)
    (output/'protocol.json').write_text(json.dumps(dict(
        sample=args.sample, cells=len(x), features=x.shape[1], cell_types=c.shape[1],
        equations=['Xhat=Z@H', 'P=Z/row_sum(Z)', 'Chat=P@B'],
        shared_activity=True, separate_loading_matrices=True,
        objective='RMS-scaled CCC Huber + lambda*composition KL + same program regularization',
        constraints='Z,H nonnegative; rows of H,B sum to one',
        initialization='Identical NMF initialization for CCC-only and dual direct-factor controls',
        parameters='Direct per-spot activities; no encoder, no graph regularization, no phenotype labels',
        scope='Full observed section, in-sample reconstruction; not a complete SpaNiche implementation',
        optimizer='Adam, activity/B lr0.01, projected H lr0.00005; full-batch, minimum own total objective',
    ), indent=2)+'\n')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
