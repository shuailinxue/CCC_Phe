"""Post-fit comparisons on identical centers, including matched programs and label scopes."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .storage import digest, key


def compare(runs,source,label_key,classes,root,coarse=None,v0=None):
    """runs maps descriptive names to completed V1 runs; the first is the reference."""
    dependencies = {"source":digest(source),"label":label_key,"classes":list(classes),
                    "code":digest(__file__),"runs":{n:digest(Path(p)/"cells.npz") for n,p in runs.items()},
                    "coarse":digest(coarse) if coarse else None,"v0":digest(v0) if v0 else None}
    dest = Path(root)/("comparison-"+key(dependencies))
    if (dest/"complete").exists():
        return dest
    dest.mkdir(exist_ok=True,parents=True)
    a = ad.read_h5ad(source,backed="r")
    ids,truth = a.obs_names.to_numpy(dtype=str),a.obs[label_key].astype("string").fillna("").to_numpy(dtype=str)
    a.file.close()
    selected = np.isin(truth,classes)
    refname = next(iter(runs))
    def score(name,y):
        return dict(method=name,hub_centers=int(selected.sum()),hub_ARI=adjusted_rand_score(truth[selected],y),
                    hub_NMI=normalized_mutual_info_score(truth[selected],y))
    rows,models = [],{}
    for name,path in runs.items():
        path = Path(path)
        meta = json.loads((path/"manifest.json").read_text())
        with np.load(path/"cells.npz") as cells:
            if not np.array_equal(cells["ids"],ids):
                raise ValueError("Fit and evaluation IDs disagree")
            y = cells["niche"]
            models[name] = y
            row = score(name,y[selected])
            row["ARI_to_reference"] = adjusted_rand_score(models[refname],y)
            if meta["settings"]["mode"]=="joint" and meta["settings"]["k"]>1:
                # Choose the strongest competitor under composition alone.
                comp,lat = cells["composition"],cells["latent"]
                idx = np.arange(len(y))
                others = comp.copy()
                others[idx,y] = -np.inf
                k = others.argmax(1)
                row.update(median_composition_margin=np.median(comp[idx,y]-comp[idx,k]),
                           median_program_margin=np.median(lat[idx,y]-lat[idx,k]),
                           composition_prefers_competitor=np.mean(comp[idx,y]<comp[idx,k]),
                           program_favors_winner=np.mean(lat[idx,y]>lat[idx,k]))
        metrics = json.loads((path/"metrics.json").read_text())
        row.update(metrics["heldout"])
        row["empty_niches"] = metrics["empty_niches"]
        if (path/"programs.npz").is_file():
            with np.load(path/"programs.npz") as p:
                vectors = np.hstack((p["occurrence"],p["intensity"])).astype(float)
            vectors /= np.maximum(np.linalg.norm(vectors,axis=1,keepdims=True),1e-12)
            if name==refname:
                reference = vectors
            if name!=refname and (Path(runs[refname])/"programs.npz").exists():
                similarity = vectors@reference.T
                r,c = linear_sum_assignment(-similarity)
                row["matched_program_cosine"] = similarity[r,c].mean()
                pd.DataFrame(dict(program=r,reference_program=c,cosine=similarity[r,c])).to_csv(
                    dest/f"{name}_program_matches.tsv",sep="\t",index=False)
        rows.append(row)
    if coarse:
        with np.load(coarse) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["input_sha256"]!=dependencies["source"] or not np.array_equal(z["cell_ids"],ids):
                raise ValueError("Coarse-label scope mismatch")
            table = pd.crosstab(pd.Series(z["values"],name="V0_coarse"),pd.Series(models[refname],name="V1"))
            table.to_csv(dest/"coarse_by_niche.tsv",sep="\t")
    if v0:
        with np.load(v0) as z:
            old = pd.Series(z["labels"],index=pd.Index(z["cell_ids"],dtype=str))
            if not old.index.is_unique or set(old.index)!=set(ids[selected]):
                raise ValueError("V0 and Hub evaluation scopes differ")
            label = old.loc[ids[selected]].to_numpy()
        rows.append(score("V0_H1_Hub_trained",label))
    pd.DataFrame(rows).to_csv(dest/"metrics.tsv",sep="\t",index=False)
    (dest/"manifest.json").write_text(json.dumps(dependencies,indent=2))
    (dest/"complete").touch()
    return dest
