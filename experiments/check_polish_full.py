"""Compare fitted sampled predictions to the matching full-stream centers."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("selection",type=Path)
    p.add_argument("inference",type=Path)
    p.add_argument("--expected-cells",type=int,required=True)
    a=p.parse_args()
    if not (a.inference/"complete").is_file():
        raise ValueError("Full inference incomplete")
    with np.load(a.selection/"final73/cells.npz") as x,np.load(a.inference/"cells.npz") as y:
        idx=pd.Index(y["ids"]).get_indexer(x["ids"])
        assert len(y["ids"])==a.expected_cells and (idx>=0).all()
        np.testing.assert_array_equal(y["mass"][idx],x["mass"])
        assert np.isfinite(y["soft"]).all()
        np.testing.assert_allclose(y["soft"].sum(1),1.,atol=1e-6)
        np.testing.assert_allclose(x["soft"],y["soft"][idx],atol=3e-4)
        audit=dict(full_centers=len(y["ids"]),matched_centers=len(idx),
            memberships_max_abs_diff=float(np.abs(x["soft"]-y["soft"][idx]).max()),
            sampled_hard_label_agreement=float(np.mean(x["niche"]==y["niche"][idx])))
    (a.inference/"sample_alignment_audit.json").write_text(json.dumps(audit,indent=2))
    print(audit)


if __name__=="__main__":
    main()
