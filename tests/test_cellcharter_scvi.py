"""Native CellCharter scVI interface and actual three-hop donor isolation."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from baselines.cellcharter_adapter import safe_test_rows


def test_three_hop_dependency_purge():
    rows=np.arange(7);g=sp.diags([np.ones(6),np.ones(6)],[-1,1],shape=(7,7)).tocsr()
    owner=np.array([0,0,0,0,1,1,1])
    safe=safe_test_rows(g,owner,3)
    np.testing.assert_array_equal(safe,[True,False,False,False,False,False,False])


def test_official_scvi_config_not_pca():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'baselines/config.json').read_text())
    assert cfg['cellcharter_representation']=='scVI'
    assert cfg['cellcharter_layers']==3 and cfg['cellcharter_scvi_epochs']==400
    assert Path(cfg['cellcharter_scvi_python']).is_file()
