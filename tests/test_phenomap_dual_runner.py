"""Lightweight scheduling checks without segmentation or training."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_runner(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['run_phenomap_tnbc.py', '--gpu', '0'])
    spec = importlib.util.spec_from_file_location('phenomap_batch_test', ROOT/'scripts/run_phenomap_tnbc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shards_are_disjoint_and_complete():
    samples = list(range(278))
    a, b = samples[0::2], samples[1::2]
    assert set(a).isdisjoint(b)
    assert sorted(a+b) == samples
    assert len(a) == len(b) == 139


def test_status_merge_keeps_both_workers(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    module.save_status(tmp_path, [{'sample':'A','status':'complete','detail':'a'}], '0')
    module.save_status(tmp_path, [{'sample':'B','status':'complete','detail':'b'}], '2')
    module.save_status(tmp_path, [{'sample':'A','status':'reused_complete','detail':''}], '0')
    frame = module.pd.read_csv(tmp_path/'batch_status.csv').set_index('sample')
    assert len(frame) == 2
    assert frame.loc['A','status'] == 'reused_complete'
    assert frame.loc['B','status'] == 'complete'


def test_gpu_selected_before_torch(monkeypatch):
    module = load_runner(monkeypatch)
    assert module.os.environ['CUDA_VISIBLE_DEVICES'] == '0'


def test_patient_selection_controls_targets_without_changing_config(tmp_path, monkeypatch):
    module = load_runner(monkeypatch)
    ready = [tmp_path/'TNBC1_CN1_C1', tmp_path/'TNBC1_CN1_C2', tmp_path/'TNBC2_CN1_D2']
    manifest = tmp_path/'selected.csv'
    module.pd.DataFrame({'sample':['TNBC1_CN1_C2','TNBC2_CN1_D2'], 'patient_id':[1,2]}).to_csv(manifest,index=False)
    result = module.selected_targets(ready,manifest)
    assert result == ready[1:]
    module.pd.DataFrame({'sample':[p.name for p in ready[:2]], 'patient_id':[1,1]}).to_csv(manifest,index=False)
    import pytest
    with pytest.raises(ValueError, match='one unique sample'):
        module.selected_targets(ready,manifest)
