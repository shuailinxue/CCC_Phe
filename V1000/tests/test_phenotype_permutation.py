import torch
from phenoniche.data.challenging_synthetic import ChallengingConfig, generate_challenging
from phenoniche.evaluation.patient_protocol import split_patients, subset_bulk, permute_training_phenotype
from phenoniche.evaluation.challenging_benchmark import BenchmarkConfig, run_benchmark


def test_permutation_preserves_inputs_and_joint_time_event_pairs():
    sample = generate_challenging(ChallengingConfig(patients=60, anchors=30))
    split = split_patients(60)
    train = subset_bulk(sample.data, split.train)
    original = torch.stack((train.time, train.event), 1)
    draws = []
    for seed in (101, 102, 103, 104, 105):
        permuted = permute_training_phenotype(train, seed)
        for name, value in train.blocks().items():
            assert torch.equal(value, permuted.blocks()[name])
        pairs = torch.stack((permuted.time, permuted.event), 1)
        assert sorted(map(tuple, original.tolist())) == sorted(map(tuple, pairs.tolist()))
        assert not torch.equal(pairs, original)
        draws.append(pairs)
        again = permute_training_phenotype(train, seed)
        assert torch.equal(again.time, permuted.time)
    assert not torch.equal(draws[0], draws[1])
    assert torch.equal(train.time, sample.data.time[split.train])


def test_end_to_end_evaluation_outputs_and_selection_order(tmp_path):
    data_config = ChallengingConfig(patients=60, anchors=24)
    config = BenchmarkConfig(model_seeds=(3, 4), permutation_seeds=(10, 11), lambdas=(0.0, 0.01), warmup_epochs=3, joint_epochs=5)
    summary = run_benchmark(tmp_path, data_config, config)
    assert summary["selection"]["test_outcomes_used"] is False
    assert summary["selected_lambda"] == 0.01
    assert summary["comparison"]["supervised"]["test_c_index"]["n"] == 2
    assert summary["permutation"]["test_c_index"]["n"] == 2
    for name in ("summary.json", "lambda_sweep.json", "seed_stability.json", "phenotype_permutation.json",
                 "supervised_factors.npz", "unsupervised_factors.npz", "ground_truth.npz", "selection.json"):
        assert (tmp_path / name).is_file()
    assert (tmp_path / "selection.json").stat().st_mtime_ns <= (tmp_path / "lambda_sweep.json").stat().st_mtime_ns
