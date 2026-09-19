import torch
from phenoniche.data.synthetic import generate_synthetic
from phenoniche.evaluation.matching import match_niches, recovery_report
from phenoniche.evaluation.metrics import concordance_index


def test_generator_exact_ground_truth_without_noise():
    synthetic = generate_synthetic(noise=0)
    data, truth = synthetic.data, synthetic.truth
    for name, w, h in (("CB", "WB", "HC"), ("IB", "WB", "HI"), ("CS", "WS", "HC"), ("OS", "WS", "HO"), ("IS", "WS", "HI")):
        torch.testing.assert_close(getattr(data, name), truth[w] @ truth[h])
    assert 0 < data.event.mean() < 1
    assert truth["gamma"][0] > 0 and truth["gamma"][1] < 0
    assert (truth["gamma"][2:] == 0).all()


def test_matching_corrects_permutation_and_scaling():
    truth = generate_synthetic().truth
    order = torch.tensor([2, 0, 3, 1])
    learned = {name: value[order] * 2 if name.startswith("H") else value[:, order] / 2
               for name, value in truth.items() if name != "gamma"}
    learned["gamma"] = truth["gamma"][order] * 2
    assert match_niches(learned, truth).tolist() == [1, 3, 0, 2]
    report = recovery_report(learned, truth)
    for name in ("HC", "HI", "HO", "WB", "WS"):
        assert report[name]["cosine"] > 0.99999
        assert report[name]["pearson"] > 0.99999


def test_synthetic_recovery(recovered):
    data, truth, result = recovered
    report = recovery_report(result.model.factors(), truth)
    for name in ("HC", "HI", "HO", "WB", "WS"):
        assert report[name]["cosine"] > 0.92, (name, report[name])
        assert report[name]["pearson"] > 0.88, (name, report[name])
    assert report["matched_gamma"][0] > 0
    assert report["matched_gamma"][1] < 0
    risk = result.model()["risk"].detach().numpy()
    assert concordance_index(data.time.numpy(), data.event.numpy(), risk) > 0.75
    assert result.history[-1]["bulk_recon"] < result.history[0]["bulk_recon"] * 0.1
