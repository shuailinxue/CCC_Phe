import torch
from phenoniche.evaluation.bank_matching import bank_matching_report


def test_bank_matching_assigns_roles_and_margins():
    truth = {
        "HC": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "HI": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "HO": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "WB": torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]),
        "WS": torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]]),
        "gamma": torch.tensor([1.0, 0.0]),
    }
    factors = {
        "HCp": torch.tensor([[1.0, 0.0]]), "HIp": torch.tensor([[1.0, 0.0]]),
        "HOp": torch.tensor([[1.0, 0.0]]), "HC0": torch.tensor([[0.0, 1.0]]),
        "HI0": torch.tensor([[0.0, 1.0]]), "HO0": torch.tensor([[0.0, 1.0]]),
        "WSp": truth["WS"][:, :1], "WS0": truth["WS"][:, 1:], "gamma": torch.tensor([0.5]),
    }
    report = bank_matching_report(factors, truth, ["risk", "neutral"],
                                  truth["WB"][:, 1:], truth["WB"][:, :1], truth["WB"], ["A", "B"])
    assert report["by_label"]["A"]["best_bank"] == "phenotype"
    assert report["by_label"]["A"]["bank_margin"] > 0
    assert report["by_label"]["B"]["best_bank"] == "background"
    assert report["by_label"]["B"]["bank_margin"] < 0
    assert report["A_B_different_bank"] and report["A_B_different_factor"]
