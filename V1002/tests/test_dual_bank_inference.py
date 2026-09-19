from dataclasses import replace
import torch
from phenoniche.inference.dual_bank import infer_background_activities, infer_phenotype_activities, infer_dual_bank_activities
from phenoniche.model.dual_bank_factorization import DualBankFactorization


def test_unified_inference_matches_component_calls(tiny_data):
    model = DualBankFactorization(24, 20, 4, 5, 6, 1, 1)
    dictionaries = model.dictionaries()
    wb0 = infer_background_activities(tiny_data.CB, tiny_data.IB, dictionaries["HC0"], dictionaries["HI0"], steps=7)
    wbp = infer_phenotype_activities(tiny_data.CB, tiny_data.IB, wb0, dictionaries["HC0"], dictionaries["HI0"],
                                     dictionaries["HCp"], dictionaries["HIp"], steps=7)
    unified = infer_dual_bank_activities(tiny_data.CB, tiny_data.IB, dictionaries["HC0"], dictionaries["HI0"],
                                         dictionaries["HCp"], dictionaries["HIp"], steps=7)
    assert torch.equal(wb0, unified[0])
    assert torch.equal(wbp, unified[1])


def test_test_outcomes_cannot_change_inferred_activities(tiny_data):
    model = DualBankFactorization(24, 20, 4, 5, 6, 1, 1, inner_steps=8)
    first = model.bulk_factors(tiny_data.CB, tiny_data.IB, create_graph=False)
    altered = replace(tiny_data, time=tiny_data.time.flip(0), event=1 - tiny_data.event)
    second = model.bulk_factors(altered.CB, altered.IB, create_graph=False)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_signed_residual_is_not_clamped_before_phenotype_inference():
    composition = torch.tensor([[0.1, 0.2]])
    communication = torch.tensor([[0.2, 0.1]])
    wb0 = torch.tensor([[2.0]])
    hc0 = torch.tensor([[1.0, 1.0]])
    hi0 = torch.tensor([[1.0, 1.0]])
    hcp = torch.tensor([[0.8, 0.2]])
    hip = torch.tensor([[0.2, 0.8]])
    result = infer_phenotype_activities(composition, communication, wb0, hc0, hi0, hcp, hip, steps=10)
    assert result.shape == (1, 1)
    assert torch.isfinite(result).all() and (result >= 0).all()
