import torch
from phenoniche.losses.survival import cox_breslow_loss
from phenoniche.model.dual_bank_factorization import DualBankFactorization, set_trainable_bank


def test_cox_gradient_reaches_only_phenotype_dictionary_path(tiny_data):
    model = DualBankFactorization(24, 20, 4, 5, 6, 2, 2,
                                  composition=tiny_data.CB, communication=tiny_data.IB,
                                  inner_steps=12)
    set_trainable_bank(model, background=False, phenotype=True)
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([0.4, -0.3]))
    output = model()
    loss = cox_breslow_loss(output["risk"], tiny_data.time, tiny_data.event)
    loss.backward()
    assert model.raw_HCp.grad is not None and model.raw_HCp.grad.abs().sum() > 0
    assert model.raw_HIp.grad is not None and model.raw_HIp.grad.abs().sum() > 0
    assert model.gamma.grad is not None and model.gamma.grad.abs().sum() > 0
    assert model.raw_HC0.grad is None
    assert model.raw_HI0.grad is None
    assert model.raw_HO0.grad is None
    assert model.raw_WS0.grad is None


def test_phenotype_head_does_not_read_background_activity(tiny_data):
    model = DualBankFactorization(24, 20, 4, 5, 6, 2, 2)
    wb0 = torch.rand(24, 2, requires_grad=True)
    wbp = torch.rand(24, 2, requires_grad=True)
    risk = model.phenotype_head(wbp)
    assert torch.autograd.grad(risk.sum(), wb0, allow_unused=True)[0] is None
    assert torch.autograd.grad(risk.sum(), wbp)[0] is not None
