import torch
from phenoniche.data.scaling import BlockScaler
from phenoniche.model.niche_factorization import NicheFactorization
from phenoniche.model.normalization import normalize_factors


def test_normalization_preserves_all_reconstructions_and_risk():
    model = NicheFactorization(9, 8, 4, 6, 7, 3, composition=torch.rand(9, 4), communication=torch.rand(9, 7))
    with torch.no_grad():
        model.gamma.copy_(torch.tensor([0.3, -0.8, 0.2]))
    factors = model.factors()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    normalized = normalize_factors(factors)
    for w, h in (("WB", "HC"), ("WB", "HI"), ("WS", "HC"), ("WS", "HO"), ("WS", "HI")):
        torch.testing.assert_close(factors[w] @ factors[h], normalized[w] @ normalized[h])
    torch.testing.assert_close(factors["WB"] @ factors["gamma"], normalized["WB"] @ normalized["gamma"])
    torch.testing.assert_close(normalized["HC"].sum(1), torch.ones(3))
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())


def test_scaling_roundtrip_shared_units(tiny_data):
    scaler = BlockScaler.fit(tiny_data)
    scaled = scaler.transform(tiny_data)
    restored = scaler.inverse_transform(scaled)
    for name, value in tiny_data.blocks().items():
        torch.testing.assert_close(restored.blocks()[name], value)
    torch.testing.assert_close(scaled.CB * scaler.composition, tiny_data.CB)
    torch.testing.assert_close(scaled.CS * scaler.composition, tiny_data.CS)
    torch.testing.assert_close(scaled.IB * scaler.communication, tiny_data.IB)
    torch.testing.assert_close(scaled.IS * scaler.communication, tiny_data.IS)
    assert torch.equal(scaled.time, tiny_data.time)


def test_zero_block_scaling_is_finite(tiny_data):
    from dataclasses import replace
    data = replace(tiny_data, IB=torch.zeros_like(tiny_data.IB), IS=torch.zeros_like(tiny_data.IS))
    scaler = BlockScaler.fit(data)
    assert scaler.communication == 1
    assert torch.isfinite(scaler.transform(data).IB).all()
