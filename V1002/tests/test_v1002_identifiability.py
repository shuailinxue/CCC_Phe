import numpy as np
from phenoniche.v1002.oracle_identifiability import oracle_pass
from phenoniche.v1002.identifiability_run import cox_fit


def test_oracle_gate_rejects_reverse_direction_separation():
    good = {"N1_N2": {"CS_AUC": .51, "IS_AUC": .95},
            "N3_N4": {"CS_AUC": .93, "IS_AUC": .49},
            "background_vs_niche_CS_IS_AUC": .96}
    assert oracle_pass([good] * 3)
    bad = dict(good, N3_N4={"CS_AUC": .93, "IS_AUC": .30})
    assert not oracle_pass([good, bad, good])


def test_signed_alr_cox_design_is_accepted_and_recovers_direction():
    rng = np.random.default_rng(83)
    x = rng.normal(size=(320, 2))
    eta = 1.1 * x[:, 0] - 1.1 * x[:, 1]
    failure = rng.exponential(size=len(x)) / np.exp(eta)
    censor = rng.exponential(scale=np.median(failure) * 2, size=len(x))
    time = np.minimum(failure, censor).astype(np.float32)
    event = (failure <= censor).astype(np.float32)
    result = cox_fit(x, time, event)
    assert result["gamma"][0] > 0 and result["gamma"][1] < 0
    assert 0 <= result["validation_C"] <= 1
