from phenoniche.utils.validation import matrix, same_context


def normalize_factors(factors):
    names = ("WB", "WS", "HC", "HO", "HI")
    for name in names:
        matrix(factors[name], name)
    same_context(tuple(factors[name] for name in (*names, "gamma")))
    k = factors["HC"].shape[0]
    if any(factors[n].shape[0] != k for n in ("HO", "HI")) or any(factors[n].shape[1] != k for n in ("WB", "WS")):
        raise ValueError("All factors must have the same niche count")
    if factors["gamma"].shape != (k,) or not factors["gamma"].isfinite().all():
        raise ValueError("gamma must be finite with shape [K]")
    scale = factors["HC"].sum(dim=1)
    if (scale <= 0).any():
        raise ValueError("Cannot normalize an empty composition dictionary row")
    return {"WB": factors["WB"] * scale, "WS": factors["WS"] * scale,
            **{name: factors[name] / scale[:, None] for name in ("HC", "HO", "HI")},
            "gamma": factors["gamma"] / scale}
