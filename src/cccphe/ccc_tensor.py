from __future__ import annotations

import numpy as np


def ligand_receptor_support(
    ligand_expression: np.ndarray,
    receptor_expression: np.ndarray,
) -> np.ndarray:
    """Geometric-mean molecular support sqrt(ligand * receptor)."""
    ligand = np.maximum(np.asarray(ligand_expression, dtype=np.float64), 0.0)
    receptor = np.maximum(np.asarray(receptor_expression, dtype=np.float64), 0.0)
    return np.sqrt(ligand * receptor)


def construct_patient_ccc(
    w_st: np.ndarray,
    m_st: np.ndarray,
    m_rna: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Construct patient exact CCC propensity from the current study ST.

    C[s,a,b,p] = W_ST[a,b,p] * (M_RNA[s,a,b,p] + eps)
                                  / (M_ST[a,b,p] + eps)

    The ST used for the current study is the sole spatial anchor. This function
    intentionally has no separate reference-ST/validation-ST arguments.
    """
    w_st = np.asarray(w_st, dtype=np.float64)
    m_st = np.asarray(m_st, dtype=np.float64)
    m_rna = np.asarray(m_rna, dtype=np.float64)
    if w_st.shape != m_st.shape:
        raise ValueError("W_ST and M_ST must have identical sender x receiver x LR shape")
    if m_rna.ndim != 4 or m_rna.shape[1:] != w_st.shape:
        raise ValueError("M_RNA must have sample x sender x receiver x LR shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = w_st[None, ...] * np.exp(
        np.log(m_rna + epsilon) - np.log(m_st[None, ...] + epsilon)
    )
    return values.astype(np.float32)
