from pathlib import Path
import json
import torch
from phenoniche.v1002.exact_loss import dense_memory_bytes, exact_chunked_frobenius_mse
from phenoniche.v1002.lr_atlas import discover_lr_atlas, load_lr_atlas


REQUIRED_OUTPUTS = ("small_lr.json", "medium_lr.json", "full_lr.json", "st_recovery.json",
                    "bulk_bridge.json", "pseudobulk_bridge.json", "cox_results.json",
                    "large_lr_edge_recovery.json", "scaling_runtime_memory.json", "exact_loss_audit.json")


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def initialize_v1002(output, project_root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atlas_path, searched = discover_lr_atlas(project_root)
    if atlas_path is None:
        audit = {"status": "blocked_missing_atlas", "required_source": "CommuSpace human LR atlas",
                 "expected_snapshot_pairs": 9311, "searched_paths": searched,
                 "rejected_candidate": {"path": "/home/xueshuailin/.local/share/Trash/files/LR_database_union_all_pairs.csv",
                                        "reason": "CellChatDB/SingleCellSignalR union is not the specified CommuSpace atlas"},
                 "network_fallback_used": False, "synthetic_LR_names_used": False}
        _save(output / "lr_atlas_audit.json", audit)
        blocked = {"status": "not_run_missing_commuspace_atlas", "atlas_audit": "lr_atlas_audit.json"}
        for name in REQUIRED_OUTPUTS:
            _save(output / name, blocked)
        generator = torch.Generator().manual_seed(20261002)
        observed = torch.rand((9, 31), generator=generator, dtype=torch.float64)
        activity = torch.rand((9, 6), generator=generator, dtype=torch.float64, requires_grad=True)
        dictionary = torch.rand((6, 31), generator=generator, dtype=torch.float64, requires_grad=True)
        dense = (observed - activity @ dictionary).square().mean()
        dense_gradients = torch.autograd.grad(dense, (activity, dictionary), retain_graph=True)
        exact = exact_chunked_frobenius_mse(observed, activity, dictionary, feature_chunk_size=7)
        exact_gradients = torch.autograd.grad(exact, (activity, dictionary))
        exact_audit = {"status": "validated_without_atlas", "dense_mse": float(dense), "exact_mse": float(exact),
                       "absolute_error": abs(float(dense - exact)),
                       "activity_gradient_max_error": float((dense_gradients[0] - exact_gradients[0]).abs().max()),
                       "dictionary_gradient_max_error": float((dense_gradients[1] - exact_gradients[1]).abs().max()),
                       "allclose": bool(torch.allclose(dense, exact, atol=1e-12, rtol=1e-12))}
        _save(output / "exact_loss_audit.json", exact_audit)
        expected_lr = 9311
        features = 8 * 8 * expected_lr
        scaling = {"status": "projected_only_missing_atlas", "expected_snapshot_nLR": expected_lr,
                   "expected_F": features, "dense_float32_IS_bytes": dense_memory_bytes(1200, features),
                   "dense_float32_HI_bytes": dense_memory_bytes(6, features),
                   "dense_IS_exceeds_1GB_guard": dense_memory_bytes(1200, features) > 1_000_000_000}
        _save(output / "scaling_runtime_memory.json", scaling)
        summary = {"status": "blocked_missing_atlas", "six_answers": {
            "1_large_atlas_ST_stable": "not_evaluated", "2_irrelevant_LR_pollution": "not_evaluated",
            "3_bulk_directional_potential_recovers_niches": "not_evaluated",
            "4_CCC_adds_beyond_composition": "not_evaluated",
            "5_pseudobulk_bridge_supported": "not_evaluated",
            "6_tensor_factorization_needed": "not_evaluated"},
                   "exact_loss_validated": exact_audit["allclose"], "projected_memory": scaling,
                   "conclusion": "No A-E conclusion is valid without the specified real CommuSpace atlas."}
        _save(output / "summary.json", summary)
        return summary
    atlas = load_lr_atlas(atlas_path)
    _save(output / "lr_atlas_audit.json", atlas.audit)
    generator = torch.Generator().manual_seed(20261002)
    observed = torch.rand((9, 31), generator=generator, dtype=torch.float64)
    activity = torch.rand((9, 6), generator=generator, dtype=torch.float64, requires_grad=True)
    dictionary = torch.rand((6, 31), generator=generator, dtype=torch.float64, requires_grad=True)
    dense = (observed - activity @ dictionary).square().mean()
    dense_gradients = torch.autograd.grad(dense, (activity, dictionary), retain_graph=True)
    exact = exact_chunked_frobenius_mse(observed, activity, dictionary, feature_chunk_size=7)
    exact_gradients = torch.autograd.grad(exact, (activity, dictionary))
    _save(output / "exact_loss_audit.json", {"status": "validated", "dense_mse": float(dense), "exact_mse": float(exact),
          "absolute_error": abs(float(dense - exact)), "activity_gradient_max_error": float((dense_gradients[0] - exact_gradients[0]).abs().max()),
          "dictionary_gradient_max_error": float((dense_gradients[1] - exact_gradients[1]).abs().max()),
          "allclose": bool(torch.allclose(dense, exact, atol=1e-12, rtol=1e-12))})
    return {"status": "atlas_loaded", "atlas": atlas}
