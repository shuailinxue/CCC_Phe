"""Outcome-blind observable audit before any V1002 model fit."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from phenoniche.v1002.final_experiment import make_blocks, seed_for
from phenoniche.v1002.final_simulation import simulate_final_spatial


def heldout_auc(values, labels, coordinates, positive, negative, split_seed=9001):
    chosen = np.flatnonzero((labels == positive) | (labels == negative))
    y = (labels[chosen] == positive).astype(np.int8)
    train, test = [], []
    for niche in (positive, negative):
        local = np.flatnonzero(labels[chosen] == niche)
        center = coordinates[chosen[local]].mean(0)
        relative = coordinates[chosen[local], 0] - center[0]
        order = local[np.argsort(relative)]
        train.extend(order[:int(.55 * len(order))])
        test.extend(order[int(.75 * len(order)):])
    train, test = np.asarray(train), np.asarray(test)
    x = np.asarray(values[chosen], dtype=np.float32)
    rms = np.sqrt(np.mean(x[train].astype(np.float64) ** 2))
    x = x / max(rms, 1e-12)
    classifier = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                                    max_iter=1000, random_state=split_seed)
    classifier.fit(x[train], y[train])
    score = classifier.predict_proba(x[test])[:, 1]
    return float(roc_auc_score(y[test], score))


def audit_one(simulation, blocks, seed):
    labels = simulation.labels
    cs = blocks["HC"]
    interaction = blocks["HI"]
    combined = np.concatenate((cs / np.sqrt(np.mean(cs ** 2)),
                               interaction / np.sqrt(np.mean(interaction ** 2))), axis=1)
    centroids = {name: np.stack([x[labels == k].mean(0) for k in range(6)])
                 for name, x in (("CS", cs), ("IS", interaction))}
    def cosine(a, b):
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    def pair(a, b):
        return {"CS_AUC": heldout_auc(cs, labels, simulation.coordinates, a, b),
                "IS_AUC": heldout_auc(interaction, labels, simulation.coordinates, a, b),
                "CS_centroid_cosine": cosine(centroids["CS"][a], centroids["CS"][b]),
                "IS_centroid_cosine": cosine(centroids["IS"][a], centroids["IS"][b])}
    # Background versus pooled niches is a separate balanced binary test.
    y = (labels != 0).astype(np.int8)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=9001)
    classifier = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=1000)
    classifier.fit(combined[train], y[train])
    bg_auc = float(roc_auc_score(y[test], classifier.predict_proba(combined[test])[:, 1]))
    members = simulation.neighborhoods.indices.reshape(len(labels), -1)
    radius = np.median(np.max(np.linalg.norm(simulation.coordinates[members] - simulation.coordinates[:, None], axis=2), axis=1))
    niche_diameter = 2 * np.sqrt(np.mean([(labels == k).sum() for k in range(1, 6)]) / np.pi)
    interior = {str(k): float(np.mean(np.mean(labels[members[labels == k]] == k, axis=1) >= .8)) for k in range(1, 6)}
    return {"seed": seed, "heldout_protocol": "per niche: leftmost 55% train, rightmost 25% test, middle 20% embargo; fixed by coordinates",
            "N1_N2": pair(1, 2), "N3_N4": pair(3, 4),
            "background_vs_niche_CS_IS_AUC": bg_auc,
            "physical": {"equivalent_niche_diameter": float(niche_diameter),
                         "median_neighborhood_radius": float(radius),
                         "diameter_ratio": float(niche_diameter / (2 * radius)),
                         "interior_fraction_by_niche": interior},
            "centroids": {name: value.tolist() for name, value in centroids.items()}}


def oracle_pass(rows):
    return all(max(row["N1_N2"]["CS_AUC"], 1 - row["N1_N2"]["CS_AUC"]) <= .60
               and row["N1_N2"]["IS_AUC"] >= .90
               and row["N3_N4"]["CS_AUC"] >= .90
               and max(row["N3_N4"]["IS_AUC"], 1 - row["N3_N4"]["IS_AUC"]) <= .60
               and row["background_vs_niche_CS_IS_AUC"] >= .90 for row in rows)


def audit_replicates(simulations, seeds):
    """Leave-one-independent-tissue-out logistic audit on one fixed feature mask."""
    if len(simulations) != 3:
        raise ValueError("Oracle audit expects the three frozen primary tissues")
    common = np.logical_and.reduce([s.final_mask.ravel() for s in simulations])
    features = np.flatnonzero(common)
    cs = [s.cs for s in simulations]
    interaction = [s.communication[:, features] for s in simulations]
    labels = [s.labels for s in simulations]
    def auc(view, selected, positive, negative, background=False):
        train_x = np.concatenate([view[j][select(labels[j], positive, negative, background)]
                                  for j in range(3) if j != selected])
        train_y = np.concatenate([target(labels[j][select(labels[j], positive, negative, background)], positive, background)
                                  for j in range(3) if j != selected])
        test_mask = select(labels[selected], positive, negative, background)
        test_x = view[selected][test_mask]
        test_y = target(labels[selected][test_mask], positive, background)
        scale = np.sqrt(np.mean(train_x.astype(np.float64) ** 2))
        model = LogisticRegression(C=1, solver="liblinear", class_weight="balanced", max_iter=1000,
                                   random_state=9001)
        model.fit(train_x / max(scale, 1e-12), train_y)
        return float(roc_auc_score(test_y, model.predict_proba(test_x / max(scale, 1e-12))[:, 1]))
    def select(label, positive, negative, background):
        return np.ones(len(label), dtype=bool) if background else (label == positive) | (label == negative)
    def target(label, positive, background):
        return (label != 0).astype(np.int8) if background else (label == positive).astype(np.int8)
    rows = []
    for held in range(3):
        centroids = {name: np.stack([x[held][labels[held] == k].mean(0) for k in range(6)])
                     for name, x in (("CS", cs), ("IS", interaction))}
        def cosine(name, a, b):
            left, right = centroids[name][a], centroids[name][b]
            return float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))
        def pair(a, b):
            return {"CS_AUC": auc(cs, held, a, b), "IS_AUC": auc(interaction, held, a, b),
                    "CS_centroid_cosine": cosine("CS", a, b), "IS_centroid_cosine": cosine("IS", a, b)}
        combined = [np.concatenate((c / np.sqrt(np.mean(c ** 2)),
                                    i / np.sqrt(np.mean(i ** 2))), axis=1) for c, i in zip(cs, interaction)]
        members = simulations[held].neighborhoods.indices.reshape(len(labels[held]), -1)
        coord = simulations[held].coordinates
        radius = np.median(np.max(np.linalg.norm(coord[members] - coord[:, None], axis=2), axis=1))
        diameter = 2 * np.sqrt(np.mean([(labels[held] == k).sum() for k in range(1, 6)]) / np.pi)
        rows.append({"seed": seeds[held], "N1_N2": pair(1, 2), "N3_N4": pair(3, 4),
                     "background_vs_niche_CS_IS_AUC": auc(combined, held, 1, 0, background=True),
                     "physical": {"equivalent_niche_diameter": float(diameter),
                                  "median_neighborhood_radius": float(radius),
                                  "diameter_ratio": float(diameter / (2 * radius)),
                                  "interior_fraction_by_niche": {str(k): float(np.mean(np.mean(labels[held][members[labels[held] == k]] == k, axis=1) >= .8)) for k in range(1, 6)}}})
    return {"rows": rows, "common_feature_count": len(features), "common_feature_indices": features,
            "centroids": {"CS": np.stack([np.stack([cs[j][labels[j] == k].mean(0) for k in range(6)]) for j in range(3)]),
                          "IS": np.stack([np.stack([interaction[j][labels[j] == k].mean(0) for k in range(6)]) for j in range(3)])},
            "heldout_protocol": "leave-one-independent-tissue-out; 2 tissues train, 1 tissue test; fixed common phenotype-free feature mask"}
