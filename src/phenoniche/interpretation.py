"""Export observed or predicted CCC feature summaries for niche interpretation."""
import numpy as np
import pandas as pd

def _feature_label(row):
    if "ccc" in row and pd.notna(row["ccc"]): return str(row["ccc"])
    return f'{row["sender"]} → {row["receiver"]} | {row["ligand"]}–{row["receptor"]}'


def export_top_markers(h8, features, output, top_n=15):
    records = []
    for niche in range(h8.shape[0]):
        for rank, feature in enumerate(np.argsort(-h8[niche], kind="stable")[:top_n], 1):
            row = features.iloc[feature]
            records.append({"niche": niche + 1, "rank": rank, "feature_id": int(feature),
                            "ccc": _feature_label(row), "sender": row["sender"], "receiver": row["receiver"],
                            "ligand": row["ligand"], "receptor": row["receptor"], "weight": float(h8[niche, feature])})
    top = pd.DataFrame(records); top.to_csv(output / "top_ccc.csv", index=False)
    for columns, filename, label in [(["sender", "receiver"], "top_sender_receiver.csv", "sender_receiver"),
                                      (["ligand", "receptor"], "top_lr.csv", "lr")]:
        rows = []
        for niche in range(h8.shape[0]):
            table = features[columns].copy(); table["weight"] = h8[niche]
            table = table.groupby(columns, as_index=False).weight.sum().sort_values("weight", ascending=False).head(top_n)
            table.insert(0, "rank", np.arange(1, len(table) + 1)); table.insert(0, "niche", niche + 1)
            table[label] = table[columns].astype(str).agg(" → ".join, axis=1) if label == "sender_receiver" else table[columns].astype(str).agg("–".join, axis=1)
            rows.append(table)
        pd.concat(rows, ignore_index=True).to_csv(output / filename, index=False)
    return top
