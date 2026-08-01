"""Combine split-worker FISTA and directional-graph audits across six cancers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CANCERS = ["LUAD", "LIHC", "KIRC", "COAD", "STAD", "HNSC"]
PRIMARY = {
    "LUAD": "GSE31210", "LIHC": "GSE14520", "KIRC": "GSE29609",
    "COAD": "GSE39582", "STAD": "GSE84437", "HNSC": "GSE65858",
}


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def weights(values: dict) -> str:
    return "/".join(f"{float(values.get(name, 0)):.3f}" for name in ("coexpr", "meth", "cnv"))


def old_result(root: Path, cancer: str) -> dict:
    return json.loads((root / f"final_config_comparison_{cancer}.json").read_text(encoding="utf-8"))[cancer]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fista-root", type=Path, required=True)
    parser.add_argument("--directional-root", type=Path, required=True)
    parser.add_argument("--topology-csv", type=Path, required=True)
    parser.add_argument("--lock-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fista_rows = []
    directional_rows = []
    topology = pd.read_csv(args.topology_csv).set_index("Cancer")
    for cancer in CANCERS:
        lock = old_result(args.lock_results, cancer)
        cohort = PRIMARY[cancer]
        old_c = float(lock["external"][cohort]["GR-SAFS_v2"]["c_index"])

        fista = json.loads(find_one(args.fista_root, f"{cancer}_FISTA_ROUTING_SENSITIVITY.json").read_text(encoding="utf-8"))
        new_c = float(fista["external"][cohort]["c_index"])
        fista_rows.append({
            "Cancer": cancer, "Primary cohort": cohort,
            "Lock route": fista["old_route_mode"], "FISTA route": fista["fista_route_mode"],
            "Lock coexpr/meth/cnv": weights(fista["old_weights"]),
            "FISTA coexpr/meth/cnv": weights(fista["fista_weights"]),
            "Top20 overlap": int(fista["top20_overlap_with_lock"]),
            "Top20 Jaccard": float(fista["top20_jaccard_with_lock"]),
            "Lock primary C-index": old_c, "FISTA primary C-index": new_c,
            "FISTA minus lock C-index": new_c - old_c,
            "Elapsed minutes": float(fista["elapsed_minutes"]),
        })

        directional = json.loads(find_one(args.directional_root, f"{cancer}_DIRECTIONAL_METHYLATION_ROUTING.json").read_text(encoding="utf-8"))
        directional_c = float(directional["external"][cohort]["c_index"])
        topo = topology.loc[cancer]
        directional_rows.append({
            "Cancer": cancer, "Primary cohort": cohort,
            "Lock density": float(topo["old density"]),
            "Directional-max density": float(topo["directional-max density"]),
            "Edge Jaccard": float(topo["edge_jaccard"]),
            "Lock coexpr/meth/cnv": weights(directional["old_weights"]),
            "Directional coexpr/meth/cnv": weights(directional["weights"]),
            "Directional route": directional["route_mode"],
            "Top20 overlap": int(directional["top20_overlap_with_original"]),
            "Top20 Jaccard": float(directional["top20_jaccard_with_original"]),
            "Lock primary C-index": old_c, "Directional primary C-index": directional_c,
            "Directional minus lock C-index": directional_c - old_c,
            "Elapsed minutes": float(directional["elapsed_minutes"]),
        })

    fista_frame = pd.DataFrame(fista_rows)
    directional_frame = pd.DataFrame(directional_rows)
    fista_frame.to_csv(args.output_dir / "CMPB_FISTA_ROUTING_SENSITIVITY_ALL6.csv", index=False)
    directional_frame.to_csv(args.output_dir / "CMPB_DIRECTIONAL_METHYLATION_ROUTING_ALL6.csv", index=False)
    summary = {
        "FISTA": {
            "minimum_top20_overlap": int(fista_frame["Top20 overlap"].min()),
            "maximum_absolute_primary_cindex_change": float(fista_frame["FISTA minus lock C-index"].abs().max()),
            "route_label_changes": int((fista_frame["Lock route"] != fista_frame["FISTA route"]).sum()),
        },
        "directional_methylation": {
            "minimum_top20_overlap": int(directional_frame["Top20 overlap"].min()),
            "maximum_absolute_primary_cindex_change": float(directional_frame["Directional minus lock C-index"].abs().max()),
        },
    }
    (args.output_dir / "CMPB_FINAL_SENSITIVITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
