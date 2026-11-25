"""Build the experiments markdown table from results/*.json (training runs only).

python -m src.eval.experiments_table
"""

import json
from pathlib import Path

from src.data.convert import NAMES

ORDER = ["baseline", "exp_v8s", "exp_960", "exp_aug", "exp_rfs", "exp_multi", "fix"]
SHORT = {"longitudinal_crack": "D00", "transverse_crack": "D10", "alligator_crack": "D20", "pothole": "D40"}


def main() -> str:
    runs = {}
    for f in Path("results").glob("*.json"):
        d = json.loads(f.read_text())
        if isinstance(d, dict) and "test" in d and "config" in d:
            runs[d["name"]] = d
    names = sorted(runs, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))
    head = (
        "| Run | Change | mAP@50 | mAP@50-95 | "
        + " | ".join(f"AP50 {SHORT[n]}" for n in NAMES)
        + " | Train min |"
    )
    lines = [head, "|" + "---|" * (5 + len(NAMES))]
    for n in names:
        d = runs[n]
        t = d["test"]
        comment = (
            Path(f"configs/train/{n}.yaml").read_text().splitlines()[0].lstrip("# ")
            if Path(f"configs/train/{n}.yaml").exists()
            else ""
        )
        pcs = " | ".join(f"{t['per_class'].get(c, {}).get('ap50', float('nan')):.3f}" for c in NAMES)
        lines.append(
            f"| {n} | {comment} | {t['map50']:.3f} | {t['map50_95']:.3f} | {pcs} | {d['train_minutes']} |"
        )
    md = "\n".join(lines) + "\n"
    Path("results/experiments.md").write_text(md)
    print(md)
    return md


if __name__ == "__main__":
    main()
