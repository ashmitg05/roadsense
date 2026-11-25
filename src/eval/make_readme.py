"""Fill README.md marker blocks from saved results files, so every number is traceable.

Markers:  <!-- BEGIN:key --> ... <!-- END:key -->
    experiments  <- results/experiments.md
    benchmark    <- results/benchmark_laptop.md
    errors       <- results/error_analysis_<best>.json (+ _wild, _fix if present)

    python -m src.eval.make_readme --best exp_xxx
"""

import argparse
import json
import re
from pathlib import Path


def errors_md(best: str) -> str:
    out = []
    for tag in (best, f"{best}_wild", "fix"):
        f = Path(f"results/error_analysis_{tag}.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        out.append(f"**{tag}** on `{d['list']}` (from `{f}`)\n")
        out.append("| Slice | Images | mAP@50 | mAP@50-95 |")
        out.append("|---|---|---|---|")
        out.append(
            f"| overall | {d['overall']['images']} | {d['overall']['map50']:.3f} | {d['overall']['map50_95']:.3f} |"
        )
        for key in ("by_lighting", "by_surface", "by_box_size"):
            for k, v in d[key].items():
                if k != "unknown":
                    out.append(f"| {key[3:]}: {k} | {v['images']} | {v['map50']:.3f} | {v['map50_95']:.3f} |")
        fps = d.get("false_positives_at_conf_0.25", {})
        fns = d.get("false_negatives_at_conf_0.25", {})
        out.append("")
        out.append(f"Top false positives @0.25: {', '.join(f'{k} ({v})' for k, v in list(fps.items())[:5])}")
        out.append(f"\nFalse negatives @0.25: {', '.join(f'{k} ({v})' for k, v in fns.items())}\n")
    return "\n".join(out) or "_not run yet_"


def main(best: str) -> None:
    blocks = {
        "experiments": Path("results/experiments.md"),
        "benchmark": Path("results/benchmark_laptop.md"),
    }
    readme = Path("README.md").read_text(encoding="utf-8")
    for key, f in blocks.items():
        body = f.read_text(encoding="utf-8") if f.exists() else "_not run yet_"
        readme = re.sub(
            rf"<!-- BEGIN:{key} -->.*?<!-- END:{key} -->",
            lambda _m, k=key, b=body: f"<!-- BEGIN:{k} -->\n{b.strip()}\n<!-- END:{k} -->",
            readme,
            flags=re.S,
        )
    readme = re.sub(
        r"<!-- BEGIN:errors -->.*?<!-- END:errors -->",
        lambda _m: f"<!-- BEGIN:errors -->\n{errors_md(best)}\n<!-- END:errors -->",
        readme,
        flags=re.S,
    )
    Path("README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    main(ap.parse_args().best)
