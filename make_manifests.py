"""Create immutable validation, held-out, and stress scenario manifests."""

from __future__ import annotations

import argparse

from config import ROOT
from scenarios import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "manifests"))
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    from pathlib import Path

    output = Path(args.output).resolve()
    for split in ("train", "validation", "held_out", "structural_ood", "stress"):
        manifest = build_manifest(split, seed=args.seed)
        path = output / f"{split}.json"
        manifest.save(path)
        print(f"{split:10s} {len(manifest.scenarios):4d} scenarios  {manifest.digest()[:12]}  {path}")


if __name__ == "__main__":
    main()
