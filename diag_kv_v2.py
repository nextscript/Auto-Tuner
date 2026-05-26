"""KV-cache diagnostic v2 — like diag_kv.py, but dumps EVERY metadata
key whose name contains "attention", "head", "kv", "embed", or "rope".

Use when diag_kv.py reports head_count_kv=0 — the value is sometimes
stored under a non-canonical key name by the quantizer (e.g.
``<arch>.attention.kv_head_count`` instead of the canonical
``<arch>.attention.head_count_kv``). This script prints the raw
key/value pairs so we can spot the alternate name.

Usage:
    python diag_kv_v2.py D:/models/gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scanner import read_gguf_metadata  # noqa: E402


# Substrings that flag potentially KV-relevant metadata keys.
INTEREST = ("attention", "head", "kv", "embed", "rope", ".n_", ".num_")


def main(argv) -> None:
    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(argv[1])
    if not path.is_file():
        print(f"[!] {path} not found")
        sys.exit(2)

    md = read_gguf_metadata(path)
    if not md:
        print(f"[!] could not read {path}")
        sys.exit(3)

    arch = md.get("general.architecture", "?")
    print(f"━━━ {path.name} ━━━")
    print(f"  architecture: {arch}")
    print()
    print("  ── ALL metadata keys matching attention/head/kv/embed/rope ──")

    # Sort so canonical keys appear first, alternates underneath.
    matches = []
    for k in md.keys():
        kl = k.lower()
        if any(s in kl for s in INTEREST):
            matches.append(k)
    matches.sort()

    if not matches:
        print("    (no matching keys — this GGUF has stripped metadata)")
        return

    longest = max(len(k) for k in matches)
    for k in matches:
        v = md[k]
        # Stringify; clip giant values (e.g. lists)
        sv = repr(v)
        if len(sv) > 80:
            sv = sv[:77] + "…"
        print(f"    {k:<{longest}}  =  {sv}")
    print()
    print(f"  Total: {len(matches)} matching keys, "
          f"{len(md)} keys overall in the GGUF")


if __name__ == "__main__":
    main(sys.argv)