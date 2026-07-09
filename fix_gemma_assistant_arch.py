"""Hebt alte Gemma-4-Assistant-Drafter-GGUFs auf den Mainline-Stand (b9940+).

Alte Konvertierungen aus der PR-Ära (#23398) unterscheiden sich vom
gemergten llama.cpp-Stand in drei Punkten; alle drei werden repariert:

  1. ``general.architecture``: ``gemma4_assistant`` → ``gemma4-assistant``
     (llama.cpp registriert die Arch mit Bindestrich; sonst bricht der
     Server mit ``unknown model architecture: 'gemma4_assistant'`` ab).
     Auch alle arch-präfixierten Metadata-Keys werden mitumbenannt.
  2. Fehlende Metadata-Keys werden ergänzt:
       ``<arch>.nextn_predict_layers`` = block_count (alle Layer sind NextN;
         sonst GGML_ASSERT ``n_layer_nextn must be == n_layer_impl``)
       ``<arch>.embedding_length_out`` = alter Key ``<arch>.n_embd_backbone``
         (sonst ``requires embedding_length_out to carry the target hidden
         size``)
  3. Umbenannte Tensoren:
       ``mtp.pre_projection.weight``  → ``nextn.pre_projection.weight``
       ``mtp.post_projection.weight`` → ``nextn.post_projection.weight``
       ``mtp.centroids.weight``       → ``masked_embd_centroids.weight``
       ``mtp.token_ordering.weight``  → ``masked_embd_ordering``

Verifiziert am 2026-07-09 gegen b9940: Target + reparierter Drafter +
Vision (--mmproj) laufen zusammen, draft acceptance ~0.80.

Benötigt das ``gguf``-Python-Paket aus einem llama.cpp-Checkout (gguf-py)
sowie numpy. Der gguf-py-Pfad wird aus ``--gguf-py``, ``LLAMA_CPP_DIR``
oder dem in den AutoTuner-Settings gewählten Fork abgeleitet.

Aufruf:
    python fix_gemma_assistant_arch.py <datei-oder-ordner> [...]
    python fix_gemma_assistant_arch.py --gguf-py /pfad/zu/llama.cpp/gguf-py <...>

Das Original wird als ``<name>.gguf.bak`` gesichert; das reparierte GGUF
übernimmt den Originalnamen (damit die Drafter↔Target-Paarung des
Scanners erhalten bleibt).
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

OLD_ARCH = b"gemma4_assistant"
NEW_ARCH = b"gemma4-assistant"
assert len(OLD_ARCH) == len(NEW_ARCH)
ARCH = NEW_ARCH.decode()

# Obergrenze plausibler Vorkommen des Arch-Strings (Wert + arch-präfixierte
# Keys). Mehr deutet auf etwas Unerwartetes → Datei nicht anfassen.
MAX_HITS = 64

TENSOR_RENAMES = {
    "mtp.pre_projection.weight": "nextn.pre_projection.weight",
    "mtp.post_projection.weight": "nextn.post_projection.weight",
    "mtp.centroids.weight": "masked_embd_centroids.weight",
    "mtp.token_ordering.weight": "masked_embd_ordering",
}


def _find_gguf_py(explicit: str | None) -> Path | None:
    """gguf-py-Ordner finden: CLI-Arg → LLAMA_CPP_DIR → AutoTuner-Fork."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("LLAMA_CPP_DIR", "")
    if env:
        candidates.append(Path(env) / "gguf-py")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import app_settings

        fork = app_settings.get_fork_path()
        if fork:
            candidates.append(Path(fork) / "gguf-py")
        container = app_settings.get_fork_container_path()
        if container:
            for child in sorted(Path(container).iterdir()):
                candidates.append(child / "gguf-py")
    except Exception:
        pass
    for c in candidates:
        if (c / "gguf" / "__init__.py").is_file():
            return c
    return None


def upgrade_file(path: Path, gguf_mod) -> bool:
    """Eine Datei prüfen und ggf. vollständig reparieren. True = repariert."""
    gguf = gguf_mod

    # ── Stufe 1: Arch-String in-place (gleiche Länge, offset-neutral) ──
    head = path.open("rb").read(4)
    if head != b"GGUF":
        print(f"  übersprungen (kein GGUF): {path.name}")
        return False
    data = path.read_bytes()
    bak = path.with_suffix(path.suffix + ".bak")
    if OLD_ARCH in data:
        n = data.count(OLD_ARCH)
        if n > MAX_HITS:
            print(f"  ABBRUCH {path.name}: {n} Arch-Vorkommen (unerwartet)")
            return False
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_bytes(data.replace(OLD_ARCH, NEW_ARCH))
        print(f"  {path.name}: Arch-String repariert ({n} Vorkommen)")
    del data

    # ── Stufe 2: Keys/Tensoren prüfen und ggf. Datei neu schreiben ──
    reader = gguf.GGUFReader(path)

    def field_val(key: str):
        f = reader.get_field(key)
        return f.contents() if f else None

    if field_val("general.architecture") != ARCH:
        print(f"  übersprungen (arch != {ARCH}): {path.name}")
        return False

    have_nextn = field_val(f"{ARCH}.nextn_predict_layers") is not None
    have_out = field_val(f"{ARCH}.embedding_length_out") is not None
    needs_rename = any(t.name in TENSOR_RENAMES for t in reader.tensors)
    if have_nextn and have_out and not needs_rename:
        print(f"  OK (bereits aktuell): {path.name}")
        return False

    block_count = field_val(f"{ARCH}.block_count")
    backbone = field_val(f"{ARCH}.n_embd_backbone")
    if not have_nextn and not block_count:
        print(f"  ABBRUCH {path.name}: block_count fehlt")
        return False
    if not have_out and not backbone:
        print(f"  ABBRUCH {path.name}: n_embd_backbone fehlt")
        return False

    if not bak.exists():
        shutil.copy2(path, bak)

    tmp = path.with_name(path.stem + ".fixed.gguf")
    writer = gguf.GGUFWriter(str(tmp), ARCH, use_temp_file=False)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
    if not have_nextn:
        writer.add_key_value(
            f"{ARCH}.nextn_predict_layers", int(block_count), gguf.GGUFValueType.UINT32
        )
    if not have_out:
        writer.add_key_value(
            f"{ARCH}.embedding_length_out", int(backbone), gguf.GGUFValueType.UINT32
        )

    renamed = 0
    for tensor in reader.tensors:
        name = TENSOR_RENAMES.get(tensor.name, tensor.name)
        renamed += name != tensor.name
        writer.add_tensor_info(
            name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    # ALLE Memmap-Referenzen freigeben (Reader, KV-Feld- und Tensor-
    # Schleifenvariablen halten Views; GGUFReader enthält Zyklen → gc),
    # sonst schlägt os.replace auf der noch offenen Datei unter Windows fehl.
    reader = None
    tensor = None
    field = None
    gc.collect()

    os.replace(tmp, path)
    print(
        f"  OK {path.name}: Keys ergänzt (nextn={not have_nextn}, "
        f"embd_out={not have_out}), {renamed} Tensor(en) umbenannt "
        f"(Backup: {bak.name})"
    )
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="+", help="GGUF-Dateien oder Ordner")
    ap.add_argument("--gguf-py", help="Pfad zum gguf-py-Ordner eines llama.cpp-Checkouts")
    args = ap.parse_args(argv)

    gguf_py = _find_gguf_py(args.gguf_py)
    if gguf_py is None:
        print(
            "gguf-py nicht gefunden. --gguf-py /pfad/zu/llama.cpp/gguf-py "
            "angeben oder LLAMA_CPP_DIR setzen."
        )
        return 2
    sys.path.insert(0, str(gguf_py))
    try:
        # Zur Laufzeit aus dem llama.cpp-Checkout geladen (sys.path oben) —
        # für mypy existiert das Paket nicht.
        import gguf  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"gguf-Paket nicht ladbar ({exc}) — fehlt numpy? (pip install numpy)")
        return 2

    files: list[Path] = []
    for a in args.targets:
        p = Path(a)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.gguf")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"nicht gefunden: {a}")
    patched = sum(upgrade_file(p, gguf) for p in files)
    print(f"{patched} Datei(en) repariert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
