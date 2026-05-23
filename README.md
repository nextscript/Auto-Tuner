# AutoTuner for llama.cpp

Interactive launcher for `llama-server` that **detects your hardware**,
**scans your local GGUF collection**, and **auto-tunes** context length,
KV-cache quantization, GPU offload, threading, and batch size to fit in
the RAM/VRAM you actually have free — without manual edits.

# GUI-Design

![GUI](image.png)

# Terminal-Design

```
────────────────────────────────────────────────────────────────
  AutoTuner for llama.cpp  —  interactive launcher
────────────────────────────────────────────────────────────────

============================================================
  DEBUG / VERBOSE MODE SELECTION
============================================================
  1. Debugging OFF (standard)
  2. Debugging ON (alle Kategorien)
------------------------------------------------------------
  Kategorie-Debugging (einzelne Bereiche):
  3. Hardware-Erkennung (GPU/RAM/CPU)
  4. Model-Scanning & Profil-Matching
  5. Server-Pfad-Suche (llama.cpp)
  6. Konfigurations-Berechnung (KV-Cache, Kontext)
------------------------------------------------------------
Wahl [1-6] (default 1):
[AutoTuner] Debugging deaktiviert.
============================================================


========================================
  QUANTIZATION MODE SELECTION
========================================
  1. Standard-Quant (llama.cpp)
  2. Turbo-Quant (tq_llama.cpp)
----------------------------------------
Select mode [1/2] (default 1):
[AutoTuner] Standard-Quant mode selected.
========================================

OS:   Windows 11
CPU:  Intel(R) Core(TM) Ultra 9 285K (24C/24T)
RAM:  47.4 GB total, 18.1 GB free
GPU1: [amd] AMD Radeon AI PRO R9700 (32.0 GB total, 31.0 GB free)
GPU2: [amd] AMD Radeon RX 9070 XT (15.9 GB total, 14.0 GB free)
      (ignored: [intel] Intel(R) Graphics, 2.0 GB — too small or auxiliary)

[AutoTuner] Scanning models in: C:\LAB\ai-local\models
[AutoTuner] Loaded 17 profile(s) from C:\GitHub\Auto Tuner\settings

Available models:
────────────────────────────────────────────────────────────────
  Symbols: 👁 vision · ⚡ draft · 🧠 thinking · 🛠 tool-use
  ...

  [Alibaba/Qwen3.6]
    7.  👁 🧠 🛠   Qwen3.6-27B-UD-Q3_K_XL                              13.5 GB  (256k native)
    8.  👁 🧠 🛠   Qwen3.6-35B-A3B-UD-IQ3_S                            12.7 GB  (256k native)
  ...

  [Google]
  ...
    16.  👁 ⚡ 🧠   gemma-4-26B-A4B-it-UD-IQ4_XS                        14.0 GB  (128k native)
    17.  👁         gemma-4-E2B-it-BF16                                 8.7 GB  (128k native)
  ...

  [IBM]
    21.   🛠         granite-4.1-30b-IQ4_XS                             14.4 GB  (128k native)
    22.   🛠         granite-4.1-3b-UD-Q8_K_XL                           4.0 GB  (128k native)
  ...

  [Mistral AI]
    36.  👁 🛠       Mistral-Medium-3.5-128B-UD-IQ3_XXS                 45.9 GB  (256k native)

  [NVIDIA]
    37.  👁 🧠 🛠   NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-…    18.2 GB  (1024k native)

  [PrismML]
    38.              Bonsai-8B                                           1.1 GB  (64k native)
  ...

Select a model [1-40, q to quit]: 16
Vision aktivieren? (mmproj-gemma-4-26B-A4B-it-BF16.gguf) [Y/n] y
Draft-Modell aktivieren? (gemma-4-26B-A4B-it-assistant-Q8_0) [Y/n] n
Thinking/Reasoning aktivieren? (<|think|> / <|reserved_special_token>) [Y/n] y
────────────────────────────────────────────────────────────────
Model:    gemma-4-26B-A4B-it-UD-IQ4_XS
Profile:  Gemma 4 (Google)  (gemma-4.yaml)
Notes:    Gemma ist empfindlich gegenüber repeat_penalty > 1.0. E2B/E4B = multimodal (Text+Bild+Audio), 26B-A4B + 31B = Text+Bild. Thinking-Modus aktivierbar durch <|think|> am Anfang des System-Prompts. Tipp: Manche Community-Tests zeigen, dass Gemma 4 für Coding sogar mit temp=1.5 besser performt - bei Bedarf mit `-- --temp 1.5` überschreiben.
Vision:   mmproj-gemma-4-26B-A4B-it-BF16.gguf
────────────────────────────────────────────────────────────────
  Placement       : GPU full offload (ngl=all of 60)
  Context         : 111,616 tokens
  KV cache quant  : K=q4_0  V=q4_0
  Threads         : 8 (batch: 16)
  Batch / ubatch  : 1024 / 512
  Flash attention : on
  Sampling        : temp=1.0 top_k=64 top_p=0.95 min_p=0.0 rep=1.0

  Memory estimate:
    model on GPU  ~  12.3 GB    (free VRAM:   15.1 GB)
    model on CPU  ~   0.0 GB    (free RAM:    18.1 GB)
    KV cache      ~  17.4 GB
────────────────────────────────────────────────────────────────
[AutoTuner] Found server binary: C:\LAB\ai-local\llama.cpp\build\bin\Release\llama-server.exe
Launch llama-server now? [Y/n]
```

## Features

- **Interactive terminal menu** — pick from whatever GGUFs are in your
  models folder, no editing required.
- **Hardware auto-detection** — works on **AMD (ROCm)**, **NVIDIA**,
  **Intel**, and **Apple Silicon** (unified memory). Multi-GPU is
  supported via automatic, **priority-weighted** `--tensor-split`: a
  model that fits the largest card is pinned to it (the second GPU stays
  free for gaming/OBS); only larger models spread across both. Device
  visibility is pinned via `HIP_VISIBLE_DEVICES` *and*
  `GGML_VK_VISIBLE_DEVICES` so it works on both ROCm and Vulkan builds.
- **Free-memory aware** — context length and KV quant are picked to
  use the RAM/VRAM that's actually free *right now*, not a hard-coded
  cap. The original v1 cap of 16k context is gone.
- **Per-family YAML profiles** in `settings/` — override sampling,
  max context, chat template, and llama-server flags per model family.
  Easy for contributors to extend without touching Python.
- **Companion-file auto-pairing** — sibling files don't pollute the
  model menu, they're attached to their main model:
  - `mmproj-*.gguf` → vision (longest-prefix wins)
  - `*-assistant-*.gguf` / `*-draft-*.gguf` → speculative decoding
      (smallest matching sibling wins)
- **Capability badges in the model list** — symbols make it obvious
  what each model can do at a glance:
  - 👁 vision (mmproj projector paired)
  - ⚡ draft  (assistant sibling for speculative decoding)
  - 🧠 thinking (chat template emits `<think>` / `reasoning_content`)
  - 🛠 tool-use (chat template advertises `tool_calls` / `function_call`)

  Detection reads the GGUF chat template directly — no name-based
  guessing — so `Qwen3-Coder` (no thinking) and `Qwen3-Embedding`
  (neither thinking nor tools) are correctly excluded.
- **Reads GGUF metadata** — pulls `n_layers` and `context_length`
  straight from the file so partial GPU offload (`-ngl`) is exact.
- **Sticky GUI choices** — the Qt launcher remembers per-model
  vision/draft/thinking toggles in `autotuner_settings.json`. Switch
  to another model and back, restart the app, change the performance
  target — your manual choices stay put. They only revert when you
  click them again.
- **Fork-folder memory** — if you point the GUI at a parent folder
  that holds several `*_llama.cpp` builds (e.g. `C:\LAB\ai-local`),
  the next launch re-expands the same set of builds in the dropdown.
  No more re-navigating one folder up after every restart.
- **Window geometry & state** — QMainWindow `saveGeometry()` (base64
  in JSON) und `saveState()` (Toolbars, Dock-Positionen) werden
  persistiert. Fenstergröße, -position, Maximize-State und
  Toolbar-Status bleiben über Neustarts erhalten.
- **Globale Schriftgröße** — peristente Schriftgröße (Clamp 7..22),
  wird beim App-Start sofort angewendet (kein Flash von Default).
- **Reasoning-Effort** — pro-Modell wählbar: `auto` / `off` /
  `minimal` / `low` / `medium` / `high` / `extra_high`. Think-Budget
  (Spin-Box, -1 = aus, 0 = sofort stop, N = Token-Budget) im
  Expert-Panel.

### Vision control

You can disable vision (mmproj) support in two ways:

1. **Command-line flag**:

   ```bash
   python auto_tuner.py --model "Qwen3.6" --novision
   ```

## Installation

```bash
git clone https://github.com/<you>/llama-cpp-auto-tuner
cd llama-cpp-auto-tuner
pip install -r requirements.txt
```

You also need a working `llama-server` binary. The tuner automatically discovers binaries in common local setups (like `C:\LAB\ai-local\`), or you can specify one via `--server`.

## Usage

Point it at a folder of `*.gguf` models — it will recurse:

```bash
python auto_tuner.py --models-path /path/to/models
```

Or set the environment variable once:

```bash
export AUTOTUNER_MODELS=/path/to/models     # Linux / macOS
setx  AUTOTUNER_MODELS  D:\models           # Windows
python auto_tuner.py
```

Pick a model from the menu. Once it's running, point your client at:

```
http://127.0.0.1:1234
```

Works with the built-in **llama.cpp Web UI**, **VS Code** extensions
like Continue / Cline, **Open WebUI**, or any OpenAI-API client.

### Qt GUI

```bash
python qt_launcher.py
```

Same engine as the terminal launcher, plus a few quality-of-life bits
that only make sense with persistent state:

- **Sticky per-model options.** Toggle vision / draft / thinking once;
  the choice survives switching to another model and back, swapping
  performance targets, and restarting the app. Stored in
  `autotuner_settings.json` under `model_overrides`.
- **Fork picker remembers the parent folder.** Hit *📂 Fork* and
  pick a directory that holds multiple `*_llama.cpp` builds — every
  build appears in the dropdown next time too, not just the last one
  you used. The active build within that container is also restored.
- **Live config preview.** The right pane recomputes
  context / KV / placement whenever you tick a checkbox or change the
  performance target — no need to launch first.
- **Honest load status (`/health` handshake).** After launch the status
  bar shows *Loading model* and only flips to *Ready* once the server's
  `GET /health` returns 200. Big MoE models can take a while to load (or
  fail mid graph-build) — the GUI no longer claims "Running" the instant
  the PID exists. A crash during load is surfaced as *Server exited*.
- **Window geometry persistence.** Fenstergöße, -position,
  Maximize-State und Toolbar-Status werden gespeichert und beim
  nächsten Start wiederhergestellt (`_restore_window_geometry`).
- **Font persistence.** Globale QApplication-Schriftgröße wird
  persistiert und beim Start sofort angewendet (`_change_font`).
  Kein Flash von der Default-Schriftgröße mehr.
- **Reasoning-Panel (Expert-Panel).** Neue Sektion mit:
  - Dropdown "Effort": `auto` / `off` / `minimal` / `low` /
      `medium` / `high` / `extra_high`
  - SpinBox "Think budget": `-1` = aus, `0` = sofort stop, `N` =
      Token-Budget
  Die Werte werden als `--reasoning`, `--think-budget` und
  `--chat-template-kwargs` in `cfg.extra_cli_flags` übersetzt.

### Useful flags

| Flag | Description |
|---|---|
| `--models-path PATH` | Folder to scan (default `./models`, env `AUTOTUNER_MODELS`) |
| `--settings-path PATH` | Folder with YAML profiles (default `./settings`) |
| `--server PATH` | Path to `llama-server` (default looks on `$PATH`, env `LLAMA_SERVER`) |
| `--host HOST` | Bind address (default `127.0.0.1`) |
| `--port N` | Server port (default `1234`) |
| `--ctx N` | Override the auto-tuned context length |
| `--model SUBSTR` | Skip the menu, pick a model by name substring |
| `--dry-run` | Print the command, don't start the server |
| `--yes / -y` | Skip the launch confirmation prompt |
| `--force-mlock` | Force `--mlock` / `--no-mmap` (prevents VRAM/RAM paging) |
| `--performance-target {safe,balanced,throughput}` | VRAM utilisation preset (see below) |
| `-- <args...>` | Anything after `--` is forwarded to `llama-server` |

### Performance targets (`--performance-target`)

A single switch that controls how aggressively the AutoTuner reserves
VRAM. It changes both the safety bands and the KV-cache budget that
gets reserved up front during MoE layer placement, so picking the right
tier can move several expert layers between GPU and CPU.

| Tier | KV reservation | VRAM safety | When to use |
|---|---|---|---|
| `safe` | 128 k tokens | 0.30 GB | Long-context sessions (>64 k), maximum stability |
| `balanced` *(default)* | 64 k tokens | 0.25 GB | General use — moderate optimisation that helps everyone |
| `throughput` | 32 k tokens | 0.15 GB | Short-context inference (chat, reasoning ≤32 k); pushes more expert layers onto the GPU for higher tokens/s |

**Resolution priority** (highest wins): explicit CLI flag → GUI dropdown
→ `performance_target:` in the model's YAML profile → `balanced` default.
Unknown values are silently ignored, so a typo in a YAML never breaks
anything.

A profile can declare its preferred tier in YAML:

```yaml
# settings/qwen3_5-3_6.yaml
performance_target: throughput   # MoE — wants every spare GB on the GPU
```

The user choice (CLI / GUI) always wins over the profile recommendation.

### Memory locking (`--mlock` / `--no-mmap`)

The auto-tuner automatically decides whether to enable `--mlock` and `--no-mmap`
based on available system resources. These flags pin model data in physical
memory (RAM/VRAM) and prevent the OS from paging it to disk, which is critical
for stable inference performance.

**Automatic behavior:**

| Scenario | Condition | Result |
|---|---|---|
| **Full GPU offload** | `total_vram > 8 GB` AND `free_vram > model_size + 2 GB` | `--mlock --no-mmap` enabled |
| **Partial / CPU offload** | `total_ram > 32 GB` AND `free_ram > model_ram_on_cpu + 8 GB` | `--mlock --no-mmap` enabled |
| **Insufficient memory** | Safety reserve not met | Disabled (fallback to default mmap) |

**Force memory locking:**

Use `--force-mlock` to override the automatic decision and always enable
memory locking when the OS permits it:

```bash
python auto_tuner.py --force-mlock
```

This is useful when you know your system has enough memory but the tuner's
conservative thresholds would otherwise skip it.

**Debug output:**

The tuner prints the mlock decision before every launch:

```
  [mlock] decision: model=Qwen3.6-35B-A3B-UD-Q6_K
         full_offload=True  vram=18.5GB  ram=0.0GB
         sys: total_vram=24.0GB  free_vram=5.2GB  total_ram=32.0GB  free_ram=12.1GB
         force_mlock=False  -> mlock=True  no_mmap=True
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AUTOTUNER_MODELS` | `./models` | Where to scan for `*.gguf` files |
| `LLAMA_SERVER` | `llama-server` | Path or name of the server binary |
| `LLAMA_CPP_DIR` | (auto-detected) | Your llama.cpp checkout. If set, the auto-tuner will look for `build/bin/[Release/]llama-server[.exe]` inside it. |

### Server binary auto-discovery

The tuner automatically searches for binaries in common local layouts.
If you have a workspace like this, it "Just Works" without any flags:

```
C:\GitHub\
└── Auto Tuner\         ← clone of this repo
C:\LAB\
└── ai-local\
    ├── llama.cpp\      ← standard build
    ├── tq_llama.cpp\   ← Turbo-Quant build
    ├── ik_llama.cpp\   ← Gemma 4 externer Drafter (Fork noch nötig)
    ├── 1b_llama.cpp\   ← BitNet fork (Ternary-Bonsai)
    └── models\         ← your models
```

It looks for `llama-server` inside these directories (including `build/bin/...` subpaths).

#### Quantization Modes

When you start the tuner, you can choose between:

1. **Standard-Quant**: Uses standard `llama.cpp` binaries.
2. **Turbo-Quant**: Uses the `tq_llama.cpp` binary for faster inference.

#### Turbo-Quant Labels & KV-Quant-Options

`kv_quant_factor()` unterstützt jetzt folgende Turbo-Quant-Labels:
`turbo2`, `turbo3`, `turbo4`, `iq4_nl`, `tq3_0`, `turbo3_tcq`.

`_TURBO_QUANT_MAP` korrigiert die echten Labels:

| Label   | Turbo-Quant | Faktor (vs F16) |
|---------|-------------|-----------------|
| `q8_0`  | `turbo4`    | ~3.8x           |
| `q5_0`  | `turbo3`    | ~4.3x           |
| `q4_0`  | `turbo3`    | ~4.3x           |

(Vorher mappte es fälschlich auf `q4_1`/`q5_1`, was Mainline-Labels
sind, NICHT TurboQuant.)

`_pick_kv_quant` rechnet das Budget jetzt mit Turbo-Faktoren — beim
Umschalten zeigt sich die echte Token-Zahl-Erhöhung (gemessen: 48k →
63k bei Qwen3.6-35B-A3B).

Die KV-Dropdowns in der GUI zeigen die volle Auswahl: `iq4_nl`,
`q4_1`, `q5_1`, `turbo2`, `turbo3`, `turbo4`.

#### Specialized Binary Logic

The tuner intelligently selects the best binary based on your model and settings:

- **Gemma 4 (with external draft)** $\rightarrow$ uses `ik_llama.cpp` (external sibling drafter still requires the fork).
- **Gemma 4 (without draft)** $\rightarrow$ uses standard `llama.cpp`.
- **Integriertes MTP (z.B. Qwen3.6-27B-MTP)** $\rightarrow$ uses standard `llama.cpp` (b9190+ nativ; PR #22673 seit 16. Mai 2026 in Mainline; kein Fork nötig).
- **Ternary-Bonsai** $\rightarrow$ uses `1b_llama.cpp`.
- **Turbo-Quant Mode** $\rightarrow$ uses `tq_llama.cpp`.

Example — run Devstral, override context, and pass an extra flag
(`--metrics` is now added automatically, so this just shows pass-through):

```bash
python auto_tuner.py --model Devstral --ctx 131072 -y -- --verbose
```

## Adding profiles for new models

Drop a new YAML file into `settings/`. The filename doesn't matter;
the `patterns:` list does. The longest pattern that appears as a
substring of the model filename wins.

```yaml
# settings/my-model.yaml
display_name: "My Model"
patterns:
  - my-model
  - my-model-base

max_context: 131072
recommended_kv_quant: q8_0

sampling:
  temperature: 0.7
  top_k: 40
  top_p: 0.9
  min_p: 0.05
  repeat_penalty: 1.05

# Optional:
chat_template: chatml
extra_args:
  - --no-context-shift
notes: >
  Anything you want to remind yourself about this model.
```

Profiles with empty `patterns:` become the fallback when nothing else
matches. See `settings/_default.yaml`.

## How the auto-tuning works

1. **Detect**: total / free RAM, every GPU's total / free VRAM, total
   CPU cores.
2. **Place the model**: full GPU offload if it fits, else partial
   offload using the GGUF's exact `n_layers`, else CPU only.
3. **Compute the KV budget**: free VRAM (after the model) plus free
   RAM (minus a safety reserve).
4. **Pick KV quant + context**: try q8 → q5 → q4, pick the highest
   quality that fits the profile's `max_context`. Round context down
   to a multiple of 1024.
5. **Threads / batch**: scale with placement (full GPU offload needs
   fewer CPU threads than CPU-only inference; long context wants
   smaller batches to keep prompt-prefill memory bounded).
6. **Multi-GPU**: a model that fits the largest card alone is pinned to
   it (other GPUs hidden via the visibility env vars, so they stay free
   for gaming/OBS); larger models spread across all GPUs with a
   **priority-weighted** `--tensor-split` (priority × free VRAM), and the
   highest-scoring card becomes `--main-gpu`.
7. **Hand authority to the AutoTuner**: `--fit off` is always emitted so
   llama.cpp's own auto-fit pass never silently re-tunes the values the
   AutoTuner computed and logged. An overcommit fails loudly (OOM) instead
   of being quietly downscaled.

## Project layout

```
auto_tuner/
├── auto_tuner.py        # main entry: terminal menu + glue
├── qt_launcher.py       # Qt GUI (model picker + sticky options + fork picker)
├── hardware.py          # CPU + multi-vendor GPU detection
├── scanner.py           # GGUF scanner: mmproj/draft pairing, capability detection
├── settings_loader.py   # YAML profile loader and matcher
├── tuner.py             # config calculation + llama-server command builder
├── launcher.py          # subprocess + Ctrl+C handling (Windows + Unix)
├── app_settings.py      # persistent GUI prefs (autotuner_settings.json)
...
├── settings/
│   ├── _default.yaml
│   ...
│   ├── ministral.yaml
│   ├── bonsai.yaml
│   ...
├── requirements.txt
└── README.md
```

## Building llama.cpp and forks

Recommended build settings for this system:

- Ninja generator
- native CPU optimizations
- static build
- Release mode
- 20 parallel build jobs

```
## Building llama.cpp and forks - Example

# Example-System:
# - Intel Core Ultra 9 285K
# - AMD Radeon RX 9070 XT 16GB
# - AMD Radeon R9700 AI Pro 32GB
# - G.Skill Trident Z 48GB DDR5-8400MHz (2x24GB)

# Main-Fork b9208+ (SPIRV-Headers required since b9194) - Windows
cd H:\LAB\ai-local
git clone https://github.com/KhronosGroup/SPIRV-Headers.git
cmake -S .\SPIRV-Headers -B .\SPIRV-Headers\build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DCMAKE_INSTALL_PREFIX="H:/LAB/ai-local/SPIRV-Headers/install"
cmake --build .\SPIRV-Headers\build --config Release
cmake --install .\SPIRV-Headers\build --config Release
git clone https://github.com/ggml-org/llama.cpp.git
Push-Location .\llama.cpp\tools\ui
npm ci
npm run build
Pop-Location
cmake -S .\llama.cpp -B .\llama.cpp\build `
  -G "Visual Studio 18 2026" `
  -A x64 `
  -DGGML_VULKAN=ON `
  -DGGML_NATIVE=ON `
  -DBUILD_SHARED_LIBS=OFF `
  -DLLAMA_BUILD_SERVER=ON `
  -DLLAMA_BUILD_UI=ON `
  -DLLAMA_USE_PREBUILT_UI=OFF `
  -DLLAMA_CURL=OFF `
  -DGGML_CCACHE=OFF `
  -DGGML_VULKAN_CHECK_RESULTS=OFF `
  -DCMAKE_PREFIX_PATH="H:/LAB/ai-local/SPIRV-Headers/install"
cmake --build .\llama.cpp\build --config Release --parallel 24

# Main-Fork - Ubuntu
git clone https://github.com/ggerganov/llama.cpp llama.cpp
cd ~/llama.cpp
rm -rf build
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1200;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)```
```

## Server-Features (Stand b9297)

Die folgenden `llama-server` Features werden unterstützt (aus `tools/server/README.md`):

| Flag | Unterstützung |
|------|---------------|
| `-fa [on\|off\|auto]` | ✅ Form `-fa on` wird emittiert |
| `-ctk/-ctv f16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl` | ✅ Alle im Dropdown |
| `--fit off` | ✅ **Neu** — immer emittiert, damit llama.cpps eigener Auto-Fit-Pass (Default `on`) die berechneten Werte nicht still nachjustiert (AutoTuner ist die Autorität) |
| `--metrics` | ✅ **Neu** — Prometheus-Endpoint `GET /metrics` auf demselben host:port (siehe „Monitoring") |
| `--reasoning on/off/auto` | ✅ Via Dropdown |
| `--think-budget N` | ✅ Via SpinBox |
| `--chat-template-kwargs ...` | ✅ Dropdown produziert das automatisch |
| `--jinja` | ✅ Wird sichtbar angehakt |
| `--mlock` / `--no-mmap` | ✅ Windows-Guard; manuell überschreibbar |
| `-md` externer Drafter | ✅ Ohne `--spec-type` — die Anwesenheit von `-md` aktiviert den Draft-Pfad in Mainline automatisch (verifiziert b9297) |
| `--spec-type draft-mtp` | ✅ Integriertes MTP (Qwen3.6-MTP u.a.) — `draft-mtp` ist der Mainline-Name seit Merge von PR #22673 (16. Mai 2026) |
| `--spec-draft-n-max` | ✅ Via `draft_max` im YAML-Profil |
| `--spec-draft-p-min` | ✅ Via `draft_p_min` im YAML-Profil — Default 0.75; wird in **beiden** Spec-Paths (extern + integriert) emittiert |
| `--spec-draft-ngl` | ✅ Immer 99 (MTP-Head auf GPU halten) |
| `--n-cpu-moe` / `--override-tensor` | ✅ `--n-cpu-moe` aktiv; `-ot` für gezielte Expert-Platzierung vorbereitet |
| `--tensor-split` / `--main-gpu` | ✅ Priority-weighted, mit Single-GPU-Pinning |
| `--rope-scaling yarn` | ✅ Bereits vorhanden |
| `--numa` | ✅ Bereits vorhanden |
| `--no-context-shift` | ✅ Wird nicht mehr dupliziert (Dedup via seen-Set) |

### Monitoring (`/health` + `/metrics`)

Beim Start hängt der AutoTuner `--metrics` an, sodass `llama-server` zwei
HTTP-Endpoints auf demselben `host:port` wie die Inferenz-API bereitstellt
(es gibt **keinen** separaten Metrics-Port):

- **`GET /health`** — `503` während des Ladens, `200` wenn das Modell
  bereit ist. Die Qt-GUI pollt diesen Endpoint und schaltet den Status
  von *Loading model* auf *Ready* (siehe oben).
- **`GET /metrics`** — Prometheus-Textformat. Die wichtigsten Kennzahlen
  (Single-Model-Modus, Prefix `llamacpp:`):

  | Metrik | Typ | Bedeutung |
  |---|---|---|
  | `llamacpp:predicted_tokens_seconds` | gauge | Generierungs-Durchsatz (tok/s) |
  | `llamacpp:prompt_tokens_seconds` | gauge | Prompt-/Prefill-Durchsatz (tok/s) |
  | `llamacpp:kv_cache_usage_ratio` | gauge | KV-Cache-Füllstand (1.0 = 100 %) |
  | `llamacpp:kv_cache_tokens` | gauge | Tokens im KV-Cache |
  | `llamacpp:requests_processing` | gauge | Aktive Requests |
  | `llamacpp:tokens_predicted_total` | counter | Generierte Tokens kumuliert |
  | `llamacpp:prompt_tokens_total` | counter | Prompt-Tokens kumuliert |

  Scrapen ohne Prometheus-Client (z. B. für den System Tricorder):

  ```python
  import urllib.request
  def llama_metrics(base_url: str) -> dict[str, float]:
      out = {}
      with urllib.request.urlopen(f"{base_url}/metrics", timeout=0.5) as r:
          for line in r.read().decode().splitlines():
              if line and not line.startswith("#"):
                  name, _, val = line.partition(" ")
                  try: out[name] = float(val)
                  except ValueError: pass
      return out
  # llama_metrics("http://127.0.0.1:1234")["llamacpp:predicted_tokens_seconds"]
  ```

## License

MIT.
