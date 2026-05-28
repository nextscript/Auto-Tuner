# AutoTuner for llama.cpp

Interactive launcher for `llama-server` that **detects your hardware**,
**scans your local GGUF collection**, and **auto-tunes** context length,
KV-cache quantization, GPU offload, threading, and batch size to fit the
RAM/VRAM you actually have free — without manual edits.

Tested against llama.cpp **b9381** (Vulkan/ROCm, Windows + Linux).

![GUI](image.png)

## Features

- **Two front-ends, one engine** — an interactive terminal menu
  (`auto_tuner.py`) and a Qt GUI (`qt_launcher.py`) that share the same
  detection and tuning logic.
- **Run several models at once** — launch multiple `llama-server`
  instances concurrently, each on its own port (see
  [Running multiple models](#running-multiple-models-at-once)).
- **Hardware auto-detection** — AMD (ROCm/Vulkan), NVIDIA, Intel, and
  Apple Silicon (unified memory). Multi-GPU uses a priority-weighted
  `--tensor-split`: a model that fits the largest card is pinned to it
  (the second GPU stays free for gaming/OBS); only larger models spread
  across both. Device visibility is pinned via `HIP_VISIBLE_DEVICES`
  *and* `GGML_VK_VISIBLE_DEVICES` so it works on ROCm and Vulkan builds.
- **Free-memory aware** — context length and KV quant are picked to use
  the RAM/VRAM that's free *right now*, not a hard-coded cap.
- **Per-family YAML profiles** in `settings/` — override sampling, max
  context, chat template, and llama-server flags per model family,
  without touching Python.
- **Companion-file auto-pairing** — sibling files don't clutter the menu;
  they attach to their main model:
  - `mmproj-*.gguf` → vision (longest-prefix match wins)
  - `*-assistant-*.gguf` / `*-draft-*.gguf` → speculative decoding
    (smallest matching sibling wins)
- **Capability badges** in the model list, read straight from the GGUF
  chat template (no name guessing):
  - 👁 vision · ⚡ draft · 🧠 thinking · 🛠 tool-use
- **Reads GGUF metadata** — pulls exact `n_layers` and `context_length`
  from the file, so partial GPU offload (`-ngl`) is precise.
- **Sticky GUI choices** — per-model vision/draft/thinking toggles,
  performance target, fork selection, window geometry, and font size all
  persist in `autotuner_settings.json` across restarts.

## Installation

```bash
git clone https://github.com/<you>/llama-cpp-auto-tuner
cd llama-cpp-auto-tuner
pip install -r requirements.txt
```

You also need a working `llama-server` binary. The tuner auto-discovers
binaries in common local layouts (see
[Server binary discovery](#server-binary-discovery)), or pass one via
`--server`.

## Usage

### Terminal

Point it at a folder of `*.gguf` models (it recurses):

```bash
python auto_tuner.py --models-path /path/to/models
```

Or set the folder once via the environment:

```bash
export AUTOTUNER_MODELS=/path/to/models     # Linux / macOS
setx  AUTOTUNER_MODELS  D:\models           # Windows
python auto_tuner.py
```

Pick a model from the menu; once it's running, point any OpenAI-API
client at `http://127.0.0.1:1234` — the built-in llama.cpp Web UI, VS Code
extensions (Continue / Cline / RooCode), Open WebUI, etc.

A representative config summary printed before launch:

```
────────────────────────────────────────────────────────────────
Model:    gemma-4-26B-A4B-it-UD-IQ4_XS
Profile:  Gemma 4 (Google)  (gemma-4.yaml)
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
    KV cache      ~  17.4 GB
────────────────────────────────────────────────────────────────
```

### Qt GUI

```bash
python qt_launcher.py
```

Same engine plus quality-of-life features that rely on persistent state:

- **Sticky per-model options** — toggle vision/draft/thinking once; the
  choice survives model switches, target changes, and app restarts.
- **Fork picker remembers the parent folder** — point *📂 Fork* at a
  directory holding several `*_llama.cpp` builds and they all reappear in
  the dropdown next time.
- **Live config preview** — the right pane recomputes context / KV /
  placement whenever you toggle an option.
- **Honest load status** — after launch the status bar shows *Loading*
  and only flips to *Ready* once the server's `GET /health` returns 200.
  A crash during load surfaces as *exited*.
- **Reasoning panel** — effort dropdown (`auto`/`off`/`minimal`/`low`/
  `medium`/`high`/`extra_high`) and a think-budget spin-box, translated
  into `--reasoning`, `--think-budget`, and `--chat-template-kwargs`.

### Running multiple models at once

AutoTuner can keep several `llama-server` instances alive concurrently —
one per port, each in its own console window. This is what enables
multi-agent setups (e.g. a coding agent that spawns subagents locally):
an orchestrator model plus one or more subagent/draft models all serving
at the same time.

**In the GUI:** just click **▶ Launch** again for the next model. Each
launch automatically picks the next free port (skipping ports already
used by a tracked instance or any other process) and advances the *Port*
field, so back-to-back launches never collide. Every live instance shows
up in the **Running** dropdown as `:port  model  [loading/ready]  PID`.
Stop a single one with **■ Stop**, or all of them with **■ Stop all**.
The status bar reports `N server(s) running — M ready`.

**From the command line** use `--detach`: it starts the server in its own
session/console and returns immediately instead of waiting, so a script
or agent can bring up several models in a row (give each a different
`--port`):

```bash
python auto_tuner.py --model Orchestrator --port 1234 --detach
python auto_tuner.py --model Subagent     --port 1235 --detach
```

Each detached server keeps running in its own window after the command
returns. `--detach` implies `--yes` (non-interactive).

### Command-line flags

| Flag | Description |
|---|---|
| `--models-path PATH` | Folder to scan (default `./models`, env `AUTOTUNER_MODELS`) |
| `--settings-path PATH` | Folder with YAML profiles (default `./settings`) |
| `--server PATH` | Path to `llama-server` (default `$PATH`, env `LLAMA_SERVER`) |
| `--host HOST` | Bind address (default `127.0.0.1`) |
| `--port N` | Server port (default `1234`) |
| `--ctx N` | Override the auto-tuned context length |
| `--model SUBSTR` | Skip the menu, pick a model by name substring |
| `--detach` | Spawn detached and return immediately (for multi-model / agent use); implies `--yes` |
| `--dry-run` | Print the command, don't start the server |
| `--yes` / `-y` | Skip the launch confirmation prompt |
| `--novision` | Disable vision (mmproj) even if available |
| `--nodraft` | Disable speculative decoding / draft model |
| `--ngram` | Enable draftless n-gram self-speculation |
| `--force-mlock` | Force `--mlock` / `--no-mmap` even on full GPU offload |
| `--performance-target {safe,balanced,throughput}` | VRAM utilisation preset (see below) |
| `--gui` | Open the Qt log-viewer window after the server starts |
| `-- <args...>` | Anything after `--` is forwarded verbatim to `llama-server` |

## Tuning controls

### Performance targets

A single switch controlling how aggressively VRAM is reserved. It changes
both the safety bands and the KV budget reserved during MoE layer
placement, so the tier can move several expert layers between GPU and CPU.

| Tier | KV reservation | VRAM safety | When to use |
|---|---|---|---|
| `safe` | 128k tokens | 0.30 GB | Long-context (>64k), maximum stability |
| `balanced` *(default)* | 64k tokens | 0.25 GB | General use |
| `throughput` | 32k tokens | 0.15 GB | Short-context chat/reasoning (≤32k); more expert layers on GPU for higher tok/s |

**Resolution priority** (highest wins): CLI flag → GUI dropdown →
`performance_target:` in the model's YAML → `balanced`. Unknown values are
ignored, so a typo never breaks anything.

### Memory locking (`--mlock` / `--no-mmap`)

These pin model data in physical memory and stop the OS paging it to
disk. The tuner enables them automatically when there's enough headroom:

| Scenario | Condition | Result |
|---|---|---|
| Full GPU offload | `total_vram > 8 GB` and `free_vram > model + 2 GB` | enabled |
| Partial / CPU offload | `total_ram > 32 GB` and `free_ram > model_on_cpu + 8 GB` | enabled |
| Insufficient memory | safety reserve not met | disabled (default mmap) |

Use `--force-mlock` to override the conservative thresholds. On **Windows**
both flags are disabled by default unless forced, because `VirtualLock`
needs the `SeLockMemoryPrivilege` that isn't granted automatically (even to
Administrators). The decision is printed before every launch.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AUTOTUNER_MODELS` | `./models` | Where to scan for `*.gguf` files |
| `LLAMA_SERVER` | `llama-server` | Path or name of the server binary |
| `LLAMA_CPP_DIR` | (auto-detected) | llama.cpp checkout; the tuner looks for `build/bin/[Release/]llama-server[.exe]` inside it |

> Note: since b9371, llama.cpp renamed several of its own env vars to the
> `LLAMA_ARG_` prefix (e.g. `LLAMA_LOG_*` → `LLAMA_ARG_LOG_*`,
> `LLAMA_OFFLINE` → `LLAMA_ARG_OFFLINE`). The **CLI flags are unchanged**,
> and AutoTuner only sets the GGML visibility vars, so this has no effect —
> but use the `LLAMA_ARG_` prefix if you add llama log/offline overrides.

## Binaries and profiles

### Server binary discovery

The tuner searches common local layouts, so a workspace like this "just
works" without flags:

```
…\ai-local\
  ├── llama.cpp\       ← standard build
  ├── tq_llama.cpp\    ← Turbo-Quant build
  ├── ik_llama.cpp\    ← Gemma 4 external drafter (fork still required)
  └── 1b_llama.cpp\    ← BitNet fork (Ternary Bonsai)
```

It looks for `llama-server` inside these directories (including
`build/bin/...` subpaths). Binary selection per model:

- **Gemma 4 with external draft** → `ik_llama.cpp` (the external sibling
  drafter still needs the fork)
- **Gemma 4 without draft** / **integrated MTP** (e.g. Qwen3.6-MTP) →
  standard `llama.cpp` (MTP is native in mainline b9190+)
- **Ternary Bonsai** → `1b_llama.cpp`
- **Turbo-Quant mode** → `tq_llama.cpp`

### Adding a profile for a new model

Drop a YAML file into `settings/`. The filename doesn't matter; the
`patterns:` list does — the longest pattern that is a substring of the
model filename wins. Empty `patterns:` makes a profile the fallback
(see `settings/_default.yaml`).

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

# Optional
chat_template: chatml
performance_target: balanced
extra_args:
  - --no-context-shift
notes: >
  Anything you want to remember about this model.
```

## How the auto-tuning works

1. **Detect** total/free RAM, every GPU's total/free VRAM, CPU cores.
2. **Place the model**: full GPU offload if it fits, else partial offload
   using the GGUF's exact `n_layers`, else CPU only.
3. **Compute the KV budget**: free VRAM after the model, plus free RAM
   minus a safety reserve. For MoE models the budget is **VRAM-only**
   (including free RAM crashes Vulkan with `GGML_ASSERT(addr)`).
4. **Pick KV quant + context**: try q8 → q5 → q4, highest quality that
   fits the profile's `max_context`; context rounded down to a 1024
   multiple. KV is reserved for a realistic working context (~32k), not
   the profile maximum, to avoid overcommitment.
5. **Threads / batch** scale with placement and context length.
6. **MoE** layers spill to CPU automatically via `--n-cpu-moe`; no
   separate YAML variants needed.
7. **Hand authority to AutoTuner**: `--fit off` is always emitted so
   llama.cpp's own auto-fit pass never silently re-tunes the computed
   values — an overcommit fails loudly (OOM) instead of being downscaled.

## Speculative decoding (MTP + n-gram)

AutoTuner combines up to three speculative paths in one `--spec-type` list:

- **Path A — external sibling drafter** (`-md`): a small `*-draft-*` /
  `*-assistant-*` model. Skipped when vision (`--mmproj`) is active (three
  large graphs at once is risky on 16 GB cards).
- **Path B — integrated MTP** (`--spec-type draft-mtp`): the trained MTP
  head inside the main GGUF (Qwen3.6-MTP, etc.). Coexists with vision.
- **Path C — draftless n-gram** (`--spec-type <ngram_method>`): no draft
  model needed; method chosen per profile via `ngram_method`
  (`ngram-mod` default, plus `ngram-map-k`, `ngram-map-k4v`,
  `ngram-simple`, `ngram-cache`).

**MTP + n-gram together:** only `ngram-mod` conflicts with `draft-mtp`
(combining them causes random mid-generation crashes, llama.cpp #23154),
so the tuner suppresses `ngram-mod` next to MTP. The `ngram-map-*` methods
were built for coexistence (PR #23269); set `ngram_method: ngram-map-k4v`
in an MTP profile to run both:

```yaml
# settings/qwen3_5-3_6.yaml  (Qwen3.6-MTP)
ngram_method: ngram-map-k4v   # runs alongside draft-mtp instead of being suppressed
ngram_k4v_size_n: 16          # optional (defaults from PR #23269)
ngram_k4v_size_m: 24
ngram_k4v_min_hits: 1
```

An unknown `ngram_method` falls back to `ngram-mod` with a warning at load
time rather than crashing at server start.

> Reality check: on bandwidth-bound MoE-A3B models, speculative decoding
> often does *not* beat the baseline (expert saturation). `ngram_method` is
> opt-in for that reason — measure tok/s before and after.

## Supported `llama-server` features (as of b9381)

| Flag | Status |
|---|---|
| `-fa [on\|off\|auto]` | ✅ emits `-fa on` |
| `-ctk/-ctv f16/q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl` | ✅ all in the KV dropdown |
| `--fit off` | ✅ always emitted (AutoTuner is the authority) |
| `--metrics` | ✅ Prometheus `GET /metrics` on the same host:port |
| `--reasoning` / `--think-budget` / `--chat-template-kwargs` | ✅ via reasoning panel |
| `--jinja` | ✅ visible toggle |
| `--mlock` / `--no-mmap` | ✅ auto + Windows guard; overridable |
| `-md` external drafter | ✅ presence of `-md` auto-enables the draft path |
| `--spec-type draft-mtp` | ✅ integrated MTP (mainline name since PR #22673) |
| `--spec-type ngram-mod` | ✅ default; suppressed next to MTP (#23154) |
| `--spec-type ngram-map-k4v` | ✅ MTP-compatible; runs with `draft-mtp` |
| `--spec-type ngram-map-k / ngram-simple / ngram-cache` | ✅ type token only; sub-params left to llama.cpp defaults |
| `--spec-draft-n-max / -p-min / -ngl` | ✅ via `draft_max` / `draft_p_min`; `-ngl` always 99 |
| `--spec-ngram-map-k4v-size-n / -size-m / -min-hits` | ✅ via `ngram_k4v_*` |
| `--n-cpu-moe` | ✅ automatic MoE offload |
| `--tensor-split` / `--main-gpu` | ✅ priority-weighted, single-GPU pinning |
| `--rope-scaling yarn` / `--numa` / `--no-context-shift` | ✅ (the last is de-duplicated) |

The b9371 → b9381 range is backend-only (WebGPU/Vulkan internals); no
server CLI flags changed, so command generation is unaffected.

## Monitoring (`/health` + `/metrics`)

`--metrics` is added at launch, exposing two HTTP endpoints on the same
`host:port` as the inference API (there is no separate metrics port):

- **`GET /health`** — `503` while loading, `200` when ready. The GUI polls
  this per instance to flip *Loading* → *Ready*.
- **`GET /metrics`** — Prometheus text format. Key gauges/counters use the
  `llamacpp:` prefix: `predicted_tokens_seconds`, `prompt_tokens_seconds`,
  `kv_cache_usage_ratio`, `kv_cache_tokens`, `requests_processing`,
  `tokens_predicted_total`, `prompt_tokens_total`.

Scraping without a Prometheus client:

```python
import urllib.request

def llama_metrics(base_url: str) -> dict[str, float]:
    out = {}
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=0.5) as r:
        for line in r.read().decode().splitlines():
            if line and not line.startswith("#"):
                name, _, val = line.partition(" ")
                try:
                    out[name] = float(val)
                except ValueError:
                    pass
    return out

# llama_metrics("http://127.0.0.1:1234")["llamacpp:predicted_tokens_seconds"]
```

`get_metadata.py` (drop it in the models folder, `pip install gguf`) dumps
the metadata of every model for debugging.

## Building llama.cpp

Recommended: Ninja or VS generator, native CPU optimizations, static
Release build, parallel jobs to taste. Example for the reference system
(Core Ultra 9 285K, RX 9070 XT 16 GB + R9700 AI Pro 32 GB):

```powershell
# Windows — Vulkan main build (SPIRV-Headers required since b9194)
cd ai-local
git clone https://github.com/KhronosGroup/SPIRV-Headers.git
cmake -S .\SPIRV-Headers -B .\SPIRV-Headers\build -G "Visual Studio 18 2026" -A x64 `
  -DCMAKE_INSTALL_PREFIX="ai-local/SPIRV-Headers/install"
cmake --build .\SPIRV-Headers\build --config Release
cmake --install .\SPIRV-Headers\build --config Release

git clone https://github.com/ggml-org/llama.cpp.git
Push-Location .\llama.cpp\tools\ui; npm ci; npm run build; Pop-Location
cmake -S .\llama.cpp -B .\llama.cpp\build -G "Visual Studio 18 2026" -A x64 `
  -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF `
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF `
  -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF -DGGML_VULKAN_CHECK_RESULTS=OFF `
  -DCMAKE_PREFIX_PATH="ai-local/SPIRV-Headers/install"
cmake --build .\llama.cpp\build --config Release --parallel 24
```

```bash
# Linux — ROCm/HIP main build
git clone https://github.com/ggml-org/llama.cpp llama.cpp && cd llama.cpp
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1200;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

## Project layout

```
auto_tuner.py        # terminal entry: menu + glue (+ --detach)
qt_launcher.py       # Qt GUI: picker, sticky options, multi-instance control
hardware.py          # CPU + multi-vendor GPU detection
scanner.py           # GGUF scanner: mmproj/draft pairing, capability detection
settings_loader.py   # YAML profile loader and matcher
tuner.py             # config calculation + llama-server command builder
launcher.py          # subprocess + Ctrl+C handling (Windows + Unix)
server_process.py    # ServerProcess wrapper (log capture, graceful stop)
app_settings.py      # persistent GUI prefs (autotuner_settings.json)
performance_target.py# VRAM/KV presets
get_metadata.py      # GGUF metadata dump (debugging)
settings/            # per-family YAML profiles
requirements.txt
README.md
```

## License

MIT.