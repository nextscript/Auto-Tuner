# llama.cpp b12608 (1.0.0-dev) audit — AutoTuner v5.6.5

Checked 2026-11-21 against the locally built Windows **b12608** HIP and Vulkan
servers (`1.0.0-dev`, commit `93cb37ea7`) on Core Ultra 9 285K, AI PRO R9700
32 GB + RX 9070 XT 16 GB. The
[b12187…b12608 comparison](https://github.com/ggml-org/llama.cpp/compare/b12187...b12608)
contains **421 commits** and spans stable **v1.0.0** (`b12359`). See the
[binary/CLI manifest](llama-b12608-server-flags.json) and the
[validation record](v5.6.5-validation.md).

## CLI and KV-cache changes

Both backends expose **429 option names / 343 long options** (up from 423 / 336):

- **Added (8):**
  - `--ple` / `--shard FNAME,...` (b12233, PR #31185): repeatable external GGUF
    shard / PLE loaders. Registered as repeatable value aliases in `tuner.py`.
  - `--moe-cache-cpu-share F` (b12191, PR #30966): CPU share for MoE cache
    misses. Registered as a value flag.
  - `--tts-lang`, `--tts-voice` (server TTS options, exposed on `llama-server`):
    registered as value flags.
  - `--allow-unauthenticated-tools` (b12339, PR #32113),
    `--media-allow-private-urls` (b12340, PR #32115),
    `--ui-mcp-proxy-allow-private-urls` (b12580, PR #32633): boolean security
    opt-ins; AutoTuner never emits them automatically.
- **Removed (2):** `-dt` / `--defrag-thold` (b12347, PR #30565). AutoTuner
  never emitted them.
- **FP8 KV cache (`f8_e4m3`, b12274, PR #28392):** added to `kv_quant_factor()`
  (`0.50`) and the Expert K/V dropdown. Live check on `b12608`: works on **HIP**
  (and CUDA/CPU), while **Vulkan** currently aborts with `SET_ROWS` unsupported
  on `cache_k_l0`. Auto mode stays on `q8_0`.
- **DFlash refactor (b12377, PR #32016):** adds explicit `draft-dflash2` and
  `draft-lilicorr` types; `draft-dflash` continues to auto-select DFlash,
  DFlash2, and LiLiCorr from draft metadata, preserving compatibility across
  older and newer builds.

## New architectures, projectors, and profiles

### Build floors added (`_ARCH_MIN_BUILD` and `_MMPROJ_PROJECTOR_MIN_BUILD`)

| Architecture | Build | PR |
|---|---|---|
| `inkling2` (Inkling 2) | b12406 | #32217 |
| `zonos3` (Zonos3-mini TTS) | b12424 | #32020 |
| `kokoro` (Kokoro TTS) | b12494 | #32413 |
| `tiny-voca` (Tiny-Voca TTS) | b12497 | #32427 |
| `zaya1` (Zyphra Zaya1 8B) | b12519 | #32071 |
| `domynedge` (Domyn Edge 4B) | b12568 | #32559 |
| `inkling2_loop` (Inkling 2 Mini) | b12600 | #32694 |

Projectors (`clip.projector_type` / `clip.vision.projector_type`): `magevl`
b12192, `kimik31` b12259, `inkling2v` b12406, `zonos3_gen` / `zonos3_spkenc`
b12424, `higgs_audio` / `vibevoice_asr` b12426, `fish_s2_gen` / `fish_s2_spkenc`
b12452, `yue2_gen` b12458, `minimax_music3_gen` b12479, `kokoro` b12494,
`tiny_voca` b12497, `voxcpm2_gen` / `voxcpm2_spkenc` b12531.

### New and updated profiles

- **`inkling-2.yaml`** (`inkling2` b12406+, `inkling2_loop` b12600+,
  `arch_authoritative: true`, 1M context): fixes a filename collision where
  `Inkling-2-Mini` previously matched `ling-ring.yaml` (substring `ling-2`).
  Early pre-PR Inkling 2 Mini GGUFs written with `general.architecture = inkling2`
  and `inkling2.num_loops` (24 physical blocks with `block_count=48`) abort on
  mainline `b12608` (`tensor 'blk.24.attn_q.weight' not found`) and are now
  refused pre-launch with instructions to use a mainline `inkling2_loop` GGUF.
- **`domyn-edge.yaml`** (`domynedge`, b12568+, 256K recommended / 1M native).
- **`zaya1.yaml`** (`zaya1`, b12519+, 131K, hybrid MoE).
- **`raon-1.yaml`** (KRAFTON Raon-1 Mini `raon1` and Raon-1 Nano `raon1nano`,
  262K): mainline `b12608` has no loader (`unknown model architecture`), so the
  profile checks for the `raon1` runtime marker (allowing `krafton-ai/llama.cpp`
  or a future mainline merge) and applies KRAFTON's sampling defaults.
- **`kimi-k3_1.yaml`** and **`minimax-m3.yaml`** notes updated for Kimi K3.1
  vision (`b12259`) and MiniMax-M3.1 (`b12432`).
- **`PaddleOCR-VL-1.6`** (`paddleocr-vl.yaml`) loads and pairs with
  `mmproj-PaddleOCR-VL-1.6.gguf` cleanly on `b12608`.

## Build recipes

- **ROCm 7.2 + Visual Studio 2026 (`14.52+` / `14.53+`):** MSVC's newer STL
  rejects ROCm 7.2's Clang 21 (`STL1000`). `Select-HipMsvcToolset` in
  `windows_llama_build_common.ps1` probes installed MSVC toolsets newest-first
  with ROCm `clang++` and swaps `VCToolsInstallDir`/`INCLUDE`/`LIB`/`LIBPATH`/`PATH`
  to the newest compatible toolset (e.g. `14.50.35717`) for all Windows HIP
  recipes. Normalized trailing-slash comparison and case-insensitive path
  replacement.
- **Llama Chat UI path (`b12451`, PR #32032):** the embedded UI moved from
  `tools/ui` to `ui/apps/chat` and is built out-of-tree by `scripts/ui-assets.cmake`.
  `Invoke-LlamaUiBuild` (`windows_llama_build_common.ps1`) plus
  `setup_llamacpp_cuda.ps1` and `setup_llamacpp_turboquant_cuda.ps1` now detect
  `ui/apps/chat/package.json` + `scripts/ui-assets.cmake` and pass
  `-DLLAMA_USE_PREBUILT_UI=OFF` cleanly instead of falling back to prebuilt HF
  downloads.
- **SPIRV-Headers (`#31416`):** `GGML_VULKAN_SPIRV_INCLUDE_DIR` is an optional
  override; existing `-DCMAKE_PREFIX_PATH` works unchanged.
- **Stable channel:** latest stable is now `v1.0.0` (`b12359`); resolved
  automatically by `llama_stable_*`.

## Rechecked gates on b12608

- **Qwen3.8 Flash Next + separate MTP head:** still aborts with
  `GGML_ASSERT(buffer) failed` in `ggml-backend.cpp:345` on `b12608` (gate
  kept, `_QWEN4EXP_MTP_HEAD_VERIFIED_BUILD = 12608`).
- **Agnes ROCmFP4 (type 100)** and **Ternary-Bonsai 2 PTQ1_0 (type 143):** still
  rejected on `b12608` (`should be in [0, 44)`), refused pre-launch by AutoTuner.
