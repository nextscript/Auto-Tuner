# llama.cpp b12811 (1.1.0-dev) audit — AutoTuner v5.6.6

Checked 2026-11-27 against the locally built Windows **b12811** HIP and Vulkan
servers (`1.1.0-dev`, commit `00d5c142c`) on Core Ultra 9 285K, AI PRO R9700
32 GB + RX 9070 XT 16 GB. The
[b12608…b12811 comparison](https://github.com/ggml-org/llama.cpp/compare/b12608...b12811)
contains **203 commits** and spans stable **v1.1.0** (`b12774`, commit
`074f1f3f`). See the [binary/CLI manifest](llama-b12811-server-flags.json) and
the [validation record](v5.6.6-validation.md).

## CLI and runtime changes

Both backends expose **430 option names / 344 long options** (up from 429 / 343):

- **Added (1):** `--ui-agent` (b12724, PR #33004): boolean flag enabling agent
  mode in Llama Chat with browser-side tools and browser-reachable MCP servers
  only (no server tools, no MCP proxy). AutoTuner does not emit it automatically
  and prunes it as a boolean flag on older builds.
- **Removed (0):** none.
- **Hadamard-folded models and tensor split (b12779, PR #33080):** llama.cpp now
  explicitly refuses `LLAMA_SPLIT_MODE_TENSOR` (`-sm tensor`) on `prism.hadamard`
  models and requires `-sm layer`. AutoTuner never emits `-sm tensor` (llama.cpp
  defaults to layer split), so Hadamard-folded models remain compatible.
- **Windows clean shutdown (PR #32080):** `llama-server` now handles
  `CTRL_BREAK_EVENT` cleanly on Windows.
- **FP8 KV cache (`f8_e4m3`) rechecked on b12811:** still works on **HIP**
  (80.4 tok/s on `fastcontext-1.0-4b-rl-q8_0.gguf`, answer `"4"`), while
  **Vulkan** still aborts in `ggml-backend.cpp:1372` (`SET_ROWS` unsupported on
  `Vulkan0`, upstream issue #32315). Auto stays on `q8_0`.

## New architectures, projectors, and draft support

### Build floors added (`_ARCH_MIN_BUILD` and `_MMPROJ_PROJECTOR_MIN_BUILD`)

| Architecture / Projector | Build | PR |
|---|---|---|
| `moshi` (Kyutai Hibiki / Moshi arch + `moshi` / `moshi_gen` projectors) | b12619 | #32491 |
| `hy_v4_flash` (Tencent HY4 Flash 220B-A12B arch + `hy_v4_flash` projector) | b12755 | #32960 |
| `raon1` (KRAFTON Raon-1 Mini Preview; kept on runtime-marker check for fork compat) | b12758 | #32176 |

### Speculative-decoding guards (`check_draft_model_build`)

- **Inkling 2 DSpark (`dflash.has_markov_context`, b12720, PR #32218):**
  enforced at `b12720+`.
- **Inkling 2 Mini DFlash2 (`dflash.selector_softcap_after_correction`, b12785,
  PR #33073):** enforced at `b12785+`.
- **Multi-stream DFlash (`dflash.target_hidden_size != dflash.embedding_length`):**
  KRAFTON's `dflash-raon-1-mini-preview-Q8_0.gguf` stores 4-stream mHC target
  features (`target_hidden_size = 10240` vs `embedding_length = 2560`), which
  mainline `b12811` does not read yet and aborts on (`check_tensor_dims: tensor
  'fc.weight' has wrong shape; expected 12800, 2560, got 51200, 2560`).
  `check_draft_model_build` now refuses multi-stream DFlash sidecars pre-launch
  on mainline builds (`_DFLASH_TARGET_HIDDEN_SIZE_VERIFIED_BUILD = 12811`, note
  that `llama-server` already contains `%s.target_hidden_size` for EAGLE-3 even
  though `dflash.cpp` ignores it).

## New and updated profiles

- **`bonsai-2_5.yaml`** (NEW — PrismML Ternary-Bonsai 2.5, `qwen35`, 262K,
  no `server_binary` pin): previously `Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf`
  matched the Bonsai 1 profile (`bonsai-ternary.yaml`, 64K + `2b_llama` pin).
  The mainline `Q2_0` GGUF (ggml type 42, `prism.hadamard.*`) with `mmproj` and
  embedded `draft-mtp` runs natively on mainline `b12811` HIP (61.4 tok/s,
  21/36 drafts accepted) and Vulkan (65.2 tok/s, 21/36 drafts accepted). Uses
  PrismML's Bonsai 2.5 thinking defaults (`temp 1.0, top_p 0.95, top_k 20,
  min_p 0.0`).
- **`sensenova-loop.yaml`** (NEW — SenseTime SenseNova-Loop 9B,
  `sensenova_loop`, `arch_authoritative: true`, 524,288 context, hybrid):
  previously fell back to `_default.yaml`. Upstream PR #32588 is still open and
  mainline `b12811` aborts with `unknown model architecture: 'sensenova_loop'`,
  so the profile gates on the `sensenova_loop` runtime marker and unlocks
  automatically once merged.
- **`raon-1.yaml`** (UPDATED) & **`raon-1-nano.yaml`** (NEW):
  - **Raon-1 Mini (`raon1`)** is now **mainline since b12758 (PR #32176)** and
    live-verified on `b12811` HIP (49.9 tok/s) and Vulkan (18.9 tok/s). Keeps
    `required_runtime_markers: [raon1]` so both mainline `b12758+` and
    `krafton-ai/llama.cpp` pass; sets `auto_draft: false` so the incompatible
    multi-stream DFlash sidecar does not block default launches.
  - **Raon-1 Nano (`raon1nano`)** is still **not mainline** (upstream PR #32200
    is open; `b12811` aborts with `unknown model architecture: 'raon1nano'`).
    Splitting it into `raon-1-nano.yaml` with `required_runtime_markers:
    [raon1nano]` prevents it from slipping past `raon-1.yaml`'s `raon1` marker
    now that `raon1` is in mainline binaries.
- **`inkling-2.yaml`** (UPDATED notes): mainline `inkling2_loop` (b12600+, PR
  #32694) + Inkling 2 DSpark (b12720+, PR #32218) + HIP/Vulkan Flash Attention
  indexed/compact bias (b12801 #33063, b12809 #32898) live-verified end-to-end
  on `b12811` HIP (21/28 drafts accepted) and Vulkan (19/30 drafts accepted)
  with `mmproj-Inkling-2-Mini-F32.gguf` and `dspark-Inkling-2-Mini-Q8_0.gguf`.
- **`hy4-mainline.yaml`** (UPDATED): adds `hy_v4_flash` (Tencent HY4 Flash
  220B-A12B, `b12755+`, PR #32960) alongside `hy_v4`.

## Rechecked gates on b12811

- **Inkling 2 Mini (`inkling2_loop` + `dspark`):** **MAINLINE & VERIFIED** on
  both HIP and Vulkan (`b12811`). Early pre-PR `inkling2` + `inkling2.num_loops`
  guard stays active for stale GGUFs.
- **Raon-1 Mini (`raon1`):** **MAINLINE & VERIFIED** on both HIP and Vulkan
  (`b12758+` / `b12811`). Companion DFlash (`target_hidden_size=10240`) remains
  gated until mainline supports multi-stream `target_hidden_size`.
- **Raon-1 Nano (`raon1nano`):** still not mainline (PR #32200 open), gated via
  `raon-1-nano.yaml`.
- **SenseNova-Loop 9B (`sensenova_loop`):** still not mainline (PR #32588 open),
  gated via `sensenova-loop.yaml`.
- **Qwen3.8 Flash Next + separate MTP head (`qwen4exp`):** still aborts with
  `GGML_ASSERT(buffer) failed` in `ggml-backend.cpp:346` on both `b12811` HIP
  and Vulkan (gate kept, `_QWEN4EXP_MTP_HEAD_VERIFIED_BUILD = 12811`).
- **Agnes ROCmFP4 (ggml type 100/101)** and **Ternary-Bonsai 2 PTQ1_0 / PQ2_0
  (ggml type 143 / 142):** still rejected on `b12811` (`should be in [0, 44)`),
  refused pre-launch by AutoTuner.
