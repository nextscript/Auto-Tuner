# Changelog

Release history of AutoTuner. Each llama.cpp audit and each release
validation has its own file under [docs/](docs/).

### v5.6.4 — llama.cpp b12187 audit (after v0.8.0 / v0.9.0), Xing 4.0 and DeepSeek-V4.1 on mainline, eleven new profiles

- **518 upstream commits audited (b11669 → b12187, spanning stable v0.8.0 =
  b11960 and v0.9.0 = b12110):** new options `--connect` / `--connect-code`
  (peer-to-peer tunnel) and the router-only `--models-memory-margin`
  (423 names / 336 long options); nothing removed. `--connect-code` is a value
  flag and is redacted like `--api-key` in the launch log, the OCR log and
  the diagnostics report. Unicode model paths still work after upstream
  dropped its Windows argv reconstruction (checked on both backends).
- **Xing 4.0 runs:** mainline loads `xing4_0` since b12156 (PR #31589). The
  block became a build floor; the profile uses the embedded MTP head with
  one draft token and no confidence gate (mirrored A/B through AutoTuner's
  plan: Vulkan 82 → 90–92 tok/s, HIP 64.5 → 74–77 tok/s; the usual p-min
  0.75 removed most of the gain).
- **DeepSeek-V4.1-Flash runs:** mainline `deepseek41` since b11870
  (PR #31086). The profile is runnable from b11870 and claims its
  architecture over a "DeepSeek-V4-…" filename (`arch_authoritative`); GGUFs
  of the old conversion-only layout stay refused. The plan is unmeasured (no
  local GGUF fits).
- **Kolibri 1** runs on mainline b11709+ (PR #30673). GGUFs made with the
  earlier hob patch store pre-tokenizer `kolibri1` and gating type 5, which
  mainline rejects; AutoTuner now adds the equivalent `--override-kv` (qwen2
  pre-tokenizer, sigmoid gating) on mainline. The local Q4_K_M answers
  correctly at 67.5 tok/s (Vulkan) / 63.0 (HIP). The runtime-marker check
  stays because the hob-llama.cpp fork reports b11434.
- **New profiles:** Kimi K3.1 (b12180+; matched the K3 profile with the
  b10749 floor before), LiquidAI LFM3 1.7B / VL / Audio / Omni, Muse Spark
  Open, Mistral Large 4, DiffuSpace Clover, Ricursive Ripple, Limite 1B
  Violetto, Recursive Flint, MiniCPM-OCR 1.0 and Nemotron 3.5 Transcribe (both
  with `--reasoning off`, as their PRs require) and Nemotron-3/3.5-Embed
  (embeddings). All fell back to the generic 8K profile. The decision
  profile now matches LiquidAI d1-2B, H2O-Lightning-4B and MiniCPM-OCR-Jev.
- **New build floors:** every architecture added after b11669
  (`_ARCH_MIN_BUILD`, 13 entries), the decision types `granite-switch`,
  `h2o-lightning`, `openbmb-minicpm-ocr-jev`, `lfm3-d1(-omni)` and 13 new
  image/audio projector types. The hybrid and MoE tables know the new
  architectures.
- **LiLiCorr drafters run as DFlash again:** NVIDIA's Bolt drafter keeps
  lattice layers beyond `dflash.block_count`; the scan took them for an MTP
  block and AutoTuner sent `draft-mtp`, which stopped llama-server ("failed
  to create MTP context"). EAGLE-3/DFlash/DSpark files are drafts by
  architecture now.
- **Drafters measured slower stay off by default** (new profile key
  `auto_draft: false`, GUI and CLI; a remembered choice wins): Nemotron 3.5
  Bolt + LiLiCorr cut the planned context on 48 GB from 168K/262K to 16K/24K
  for +7 % (Vulkan) / −25 % (HIP) decode; MiniCPM-V 4.7 + DSpark only sped
  up code on Vulkan and slowed German prose and every HIP run (down to
  7.6 tok/s from 83). MiniCPM now uses 7 draft tokens when chosen.
- **Draft pairing fix:** a DFlash/DSpark/MTP head whose size label differs
  from the target's is no longer paired (found when `NVFP4` became a quant
  tag: the Bolt DFlash head would have bound to Nemotron 3 Nano Omni). All
  86 local models keep their mmproj and draft.
- **Release build:** the Linux package is built on Ubuntu 24.04 again. GitHub
  moved `ubuntu-latest` to 26.04 (glibc 2.43), and the first v5.6.4 build
  failed to start on Debian stable and Linux Mint 22.3 (`GLIBC_2.43 not
  found` in the bundled libglib); the release check stopped it before
  publishing.
- **Build recipes:** no change needed. b12187 adds only `LLAMA_CONNECT`
  (OFF) and a SYCL option; the user-built trees match the recipes and the
  HIP tree now links OpenSSL.
- **Live chain on HIP and Vulkan (b12187):** all twenty planner/runtime
  scripts, negative loads (Agnes, Bonsai 2; Xing refused on b11669 and
  allowed on b12187), FastContext, Muse Glimmer JSON schema and the six new
  local models pass. Image + DFlash2 keeps 152/168 accepted drafts;
  Ling-3.0-flash on HIP now also passes the list bench; Qwen3.8 Flash Next
  `draft-mtp` still aborts upstream (gate kept, verified on b12187).

### v5.6.3 — llama.cpp b11669 / v0.7.0 audit, Bonsai 2 on mainline, new NVIDIA and OpenBMB models

- **131 upstream commits audited (b11538 → b11669, = stable v0.7.0):** the only
  CLI addition is `--spec-draft-n-ubatch` (420 names / 333 long options).
  AutoTuner now sends `--spec-draft-n-ubatch 64` for external DFlash, DFlash2
  and LiLiCorr drafts and for MTP heads: measured on RDNA4 this frees
  1.26 GiB of VRAM for Qwen3.8-27B + DFlash2 on Vulkan and 0.22 GiB for
  Gemma 4 31B + MTP head, with unchanged prompt speed, decode speed and
  accepted drafts. Plain draft models keep the full batch.
- **Decision models without prompt cache:** llama.cpp b11642 turns the prompt
  cache off for decision models unless `--cache-ram` is given; AutoTuner
  always sent it, so it now applies that default itself and reserves no RAM
  for it.
- **Prism Bonsai 2 on mainline:** b11556 loads the Hadamard fold for mainline
  conversions (`Q2_0`, `F16`). The PrismML `PTQ1_0`/`PQ2_0` files still need
  the prism fork, but b11669 contains the `prism.hadamard` loader string that
  AutoTuner's gate looked for, so it would have started them and failed while
  loading. The gate now also requires the fork's own type names. New profile
  `bonsai-2.yaml` for Bonsai 2 2B/4B/9B (they used the first-generation
  profile with 64K context).
- **New profiles:** NVIDIA Nemotron 3.5 Super VL and Nemotron 3.5 Bolt (both
  b11668+, previously the generic profile with 8K context), OpenBMB MiniCPM-V
  4.7, Poolside Laguna XS 2.1 (b11598+), Aleph Alpha Kolibri 1 (needs the
  hob-llama.cpp patch; refused on mainline instead of failing with "unknown
  model architecture"). `OrcaSAQ-2-27B` now uses the Qwen3.8 profile.
- **New model checks:** image projectors `minicpmv4_7` (b11551+) and
  `nemotron_omni` (b11668+) are refused on older builds; vLLM Decision 2 and
  D3 use the decision profile with their build floors, and the fork-format
  vllm-sr Decision-2.0 GGUFs name the fork they need; NVIDIA LiLiCorr drafters
  pair with their target and need b11668+.
- **Build recipes:** HIP builds now link OpenSSL — ROCm clang's GNU driver
  never found the Windows OpenSSL install, so every HIP `llama-server` was
  built without HTTPS. Re-running the HIP recipe rebuilds an existing tree
  with it (verified with a ROCm-clang build: `libssl`/`libcrypto` linked). The
  stable recipe no
  longer accepts a `0.7.0-dev` binary as stable `0.7.0`. No CMake option
  changed upstream; the b11669 caches equal b11538's.
- **Rechecked on b11669:** Qwen3.8 Flash Next `draft-mtp`, DeepSeek-V4.1,
  Xing 4.0, Prism PQ2_0/PTQ1_0 on mainline and ROCmFPX stay blocked; image +
  DFlash2 keeps 152/168 accepted drafts. Throughput is within ±2 % of b11538.
- [Audit](docs/llama-b11669-audit.md) · [Validation](docs/v5.6.3-validation.md).

### v5.6.2 — llama.cpp b11538 audit, Escha W2 models (issue #7), new upstream models

- **167 upstream commits audited (b11371 → b11538):** the only CLI addition is
  `--moe-cache-mib` (418 names / 331 long options). AutoTuner knows it as a
  value flag but does not set it: measured on RDNA4, the cache made decoding
  slower on Vulkan and on HIP traded +3…16 % decode for −13…32 % prompt
  speed in a realistic hybrid plan. The new default port 9931 does not matter
  because AutoTuner always passes `--port`.
- **EschaLabs W2 models ([#7](https://github.com/DaWasteh/Auto-Tuner/issues/7)):**
  `Escha-Qwen3.8-27B-W2` and its MTP head start on the Escha fork of
  llama.cpp. The fork is recognised by its loader in the selected runtime
  (like Prism and ROCmFPX), not by its folder name, and only the tensors the
  fork's MTP-only loader really skips are exempt. Mainline llama.cpp cannot
  load these files at all; AutoTuner now says so before launch instead of
  letting llama-server abort. The head is also recognised under its original
  Hugging Face name (no `mtp-` prefix) and paired automatically instead of
  appearing as a separate model. Thanks to Freak288 for the report and patch.
- **K2 Horizon** runs on mainline b11454+ (dense and MoVA-36B-A4B, tested on
  HIP and Vulkan); the profile no longer says a fork is required.
- **New decision models:** Perplexity pplx-decider and LiquidAI d1-3B /
  d1-omni-600M use the decision profile instead of a chat profile; each
  decision type needs the llama.cpp build that added it. Clef accepts images
  from b11418 on; AutoTuner warns when an older Clef GGUF (template without
  an image slot, HTTP 500 on every image) is used with its mmproj.
- **EmbeddingGemma 2:** new embedding profile (8K context in one batch instead
  of a 262K chat plan).
- **DFlash drafts with embeddings but no LM head** get a warning on
  b11132–b11511, where they accept almost no tokens; **PLaMo-3** GGUFs need
  b11455+.
- **RDNA4 Vulkan MoE regression fixed upstream:** b11423 reverted #29182;
  stock b11538 matches the opt-in RDNA4 build (+15…43 % MoE prompt speed
  against stock b11371), so the workaround build is no longer needed. HIP
  prompt processing is 4–7 % faster than on b11371.
- **Blockers rechecked on b11538:** Qwen3.8 Flash Next `draft-mtp`,
  DeepSeek-V4.1, Xing 4.0, Prism PQ2_0/PTQ1_0 and ROCmFPX stay blocked; image +
  DFlash2 keeps working (152/168 drafts).
- [Audit](docs/llama-b11538-audit.md) · [Validation](docs/v5.6.2-validation.md).

### v5.6.1 — llama.cpp b11371 audit, measured memory estimates, new model profiles

- **52 upstream commits audited (b11319 → b11371):** the only CLI change is
  `--spec-draft-sampling` (417 names / 330 long options). AutoTuner knows the
  flag; it is not set by default because both modes measured within run
  variation with AutoTuner's MTP plans.
- **Memory estimate measured against the real servers:** every loadable local
  model was started from its AutoTuner plan and compared with the process'
  GPU and host counters, with BF16, Q8_0 and Q4_0 KV for each model class.
  Weights, KV and recurrent state were already accurate; llama.cpp's
  **compute buffers** were missing. They are small on one GPU (about two
  attention masks) but about ten masks when layers are split across GPUs —
  4–6 GiB at 262k context and up to 16 GiB at 1M — so large dual-GPU plans
  spilled into shared memory. The estimate now shows `Compute GPU` /
  `Compute RAM`, a split plan lowers `-ub` (down to 256) before it would
  spill, and Auto context shrinks only if that is still not enough. An
  explicit context is never changed.
- **Windows RAM use of GPU plans:** with mmap the whole model file stayed in
  the server's working set (16 GiB of RAM "in use" for a model that lives in
  VRAM). Fully offloaded plans now load with `--load-mode none`: same speed,
  faster load, host RAM equal to the estimate. Hybrid plans, lazy tables,
  Linux and an explicit Expert choice are unchanged.
- **Gemma 4 images keep their full budget:** b11327+ caps a non-causal
  projector's image tokens to `--ubatch-size`. Plans with a Gemma 4 mmproj
  now use a physical batch of at least 1120 tokens (1078 instead of 536
  prompt tokens for a detailed image, and the correct answer).
- **New profiles:** decision models served through `/v1/systemone` (OpenJev,
  lev, Laya, Julia-1, Kev, Nimble, Clef / Clef-Flash; the whole prompt fits
  one batch and the context is capped), LLM-jp-4.1 (b11320+ Harmony dialect
  parser), GLM-OCR and dots.ocr (deterministic OCR sampling instead of the
  generic fallback).
- **Qwen3.8 Flash Next MTP:** a GGUF with an embedded MTP block needs b11330+.
  `draft-mtp` itself still aborts in llama.cpp b11371 (separate head and
  embedded block, HIP and Vulkan), so it stays off with the real reason;
  `merge_mtp_head.py` writes the embedded form for when upstream fixes it.
- **RDNA4 Vulkan MoE workaround re-qualified for b11371:** stock b11371 is
  still 14–31% slower in MoE prompt processing (upstream issue #29892); the
  opt-in build restores b11249 throughput.
- **Blockers rechecked on b11371:** DeepSeek-V4.1, Xing 4.0, Prism PQ2_0/PTQ1_0
  and ROCmFPX stay blocked; image + DFlash2 keeps working (152/168 drafts).
- **README:** release history, profile notes and review notes moved to this
  file.
- [Audit](docs/llama-b11371-audit.md) · [Validation](docs/v5.6.1-validation.md).

### v5.6.0 — authoritative MTP detection, OCR tasks, qualified RDNA4 runtime

- A complete tensor scan proving MTP weights absent overrides `MTP` in the
  filename as well as stale metadata; attached heads and legacy scans keep
  working.
- GLM-OCR, PaddleOCR-VL and dots.ocr use their publishers' task prompts in the
  shared document workflow.
- Image + DFlash2 is released again from the locally qualified b11319 on; only
  b10896–b11318 stays gated.
- Opt-in, exact-version RDNA4 build workaround for the Vulkan MoE prompt
  slowdown from upstream PR #29182.
- [b11302 audit](docs/llama-b11302-audit.md) ·
  [b11319 follow-up](docs/llama-b11319-audit.md) ·
  [Validation](docs/v5.6.0-validation.md).

### v5.5.9 — llama.cpp b11249 audit, MiMo-V2.6-Distill profile, Muse Glimmer JSON output

- **54 upstream commits audited (b11195 → b11249):** the only CLI change is
  `--rpc`, which every build now lists (RPC-less builds reject it at parse
  time; AutoTuner never plans RPC). Stable is still v0.5.0, the mainline
  build recipes stay unchanged and optimal for the 285K + 2× gfx1201 system.
- **New profile: MiMo-V2.6-Distill-Qwen-9B.** Xiaomi's Qwen3.5-9B fine-tune
  used to fall into the MiMo-V2.6 Flash/Pro MoE profile because of its file
  name (temp 1.0 / top_k 0, MoE notes). It now gets the official
  0.6 / 20 / 0.95 sampling; thinking, tool calls and images were tested
  locally on both backends.
- **Fix: default llama.cpp build.** When the saved build no longer existed
  (for example b11224 replaced by b11249) or on a fresh install, AutoTuner
  selected the first listed family — the Ternary/Bonsai Prism fork — instead
  of mainline. It now keeps the saved family and backend and takes its newest
  build, otherwise the newest mainline build; explicit choices are unchanged.
- **Muse Glimmer structured output:** `response_format` `json_schema` is
  enforced from b11249 on (PR #29615); on b11195 the same request came back
  as prose. **Nemotron 3 Puzzle:** HIP b11249+ runs its 96-channel Mamba-2
  scan on the GPU (PR #28717); the note recommends HIP.
- **Old blockers rechecked:** Qwen3.5/3.8 vision + DFlash2 still fails with
  HTTP 500 on b11249 (HIP and Vulkan), DeepSeek-V4.1 and Xing 4.0 still have
  no loader, Prism and ROCmFPX weights still need their forks; wording names
  b11249. The ROCmFPX pin stays because the fork's new main does not compile
  on Windows.
- **Measured:** `llama-bench` A/B on the R9700 (dense 27B and MoE 35B-A3B,
  Vulkan and HIP): throughput unchanged within ±1 %.
- [Audit](docs/llama-b11249-audit.md) · [Validation](docs/v5.5.9-validation.md).

### v5.5.8 — honest low-memory planning, adaptive Auto for 8 GB GPUs

- **No more fake 2k contexts:** when memory cannot hold even the 2,048-token
  minimum, Auto and pinned contexts are refused with the reason instead of a
  2k plan that could not allocate. On the development machine every default
  plan (64 local models × 4 tiers × 2 backends) is byte-identical to v5.5.7.
- **Adaptive low-memory Auto** for Windows + one Vulkan GPU ≤ 12 GiB + RAM
  ≤ 32 GiB (see [Performance targets](#small-windows-systems-adaptive-low-memory-auto)):
  on the reporter's RX Vega 8 GiB / 16 GiB RAM machine Gemma-4-26B Q4_0 and
  Qwen3.6-35B-A3B IQ3_XXS now load and answer in all four tiers with
  60k–210k context; reproduced here on b11195 with the same budget.
- **low_vram MoE placement** no longer reserves GPU KV although the KV lives
  in RAM, so more expert blocks fit on the GPU.
- **RAM-aware MoE placement:** if CPU experts would overcommit free RAM and
  moving experts to the GPU removes that while keeping an 8k working
  context, Auto moves them; otherwise the previous plan and warning stay.
- [Validation](docs/v5.5.8-validation.md).

### v5.5.7 — llama.cpp b11195 audit, faster Ternary-Bonsai fork, newest fork wins

- **35 upstream commits audited (b11160 → b11195):** identical flags and
  help text on both backends; stable is still v0.5.0. The mainline build
  recipes stay unchanged and optimal for the 285K + 2× gfx1201 system.
- **MiMo-V2 MTP heads:** the converter can now write a separate MTP head
  for MiMo-V2 (PR #29294); such a head needs b11195+, and AutoTuner refuses it
  on older builds instead of letting llama-server abort.
- **Ternary-Bonsai 2 fork pin → prism-b10743:** Vulkan now runs PQ2_0 on
  the GPU (57.6 instead of 3.0 tok/s through AutoTuner's plan) and PTQ1_0
  decodes 5.7× faster; on HIP PTQ1_0 decode +76 % and PQ2_0 prompt +37 %
  (R9700, `llama-bench`). Includes the fix for the PQ2_0 CPU-repack crash
  (Prism #180).
- **Fix:** with an older and a newer pinned fork installed side by side,
  AutoTuner picked the oldest tree for a profile such as Ternary-Bonsai 2.
  It now takes the newest build (active backend first); an explicit fork
  choice still wins.
- **Measured:** the new tiled CPU matmul (PR #27851) is slower for
  CPU-only IQ4_XS prompt processing on this AVX2 CPU; GPU and hybrid plans
  are unaffected. Old blockers rechecked (all still open); wording names b11195.
- [Audit](docs/llama-b11195-audit.md) · [Validation](docs/v5.5.7-validation.md).

### v5.5.6 — llama.cpp b11160 audit, start minimized at login, Ling-3.0-flash-VL

- **Start minimized after login:** new option under *Start after login*. A
  login start opens without a window — in the notification area when *Hide
  on close* is on, otherwise minimized on the taskbar; until you choose it
  follows *Hide on close*. The login entry now passes `--autostart`; older
  entries for the same installation are upgraded automatically, manual starts
  are unchanged and a second start still restores the window.
- **55 upstream commits audited (b11105 → b11160, llama.cpp 0.5.0-dev):**
  identical flags and help text; stable v0.5.0 = b11146. Build recipes stay
  unchanged and optimal for the 285K + 2× gfx1201 system. The new Vulkan
  int8 cooperative-matrix path (PR #27952) is on automatically and measured
  **+13–15 % Q8_0 prompt processing** on the R9700 (Q4_K/IQ4_XS unchanged).
- **Ling-3.0-flash-VL** profile (b11156+) and M-RoPE metadata gate; **Gemma 4
  DSpark/DFlash** drafts need b11132+.
- **Old blockers rechecked** (all still open): Qwen vision + DFlash2 (#27408),
  DeepSeek-V4.1 (#28696), Xing, Prism PQ2_0/PTQ1_0, ROCmFPX; wording names b11160.
- [Audit](docs/llama-b11160-audit.md) · [Validation](docs/v5.5.6-validation.md).

### v5.5.5 — llama.cpp b11105 audit, new profiles and parser fixes

- **42 upstream commits audited:** same CLI flags; presence penalty zero is
  now explicit despite the new environment defaults. No automatic network
  exposure from the new multi-bind `--host` support.
- **Profiles:** MiMo-V2.6 Flash/Pro RL, FastContext 1.0 4B SFT/RL, plus
  recognition-only safeguards for unsupported Xing4.0 and the Voice Lab
  VoxCPM2 BaseLM component. All nine language packs updated.
- **Muse Glimmer:** recommend b11100+ for the first-token tool-call parser
  fix (#29242), while retaining its b10353 loader minimum.
- **Old issues rechecked:** safeguards retained unless a fix is evidenced;
  the audit distinguishes open, stale-closed, candidate patches and actual
  runtime results. DeepSeek-V4.1 and vision/DFlash2 wording names b11105.
- Includes the refreshed `models_metadata.md` and previously unpublished
  [Voice Lab Vulkan PS1](building%20llama.cpp/voicelab_voxcpm2_vulkan_build.ps1).
  This builds the separate `voxcpm2-cli` / `llama-tts-server` pipeline; it
  does not turn AutoTuner into a TTS server or modify existing llama builds.
- [Audit](docs/llama-b11105-audit.md) · [Validation](docs/v5.5.5-validation.md).

### v5.5.4 — llama.cpp b11063 audit, Ling 3.0 (Bailing V3) parser

- **llama.cpp b11063 audit** (21 commits after b11042): option set
  unchanged, the new timestamped startup log line from PR #29125 is
  stripped from the `--help` capture before hashing, and the full HIP and
  Vulkan live chain passes on the new trees (Nemotron 3 Nano Omni back to
  25.7 / 71.9 tok/s without the foreign load of the b11042 run).
- **Ling 3.0:** b11063's dedicated Bailing V3 chat parser (PR #28682) is
  selected automatically from the template with `--jinja`, which
  `ling-3.yaml` already passes together with `--reasoning-preserve`;
  AutoTuner sets no `--reasoning-format` that would bypass it. Live with the
  local Ling-3.0-flash IQ3_M on both backends: with thinking on, a tool
  call now arrives as `tool_calls` (auto and required tool choice) and the
  reasoning stays in `reasoning_content`. The profile note and the nine
  language packs document the parser and recommend b11063+ for tool use;
  `min_llama_build` stays b10749 (older builds load and run the model).
  The local quant turned out to be trunk-only (`nextn_predict_layers = 1`
  but no `nextn.*` head tensors): AutoTuner's split-shard scan already
  reports no embedded MTP for it and plans without `draft-mtp`, which
  matters because a forced MTP graph aborts llama-server on such files.
- **Gates re-verified on b11063:** Qwen3.5/3.8 vision + DFlash2 still
  fails with HTTP 500 on HIP and Vulkan (upstream #27408), so the gate's
  message now names b11063; DeepSeek-V4.1-Flash stays blocked (PR #28696
  still a draft, wording moved to b11063); Ternary-Bonsai 2 (Prism types
  142/143, #29058) and Agnes-3.0-Flash MTP-ROCmFP4 (types 100/101,
  PR #24185) are still refused by mainline b11063 and stay on their forks.
  PrismML's newer `prism-b10709` tag only adds DFly/DSpark drafter code (no
  Vulkan PQ2_0 kernels), so the Bonsai recipes keep the b10687 pin.
- [Audit](docs/llama-b11063-audit.md) · [Validation](docs/v5.5.4-validation.md).

### v5.5.3 — Agnes-3.0-Flash Preview profile, ROCmFPX gate and fork recipes

- **New profile `agnes-3_0-flash.yaml`:** Agnes-3.0-Flash Preview (Agnes AI,
  33B dense hybrid: 54 gated delta-rule + 18 global-attention layers, 262,144
  context, text/image/video, tool calling, Apache-2.0). Mainline llama.cpp has
  no `agnes` architecture and no support for its parallel SwiGLU branch, so
  every community GGUF is a `qwen35` conversion; the profile is
  filename-gated (`agnes-3.0-flash`, `agnes-3_0-flash`, `agnes3.0`, …) so
  plain Qwen3.5 re-quants stay with the generic Qwen profile. Sampling follows
  `generation_config` (temp 1.0 / top_p 0.95 / top_k 20, coding temp 0.6),
  `--jinja --reasoning-preserve` for the reasoning_effort / preserve_thinking
  template controls, and the in-file MTP head uses the model card's measured
  optimum (`--spec-draft-n-max 4`, `--spec-draft-p-min 0.0`). All nine
  language packs carry the note.
- **ROCmFPX gate:** the GGUF header scan now records the distinct ggml tensor
  types (`__ggml_types__`). A GGUF whose tensors or `general.file_type` fall
  into the ROCmFPX reserved range (types 100–111, file types 100–124 —
  kingjones777's `MTP-ROCmFP4-STRIX_LEAN` = 106, `-COHERENT` = 102,
  `Q8_0-ROCmFPX` = 111) is refused before launch unless the selected
  llama-server or its llama library contains the fork's `ROCmFP` loader
  string, with the exact mainline error (`invalid ggml type 100. should be in
  [0, 43)`, upstream PR #24185 open) in the message. Verified absent from the
  local b11042, v0.4.1, prism-b10687 and tq trees.
- **Fix:** a profile `draft_p_min: 0.0` (Nemotron 3.5 DSpark, Agnes) was
  silently turned back into 0.75 when the command was built; an explicit 0.0
  is now emitted.
- **New recipes `rocmfpx_vulkan_llama_build.ps1` / `rocmfpx_hip_llama_build.ps1`:**
  the ROCmFPX fork pinned to its main commit `aed0d5fd9` (2026-09-06, fork
  build 11544, mainline base b10766) as `fpx_b11544_{vulkan,hip}_llama.cpp`,
  gfx1201 HIP. `Invoke-LlamaPinnedForkBuild` gained `-PatchFiles` (unified
  diffs applied on the pinned commit; an existing tree missing a patch is
  patched and rebuilt), used for `patches/rocmfpx-rdna4-mmq-fallback.patch`:
  the fork's ROCmFPX MMQ fallback (PR #20) is wired only into the RDNA3
  selector, so unpatched gfx1201 aborts with `J_best=0` on the first prompt
  batch. Live on both backends with the kingjones777 imatrix files: mainline
  b11042 refuses them (`invalid ggml type 101. should be in [0, 43)`), the
  fork plans and serves STRIX_LEAN and COHERENT through the real planner
  (`--jinja`, `--mmproj`, `draft-mtp` n-max 4 / p-min 0): HIP about 67 tok/s
  decode for both tiers, Vulkan 29 tok/s (LEAN) / 5 tok/s (COHERENT, dual-scale
  ROCmFP4 shaders), image requests succeed with the MTP head loaded.
- llama.cpp itself is unchanged since the [b11042 audit](docs/llama-b11042-audit.md);
  the mainline trees were not rebuilt. [Validation](docs/v5.5.3-validation.md).

### v5.5.2 — Ternary-Bonsai 2 (PrismML prism-b10687), llama.cpp b11042 audit

- **Ternary/Bonsai recipes re-pinned** to the fork release
  `prism-b10687-5d80cff` (2026-09-17): the first pin that carries the
  Bonsai 2 `PTQ1_0` (dense trits, 1.75 bpw) kernels next to `PQ2_0`. Both
  Windows trees were built with the unchanged recipes; the folder now uses
  the fork's own build number (`2b_b10687_{vulkan,hip}_llama.cpp`) instead
  of the mainline merge-base count that would have collided with the old
  tree.
- **New profile `bonsai-2-27b.yaml`** for `Ternary-Bonsai-2-27B-PTQ1_0` /
  `-PQ2_0` (Qwen3.8-27B hybrid base, `qwen35`, 262k, thinking by default):
  official thinking sampling (temp 1.0 / top-p 0.95 / top-k 20), `--jinja`,
  mmproj support, `server_binary: 2b_llama` and `min_llama_build: 10687`
  (the old prism-b10660 tree rejects PTQ1_0). The patterns are longer than
  the first-generation `ternary-bonsai-27b`, so Bonsai 27B, the 8B ternary
  family and Qwen3.6 keep their profiles; notes in all nine language packs.
- **Mainline gate for Hadamard-folded GGUFs:** `check_model_build` refuses a
  GGUF with `prism.hadamard.*` metadata on any runtime whose llama library
  lacks the `prism.hadamard` loader string, with the exact upstream
  behaviour in the message (b11042: "invalid ggml type 143. should be in
  [0, 43)", issue #29058). Ordinary GGUFs never trigger the scan.
- **Live on both backends (fork prism-b10687):** HIP runs PQ2_0 and PTQ1_0
  entirely on the GPU (text, image and tool-capable `--jinja` template);
  Vulkan runs PTQ1_0 on the GPU but keeps PQ2_0 CPU-mapped (the fork has
  no PQ2_0 Vulkan kernels). Mainline b11042 Vulkan/HIP reject both files.
- **llama.cpp b11042 audit:** 12 commits after b11030, no CLI change
  (`--help` byte-identical, manifest `docs/llama-b11042-server-flags.json`);
  the full live chain was repeated on the b11042 trees.
- **Updater verified against GitHub:** the frozen-binary swap
  (`_BinaryUpdateWorker`), the git fast-forward and the source-ZIP overlay
  (`_UpdateWorker`) were run for real (see validation).
- [Audit](docs/llama-b11042-audit.md) · [Validation](docs/v5.5.2-validation.md).

### v5.5.1 — llama.cpp b11030 audit, DFM Mimir profile, Nemotron latent-MTP gate

- Local Vulkan and HIP **b11030** trees from the unchanged Windows recipes
  (built the same day, then re-verified by all four recipes including the
  `v0.4.1` stable pair). The b10977→b11030 range (53 commits) changes no
  server option: `--help` is byte-identical on both backends and to b10977,
  and all 166 profile/mode commands per backend parse. The Vulkan source
  split (PR #28732) needed no recipe change.
- **New profile `dfm-mimir.yaml`:** DFM Mimir 1B (`hrm_text`, PR #27625,
  b11003+), Danish/English HRM-Text with two 16-layer stacks and 128 KV
  slots per token, 4,096 context, `--jinja`, generic sampling; all nine
  language packs carry its note. The KV estimate follows the GGUF
  `block_count` (128), matching the 3 GiB at 4k in F16 that upstream
  documents.
- **Nemotron latent-MoE MTP gate:** `nemotron_h_moe` GGUFs with an MTP
  block and `moe_latent_size` (Nemotron 3 Super) are refused before launch on
  builds older than b11025 (PR #29018) with the exact upstream failure
  ("wrong number of tensors") instead of an opaque loader abort; Lightning,
  Nano and stripped GGUFs are untouched.
- Image + DFlash2 on Qwen3.5/3.8 was re-run and **still fails on b11030**
  (HIP and Vulkan); the b10896+ gate stays and now names b11030. The
  DeepSeek-V4.1 block names b11030 (PR #28696 is still an open draft).
- [Audit](docs/llama-b11030-audit.md) · [Validation](docs/v5.5.1-validation.md).

### v5.5.0 — llama.cpp b10977 / v0.4.1 audit, Maple-Preview profile

- Fresh local Vulkan and HIP **b10977** builds (and the **v0.4.1** stable
  trees, which are b10964) from the unchanged Windows recipes. The b10948→b10977
  range changes no server option: `--help` is byte-identical on both backends
  and all 164 profile/mode commands per backend parse. Upstream removed the
  precompiled headers again (PR #28892) after a CPU work-buffer heap
  corruption (PR #28882); the recipes needed no change.
- **New profile `maple.yaml`:** DeepGrove Maple-Preview 20B-A1B ternary
  reasoning MoE (`maple`, PR #27000, b10964+), TQ1_0/TQ2_0 GGUFs, 131k
  context, thinking-model sampling, `--jinja --reasoning-preserve`; all nine
  language packs carry its note. Run live on the b10977 Vulkan tree.
- Image + DFlash2 on Qwen3.5/3.8 was re-run and **still fails on b10977**
  (HIP and Vulkan); the b10896+ gate stays and now names b10977. The
  DeepSeek-V4.1 block names b10977 (PR #28696 is still open).
- [Audit](docs/llama-b10977-audit.md) · [Validation](docs/v5.5.0-validation.md).

### v5.4.9 — control-API state, benchmark time budget, off-thread server probes

- **Control API:** a switch the GUI rejects before touching any server
  (busy, hardware pending, unknown model or runtime, shutting down) no longer
  clears the active model; `/api/v1/status` keeps reporting the serving model
  and proxied requests without `model` keep routing to it. Rejections after
  the old server was stopped still report idle. Live-verified: 409
  `autotuner_busy` during an OCR job left Agents-A1 active and routable.
- **Performance test:** reaching the total time limit now ends exploration
  and decides with the candidates measured so far ("total time limit reached
  after N of up to M candidates" in the decision) instead of discarding every
  measurement; only a limit hit before the baseline finishes is still a
  failure. `AUTOTUNER_BENCHMARK_DEADLINE_S` overrides the limit for
  diagnostics. Live-verified with a 240 s limit on Qwen3.6-35B (4 of 18
  candidates, result saved).
- **GUI:** the `/health`, `/v1/models` and `/slots` probes moved from the
  500 ms GUI timer onto a worker thread; the window stays fluid while a
  model loads and during benchmark suites with many server starts.
- No new llama.cpp build; b10948 remains the audited runtime.
  [Validation](docs/v5.4.9-validation.md).

### v5.4.8 — llama.cpp b10948 rebuild, planner and lifecycle fixes

- Fresh local Vulkan and HIP **b10948** builds from the unchanged Windows
  recipes; upstream's `-fno-pch-timestamp` for clang (PR #28816) is active in
  the HIP tree without a recipe change. `--help` keeps b10930's exact option
  set (only the `-j/--json-schema` wording changed); all 162 profile/mode
  commands per backend parse. Image + DFlash2 was re-run and **still fails
  on b10948** (HIP and Vulkan); the b10896+ gate stays and names b10948.
- **GPU detection:** two cards from the same detector whose names are
  substrings of each other (RTX 3060 / 3060 Ti, RX 9070 / 9070 XT) were
  merged into one entry with the larger VRAM; duplicates are now only merged
  across detectors and never across differing PCI device ids.
- **Context planner:** an exhausted VRAM budget (`max_fit_ctx == 0`) was
  treated as "unknown" and produced the 32k auto floor or an unclamped
  user pin; it now clamps to the 2048 floor with the usual warning.
- **Extra CLI flags:** repeatable value flags (`-ot/--override-tensor`,
  `--override-kv`, `--lora`, `--control-vector`, `--api-key`, draft
  variants) keep every distinct value instead of dropping the second flag
  and leaking its value as a positional argument; pruning for older
  binaries removes their values too.
- **OCR cancel** no longer blocks the GUI thread until the in-flight page
  finishes (the socket is aborted instead of closed under the reader's
  lock), and a cancel that races the LibreOffice start still kills
  `soffice`. `ServerProcess` (CLI `--gui`, OCR, benchmark) decodes
  llama-server output as UTF-8 with replacement so a non-cp1252 byte can no
  longer end the log reader and stall the server on a full pipe; a graceful
  stop resets the wrapper. The CLI log viewer keeps the user's scroll
  position, inserts plain text and bounds the document.
- A missing or edited built-in English language pack degrades to source
  text instead of a startup traceback; user themes save on file systems
  without hard links; one malformed profile YAML no longer aborts loading
  of all profiles; the Windows DXGI/WMI VRAM fallbacks run only when WMI
  did not cover a card (removes a PowerShell spawn from the 6-second
  refresh); GGUF header scans seek past tokenizer string arrays.
- DeepSeek-V4.1 stays blocked (PR #28696 still open); the block and all nine
  language packs name b10948.
- [Audit](docs/llama-b10948-audit.md) · [Validation](docs/v5.4.8-validation.md).

### v5.4.7 — llama.cpp b10930 rebuild, vision + DFlash2 gate widened

- Fresh local Vulkan and HIP **b10930** builds from the unchanged Windows
  recipes; upstream's new precompiled-header/unity build works with both the
  Visual Studio 2026 and ROCm 7.2 Ninja toolchains (MSB8027 is benign).
- The Qwen3.5/3.8 vision + DFlash2 HTTP 500 **still reproduces on b10930**
  (HIP and Vulkan) and on b10903; PR #28715 (b10906) did not resolve it. The
  gate now starts at b10896 (PR #28587) and stays open-ended until an actual
  image + DFlash2 request succeeds on a newer build.
- b10930 `--help` is byte-identical to b10901; all 162 profile/mode commands
  per backend parse. DeepSeek-V4.1 stays blocked (PR #28696 still open).
- [Audit](docs/llama-b10930-audit.md) · [Validation](docs/v5.4.7-validation.md).

### v5.4.6 — model profiles and llama.cpp b10901 validation

- Separate recognition-only DeepSeek-V4.1-Flash profile with a hard runtime
  gate: no false compatibility with V4 or conversion-only GGUFs.
- New Nex-N2.5-mini and dealignai GLM-5.3-Cybersecurity profiles, including
  model-specific sampling/template behavior and notes in all nine languages.
- Reproduced and guarded b10901 Qwen3.5/3.8 vision + DFlash2 HTTP 500 on
  both HIP and Vulkan; tested the image-without-draft and text-with-draft paths.
- Existing CLI flags, Q8 KV, tools, multi-GPU placement and explicit lazy
  loading pass the b10901 checks. No blanket claim for untested models/backends.
- [Audit](docs/llama-b10901-audit.md) · [Validation](docs/v5.4.6-validation.md).

### v5.4.5 — llama.cpp b10878, removed flags and explicit lazy loading

- Migrates the removed mmap/mlock/Direct-I/O switches to `--load-mode`, keeps
  legacy semantics on old runtimes and prevents Extra flags bypassing locking
  safety checks. Corrected the Expert `auto` label: runtime/device policy.
- Giant row-table memory plans now assert `--lazy-mode on`; upstream `auto`
  may eagerly load the entire table on iGPUs. Saved `auto` Extras migrate.
- Failed/partial `--help` cannot silently prune command options.
- HIP build recipe selects the new `GGML_CUDA_FA_QUANTS=all` by source
  capability; old stable/fork trees keep their supported boolean option.
- Existing benchmark history stays intact; automatic winners need a fresh
  search after the loading-policy change (search schema 6).
- [Audit](docs/llama-b10878-audit.md) · [Validation](docs/v5.4.5-validation.md).

### v5.4.4 — llama.cpp b10863, model profiles and lazy-memory safety

- Verified Spark-X2.5 **4B** and MiniCPM5 **2B** generation/tool calls with Q8
  KV on b10863 Vulkan and HIP. Separate 2B/2.6B MiniCPM sampling profile.
- New **K2 Horizon** profile with an explicit IFM-fork gate and conservative
  MoVA-aware whole-layer placement; not confused with Kimi-K2.
- Qwen Flash Next's **51.2B-entry / 26.8-GiB PLE table is already NVMe-backed
  lazy data**, confirmed by a real launch. Unsupported flags and conflicting
  overrides cannot silently invalidate that memory plan. Active pages still
  consume RAM; the residency budget is not an OS cache limit.
- Fixed external-draft GPU placement and the HIP shared-output-head abort;
  actual two-GPU DFlash2 inference passes on both AMD backends. Added Kimi-K3
  pre-b10853 checkpoint notice, alias-safe merging and CPU forced-lock fix.
- 80 model profiles, translated notes in all nine languages; prior benchmark
  measurements remain intact, old automatic winners require a fresh search.
- [Audit](docs/llama-b10863-audit.md) · [Validation](docs/v5.4.4-validation.md).

### v5.4.3 — Q8 KV by default, llama.cpp b10839

- Q8-first placement and cache selection, Q4 only for capacity, no automatic
  F16/BF16 quality upgrade. Correct per-slot and one-sided-pin budgets.
- Compatible F16 fallback for non-FA/head-dimension restrictions; Flash
  Attention changes cascade through memory planning. Diffusion CLI enacts KV.
- Spark X2.5 profile and separate mainline HY4 support; profile notes in all
  nine languages. Exact b10839 flags checked by portable regression tests.
- Old measured winners require a fresh search. Pages retains all historical
  data and explicitly distinguishes it from the new Q8 policy.
- Validation: [`docs/v5.4.3-validation.md`](docs/v5.4.3-validation.md).

### v5.4.2 — one instance, no leftover process

- **Why:** with *Hide on close* enabled, X parks AutoTuner in the notification
  area. Windows 11 hides new tray icons in the overflow menu, so the running
  AutoTuner looked like a leftover process, and the next double-click on
  `AutoTuner.exe` started a second copy that fought over the control-API port
  and the console log (`autotuner_console.log` could not rotate because the
  first instance still held it).
- **Single instance:** `single_instance.py` holds a per-user, per-data-folder
  `QLocalServer` lock (named pipe on Windows, socket in `$TMPDIR` on Unix,
  stale sockets are cleaned after nobody answers). A second launch sends an
  *activate* request, the running window is restored from the tray or from a
  minimised state and raised (the new process grants the foreground right
  first), and the second process exits. Simultaneous launches resolve to one
  primary; `AUTOTUNER_ALLOW_MULTIPLE_INSTANCES=1` disables the guard.
- **Quit always ends the process:** *Quit* from the tray menu, the Quit
  button, and Ctrl+C now call `QApplication.quit()` after a successful close.
  A window that was already hidden in the tray is not "the last visible
  window" for Qt's `quitOnLastWindowClosed`, so the event loop could keep
  running. A daemon watchdog additionally ends the process 15 s after the
  event loop finished if interpreter teardown hangs.
- **Verification:** `test_single_instance.py` covers the lock in one process
  and across processes, the activation signal, both X paths (plain close, and
  hide-to-tray followed by a Quit that must end `app.exec()`), and window
  restoration; two opt-in tests (`AUTOTUNER_GUI_PROCESS_TESTS=1`, Windows
  desktop) start the real GUI process, close it with `WM_CLOSE`, and assert
  the process exits, then hide a first instance and check that a second
  launch restores it and exits. Evidence in
  [`docs/v5.4.2-validation.md`](docs/v5.4.2-validation.md).

### v5.4.1 — translated hover help, Русский, free KV precision, prompt-cache reuse

- **Every explanation follows the interface language:** the two-level hover
  help ("In short" / "Technical details") on all controls, the Expert panel
  labels and sections, the performance-test and OCR dialogs, message boxes,
  the model context menu, and the list/tree row tooltips are now translation
  keys. The language manager takes the generated tooltip HTML apart,
  translates each section as plain text, and rebuilds it, including
  runtime-composed lines such as `Active build:` or the per-tier bullet list.
  Formerly hard-coded German strings (`Favoriten`, `Hinzufügen…`,
  `GGUF-Ordner öffnen`, update/error dialogs) are English source text now, so
  the English (UK) interface is fully English. A test extracts every help
  constant from `qt_launcher.py` with the `ast` module and fails when any
  built-in pack lacks or leaves one untranslated.
- **Русский:** a ninth complete built-in pack (all 427 strings and all 76
  profile explanations). A Dutch sentence in the control-API help was reworded;
  the other packs were reviewed and kept.
- **Free KV precision upgrade:** Auto still plans placement and context
  against the symmetric Q4_0 baseline, but takes symmetric F16 or Q8_0 when
  that exact context still fits the same VRAM plan. Real `llama-bench` runs on
  b10797 Vulkan (R9700, Qwen3.8-27B Q4_K_XL and Devstral-Small-2 24B Q4_K_XL)
  showed Q4_0 K/V costing 4–9 % decode and 4–24 % prompt speed versus F16, with
  Q8_0 within 1–3 % of F16; the evidence is in
  [`docs/v5.4.1-validation.md`](docs/v5.4.1-validation.md). A 9B model with a
  32k request now runs F16 K/V and stays pinned to one card; a 27B Q4 on the
  32 GB card keeps its 262k context at Q8_0 without waking the second GPU.
- **Prompt-cache reuse:** whenever the host prompt cache (`--cache-ram`) is
  on, AutoTuner also emits `--cache-reuse 256`, so coding agents and chat
  clients that edit the middle of an otherwise identical prompt re-use every
  unchanged chunk through KV shifting instead of re-processing the whole
  prompt. llama-server disables it on caches that cannot shift; older binaries
  drop the flag in the compatibility filter.
- **Where the memory reserves go (checked, unchanged):** the per-card
  headroom (6 %/10 % of VRAM, floored at 1–1.5 GB), the 0.15–0.30 GB safety
  band, the 0.6 GB/slot workspace, and the 3–15 % long-context guard cost
  context, not speed, on fully offloaded models; they cost speed only where
  they move dense layers or MoE experts to the CPU, which is exactly the
  Safe/Balanced/Throughput trade-off the tier selector exposes.

### v5.4.0 — llama.cpp b10797, MTP sidecar preflight, readable performance summary

- **Exact b10797 compatibility:** the stock Vulkan binary at commit
  `832fd6f17` exposes the same 331 long options as b10786. The eleven
  upstream commits add `n_expert_used_max()` (Puzzle-style per-layer expert
  arrays), a GBNF fix for empty object schemas, SYCL/OpenCL/CUDA kernel work,
  and a CMake rebuild fix; none changes a flag, an architecture, or a
  loader contract AutoTuner depends on. The help-capture test now checks the
  newest captured build automatically.
- **MTP sidecar preflight:** llama-server loads a standalone `-md` MTP head
  with the target's full architecture loader, so a sidecar must carry the
  same root tensors as the model (`token_embd`, `output_norm`, Qwen3.8 Flash
  Next's `output_hc_norm`, …). Community "shared"/embedding-free sidecars do
  not, and every Flash Next Quick-suite lane that used them died at
  `check_tensor_dims` after the 50 GB target had already loaded. AutoTuner now
  reads the root tensors of the sidecar and of every target shard, refuses
  such a drafter before launch with the missing tensor names, and the Quick
  suite lists the lane under *Failed/skipped* instead of burning minutes per
  mode. Working sidecars (Qwen3.6 MTP, Gemma 4 assistant, DFlash/DSpark)
  are unaffected.
- **Readable performance summary:** the completion and failure reports use a
  scrollable dialog capped at three quarters of the screen height, so the OK
  button is always visible even for long multi-model summaries.
- **Validation and binary:** see [`docs/v5.4.0-validation.md`](docs/v5.4.0-validation.md)
  and [`docs/llama-b10797-audit.md`](docs/llama-b10797-audit.md).

### v5.3.9 — llama.cpp b10786, campaign-ready control API, localized profile notes

- **Exact b10786 compatibility:** stock Vulkan and HIP binaries at commit
  `de8656bd9` expose the same 331 long options as b10760. New profiles cover
  NVIDIA Nemotron 3 Super 120B-A12B and Nemotron Labs 3 Puzzle 75B-A9B; Puzzle
  is gated to b10786 because that build first reads the per-layer expert
  arrays (PR #25444). DeepSeek-V4 notes describe the new b10786 vision
  projector, the scanner's hybrid list now uses the exact upstream
  `deepseek4`/`falcon-h1`/`granitehybrid` names, and the Expert tooltip plus
  this README explain llama.cpp's new preserved-reasoning default.
- **External control API for benchmark campaigns:** `GET /api/v1/runtimes`
  lists every llama-server build in the toolbar with backend and probed build
  identity; `POST /api/v1/switch` accepts `runtime_id` and `timeout_s`, so a
  client can run several models across Vulkan/HIP/CUDA builds in sequence;
  `/api/v1/status` returns the direct llama-server URL, alias, PID, runtime,
  model, launch settings, devices, and a redacted command line;
  `/api/v1/models` adds size, quant, parameter count, family, and architecture.
  AutoTuner writes a small `~/.autotuner/control_api.json` discovery file
  (token only while enabled) so Pi and the Supercalc benchmark never parse the
  large settings file. The Pi extension reads it first and falls back to a
  bounded regex scan. See [`docs/control-api.md`](docs/control-api.md).
- **Profile explanations follow the interface language:** every profile's
  `notes` is now English in YAML, language packs carry an optional
  `profile_notes` map, and all bundled packs (English, German, Dutch,
  Swedish, Japanese, French, Greek, Polish; Russian since v5.4.1) translate
  all 76 profiles;
  switching the language re-renders the configuration preview. Missing
  entries in custom packs fall back to English. See
  [`docs/languages.md`](docs/languages.md).
- **Closer-to-optimum performance search:** decoupled thread probes, a
  low-thread probe for fully offloaded models, larger micro-batch candidates,
  and a hill-climb refinement stage with an 18-candidate Standard budget.
- **Clearer benchmark report:** a *Recommended settings per model* section,
  metric legend, applied-settings column, ranked candidate charts with
  settings captions, and collapsed per-run diagrams.
- **Validation:** source suite, exact b10786 help matrix, and the rebuilt
  Windows artifact are recorded in
  [`docs/v5.3.9-validation.md`](docs/v5.3.9-validation.md); the upstream
  review is in [`docs/llama-b10786-audit.md`](docs/llama-b10786-audit.md).

### v5.3.8 — llama.cpp b10760, current model profiles, cleaner HIP builds

- **Exact b10760 compatibility:** stock Vulkan and HIP binaries at commit
  `0f3a71be1` expose the same 331 long options as b10743; every directly
  profile-emitted option remains accepted. b10749's NextN repairs preserve the
  existing quarantine boundary, while its no-scan SSM correction now gates
  Ling 3, Kimi-K3, and the new Kimi Linear profile.
- **Popular-family coverage:** dedicated Gemma 3/3n, Mistral Small 3.1/3.2,
  Llama 4, Kimi Linear, PaddleOCR-VL, and Ornith 1.5 profiles add official
  context/sampling/runtime contracts. Llama 3 and Llama 4 no longer share the
  inaccurate 128k ceiling.
- **HIP warning audit:** the unused `CMAKE_HIP_COMPILER` argument is removed.
  A clean 630-target ROCm 7.2/gfx1201 build kept warnings visible, completed,
  exposed both AMD GPUs, and passed the deterministic two-GPU
  `HIP MULTI GPU OK` semantic gate. The remaining diagnostics are upstream
  HIP/CUDA template and Windows-portability warnings, not recipe failures.
- **Validation and binary:** full source, exact-runtime, warning-classification,
  and rebuilt Windows artifact evidence is recorded in
  [`docs/v5.3.8-validation.md`](docs/v5.3.8-validation.md); the upstream review
  is in [`docs/llama-b10760-audit.md`](docs/llama-b10760-audit.md).

### v5.3.7 — diagram-first benchmarks and a live hardware dashboard

- **Charts before detail:** the self-contained performance report now opens with
  the stored hardware snapshot, fastest-result cards, one aligned fastest lane
  per model, drafted-token acceptance, and every successful candidate chart.
  The compact winner table, methodology, and native expandable run evidence are
  grouped below the visual dashboard, with sticky navigation and responsive,
  keyboard-focusable overflow regions.
- **Public benchmark snapshot:** `python publish_benchmarks.py` exports all local
  Quick pass, Standard, and Custom evidence to `benchmark-site/index.html` while
  removing Windows, POSIX, UNC, and `file:` paths from every public dynamic
  field. A script-free CSP and a final parsed-HTML path scan fail closed before
  an unsafe snapshot can be written.
- **GitHub Pages deployment:** pushes that change the static site run a dedicated
  Pages workflow. Repository validation receives read-only source access; the
  separate deployment-only job executes no repository code and alone receives
  `pages: write` plus OIDC permission. The live result is linked at
  [dawasteh.github.io/Auto-Tuner](https://dawasteh.github.io/Auto-Tuner/), with
  setup and refresh steps in
  [`docs/benchmark-pages.md`](docs/benchmark-pages.md).
- **Richer future hardware evidence:** new successful benchmark records retain
  total system RAM alongside OS, CPU/core, GPU, and VRAM data. Existing records
  remain compatible and display every hardware field they already captured.
- **Validation and local binary:** 464 source tests pass with 7 optional/platform
  skips; the committed 227-run public snapshot has 992 unique element IDs, 146
  candidate chart cards, no scripts or local paths, and was inspected in desktop
  and narrow browser layouts. The tracked Windows x64 `dist/AutoTuner.exe` was
  rebuilt for v5.3.7 and passed frozen and visible-window/icon smokes. Full
  evidence is in [`docs/v5.3.7-validation.md`](docs/v5.3.7-validation.md).

### v5.3.6 — llama.cpp b10743 compatibility and Fedora-safe responsiveness

- **Current lazy-row CLI:** generated commands now use b10700+'s
  `--lazy-mode auto`. Binary-aware preparation translates to the legacy
  `--tensor-read-lazy` spelling only when an older runtime advertises it, so
  Qwen3.8-Flash-Next and Gemma 4 do not silently lose their lazy-table memory
  contract on current llama.cpp.
- **NextN regression quarantine:** b10741-b10748 can abort on integrated MTP,
  standalone Gemma 4 assistant heads, or per-layer metadata arrays. AutoTuner
  now removes only `draft-mtp` and its draft-only arguments while retaining
  compatible ngram methods, blocks model/draft shapes that cannot fall back,
  refuses to record a disabled MTP benchmark under the wrong label, and
  identifies b10749+ (PRs #28173/#28183) as the fixed boundary.
- **Qwen3.8-Flash-Next correctness gate:** the qwen4exp profile now requires
  b10737+, which includes the QSA sequence-copy, block-position, multimodal
  input, and CUDA-abort fixes from PR #27941. Existing measured lazy-weight and
  context×ubatch memory accounting is intentionally unchanged.
- **Value-bearing profile flags stay atomic:** `--samplers`, `--pooling`, and
  current reasoning/lazy aliases are classified correctly, preventing rejected
  duplicate overrides from leaving orphan values in the final argv.
- **Fedora-responsive UI test:** hardware-strip wrapping is tested immediately
  below and above its measured Qt font-metric threshold instead of assuming
  distro-specific text widths at 1320 px. The full source suite passes with
  451 tests and 7 platform/environment skips.
- **Validation:** exact b10743 Vulkan loaded and generated from the real
  Qwen3.8-Flash-Next split GGUF with `--lazy-mode auto`; the Qwen3.6 embedded
  MTP model loaded and decoded through the guarded base-model fallback. Source,
  runtime, Fedora, and release evidence is recorded in
  [`docs/v5.3.6-validation.md`](docs/v5.3.6-validation.md).

### v5.3.5 — responsive hardware, complete Quick runs, Granite 4.2

- **Responsive hardware strip:** long CPU/GPU names no longer become a hidden
  minimum-width contract. The four fields use height-for-width wrapping and
  reflow from one row to a two-column grid (or one column at compact widths),
  while wide windows retain the original single row.
- **Quick suites finish every selected job:** the shared 20-minute model budget
  and per-run Quick deadline are removed. Startup, individual HTTP-request, and
  explicit cancellation safeguards remain bounded, so a stuck server still
  fails safely without discarding later modes merely because earlier models
  were slow.
- **IBM Granite 4.2:** a dedicated filename-gated 3B/8B/30B reasoning profile
  adds the official temp 1.0/top-p 0.95 sampling, 131,072 native context,
  embedded Jinja thinking/tool template support, and generic ngram speculation
  without falsely claiming MTP. Granite 4.1 remains isolated despite the shared
  `granite` architecture.
- **llama.cpp b10717 validation:** exact stock Vulkan and HIP builds at commit
  `a32af33de` loaded the official Granite 4.2 8B Q8_0 GGUF on GPU. Both returned
  final content plus structured `reasoning_content`; Vulkan additionally
  returned a structured `get_current_weather` tool call. The full 444-test
  regression, Ruff, compileall, source/frozen smokes, and rebuilt Windows binary
  evidence are in [`docs/v5.3.5-validation.md`](docs/v5.3.5-validation.md).

### v5.3.4 — aligned reports, multilingual UI, secure model control

- **Aligned model overview:** decode, prompt/encode, and end-to-end winner panels
  now stack vertically, retain the same model/backend/build column order, and
  share one synchronized horizontal scrollbar. More model columns fit on screen
  without losing cross-metric alignment.
- **Persistent compact toolbar:** low-frequency Fonts, language, Update, and
  Settings controls live in a secondary toolbar that remains open until the
  ellipsis is clicked again. It no longer disappears when the pointer leaves.
- **English (UK) plus seven complete built-ins:** English (UK) is the canonical
  fallback, joined by grammatically reviewed Deutsch, Nederlands, Svenska,
  日本語, Français, Ελληνικά, and Polski packs. Validated schema-1 custom JSON
  packs import atomically into per-user storage, can replace their own ID, live
  retranslate open/later dialogs, and survive source or frozen upgrades.
- **Authenticated local model control:** an opt-in, loopback-only API publishes
  stable IDs for the live scanned catalogue, serializes model transitions on
  Qt's GUI thread, reuses saved AutoTuner launch settings, waits for the prior
  process to exit and the new `/health` check to pass, and stops only the
  API-managed server. In-flight proxy leases prevent a switch from truncating
  an active stream or routing it to the wrong model.
- **OpenAI and Pi integration:** authenticated `/v1/*` requests are rewritten to
  llama-server's launch alias without forwarding client credentials; SSE chunks
  are flushed immediately. The async Pi extension discovers models before
  `pi.registerProvider()`, supports refresh, and honours persisted or
  environment-owned loopback credentials and ports.
- **Validation:** 438 tests pass with 7 platform/environment skips; Ruff,
  compileall, diff checks, TypeScript/Pi runtime discovery, PyInstaller analysis,
  frozen resources, and the Windows frozen smoke gate pass. Full evidence is in
  [`docs/v5.3.4-validation.md`](docs/v5.3.4-validation.md).

### v5.3.3 — adaptive campaigns, backend evidence, complete analysis

- **Per-job adaptive context:** all-model campaigns now recompute a safe context
  for every model, selected llama build, and performance mode. Fixed context is
  the only path that requests exact real-context validation. Dense partial
  placement balances both VRAM and host-RAM KV pools, while Qwen3.8 Flash Next
  accounts for QSA/context×ubatch workspace, all parallel slots, and its
  26.82-GiB read-lazy PLE table without allowing an infeasible 2k floor.
- **Backend- and runtime-qualified evidence:** HIP, Vulkan, CUDA, CPU, and other
  measurements no longer borrow concrete sibling-backend winners. Multiple
  installed builds of the same backend can run in one campaign; every
  runtime-qualified result survives for comparison while launch snapshots and
  preferred modes remain backend-scoped and environment-validated. Profile
  export/import preserves those backend preferences.
- **Complete performance analysis:** the in-app view is selected-model-only and
  retains Winner, Measured, and Failed candidates with exact settings, samples,
  errors, and bounded logs. The generated accessible HTML remains global and
  compares model/backend/build winners with vertical side-by-side metric bars.
- **Transactional reruns and settings writes:** “Reset old measured data, then
  rerun all” clears scoped measured evidence in one fail-closed write before a
  server starts, keeps Custom profiles, and checkpoints both successes and
  bounded failures. Serialized read/modify/write transactions and unique temp
  files prevent concurrent GUI/checkpoint updates from overwriting each other.
- **Evidence-gated model profiles:** full GLM-5.3 shares the verified `glm-dsa`
  contract with GLM-5.2; GLM-5.3-Flash stays separate, requires a `glm5next`
  patched runtime, and is conservatively capped at 262,140 pending upstream
  1M-context validation. HY4 Preview requires `hyv4` runtime capability and
  documents its patched loader and GGUF-without-MTP limitation.
- **Validation:** 419 tests pass locally across estimator, persistence, Qt,
  report, profile, and build-recipe coverage; the detailed release gate is in
  [`docs/v5.3.3-validation.md`](docs/v5.3.3-validation.md).

### v5.3.2 — Q4 defaults, Flash-Next long context, robust performance search

- **Global Q4 KV Auto default:** normal llama.cpp launches now use symmetric
  `q4_0/q4_0`; higher or mixed precision remains available through Expert
  settings. Placement uses the same Q4 contract instead of spreading onto a
  peer GPU merely to upgrade KV precision.
- **Qwen3.8 Flash Next context regression fixed:** the full 26.82 GiB PLE table
  remains visible as a file-backed lazy mapping, but only its measured 5%
  active-row working set is charged as mandatory physical RAM. The profile now
  defaults to Safe/ubatch 64. On the local 16+32 GiB AMD system, both b10679
  Vulkan and HIP loaded `130000` requested context (`130048` allocated), Q4 KV,
  and completed a real request with only ~30 GiB RAM free before launch.
- **Performance/MTP decisions hardened:** Quick evidence cannot replace an
  already validated Perform profile; binary/backend/device/quality/search
  fingerprints invalidate stale results; Standard tests the top two
  thread×batch families; winner promotion requires a conservative paired 3%
  gain; and MTP depth needs two meaningful regressions before stopping.
  Cross-mode fastest selection runs only for identical workloads.
- **Real MTP counter validation:** the Qwen3.8-27B embedded MTP path reported
  bounded native acceptance counters on both backends (Vulkan 79/85, HIP
  72/74). Full commands and bounded evidence are in
  [`docs/v5.3.2-validation.md`](docs/v5.3.2-validation.md).

## Profile notes

Notes on individual profiles, newest first.

- **v5.6.0:** A complete tensor scan proving MTP weights absent now overrides
  `MTP` in the filename as well as stale metadata; separately attached heads
  and inconclusive/legacy scans keep working. Shared OCR presets use the
  documented GLM, Paddle and dots tasks. b11302 HIP/Vulkan expose the same
  416 flags as b11249; no new tuning flag or build-recipe change is needed.
  Real image+DFlash2 requests now pass on b11319 HIP/Vulkan with accepted
  drafts; only b10896–b11318 remains gated. Stock b11319 still has the Vulkan
  MoE prompt slowdown, isolated to upstream PR #29182. An opt-in, exact-version
  [RDNA4 build workaround](docs/rdna4-moe-workaround.md) restores throughput on
  the qualified AMD boards without replacing stock builds or saved selections.
  [b11302 audit](docs/llama-b11302-audit.md),
  [b11319 follow-up](docs/llama-b11319-audit.md),
  [validation](docs/v5.6.0-validation.md).

- **v5.5.9:** Muse Glimmer's `--jinja` parser enforces `response_format`
  `json_schema` only from b11249 (PR #29615); older builds answer in prose.
  Nemotron Labs 3 Puzzle's 96-channel Mamba-2 scan runs on the GPU with HIP
  b11249+ (PR #28717); Vulkan still schedules it on the CPU, so the note
  recommends HIP. Loader floors are unchanged. [Audit](docs/llama-b11249-audit.md).

- **v5.5.7:** MiMo-V2 can now be exported as trunk plus a separate MTP head
  (converter `--mtp` / `--no-nextn`, PR #29294); such an MTP-only head loads
  only on b11195+, so older builds refuse it as a draft before launch. The
  Ternary-Bonsai 2 note names the new fork pin prism-b10743, whose Vulkan
  backend runs PQ2_0 on the GPU. [Audit](docs/llama-b11195-audit.md).

- **v5.5.6:** Ling-3.0-flash-VL gets its own profile because the text
  profile's `ling-3.0-flash` pattern would otherwise have claimed it; its
  M-RoPE GGUFs load silently wrong before b11156 (PR #29151), so a
  metadata gate also refuses renamed files there. Gemma 4 DSpark/DFlash
  drafts (PR #29226) are refused below b11132. Chat sampling follows the
  published generation config (1.0/0.95/20), coding the card's evaluation
  default 0.6. [Audit](docs/llama-b11160-audit.md).

- **v5.5.5:** MiMo-V2.6 adds a b11102-gated profile for the new conversion
  and corrected tool parser; its native 1M limit is not a local-memory
  guarantee. FastContext 4B SFT/RL no longer falls through to generic 8k
  settings. Xing4.0 and VoxCPM2 BaseLM are explicitly recognized **without
  claiming standalone runtime support**. All nine language packs include
  the new notes. [Sources, boundaries and audit](docs/llama-b11105-audit.md).

- **v5.5.1:** **DFM Mimir 1B** (`hrm_text`, PR #27625, first tagged in
  b11003) is Danish Foundation Models' Danish/English instruction-tuned
  Hierarchical Reasoning Model: two 16-layer transformer stacks (low/high)
  run in alternating cycles (2 H × 3 L) over the same tokens, so every token
  passes 128 block slots built from 32 physical layers. llama.cpp keeps one
  KV entry per pass (128 slots, about 3 GiB at the native 4,096 context in
  F16), which AutoTuner sizes from the GGUF `block_count`; decode costs
  roughly four times a dense model of equal width and upstream implements
  causal attention only (no prefix-LM). The profile requires b11003+, caps
  the context at 4,096, keeps `--jinja` for the Gemma-4-style template
  (thinking opt-in through `enable_thinking`) and uses the generic sampling
  defaults because DFM ships no `generation_config`. Community GGUFs:
  `noctrex/DFM-Mimir` (BF16/F16/Q8_0). The same release adds a
  **Nemotron latent-MoE MTP gate**: Nemotron 3 Super GGUFs that carry their
  MTP block (`moe_latent_size` 1024) fail llama.cpp's tensor count before
  b11025 (PR #29018: "wrong number of tensors; expected 781, got 779"), with
  or without speculation, so AutoTuner refuses those builds before launch
  instead of letting the loader abort. [Audit](docs/llama-b11030-audit.md).

- **v5.5.0:** **DeepGrove Maple-Preview** (`maple`, PR #27000, first tagged
  in b10964 = stable v0.4.1) is a 20B-A1B ternary reasoning MoE with 24
  layers, 256 experts (8 active), a 3:1 sliding-window-512/global attention
  pattern and 131,072 native context; the official GGUFs are TQ1_0/TQ2_0
  with a Q4_K or F16 head. The profile requires b10964+, keeps `--jinja
  --reasoning-preserve` for the thinking-by-default ChatML template and
  uses the usual thinking-model sampling (temp 0.6 / top_p 0.95 / top_k
  20) because DeepGrove ships no `generation_config`. Vulkan has TQ1_0/TQ2_0
  kernels; CUDA/HIP have none, so llama.cpp keeps the ternary expert
  tensors on the CPU there. [Audit](docs/llama-b10977-audit.md).

- **v5.4.6:** DeepSeek-V4.1-Flash has a new CED/CSA2/Engram architecture;
  conversion-only PR #28696 is not inference support. Its profile blocks
  unsafe V4 fallback, including early GGUFs mislabeled `deepseek4`.
  **Nex-N2.5-mini** uses temp 0.7/top_p 0.95/top_k 40 and its embedded Nex
  template (`chat_template_kwargs.reasoning_effort`: none/medium/high).
  **dealignai GLM-5.3-CYBERSECURITY-FP8** shares base GLM's architecture but
  needs repeat penalty 1.1 and `--no-reasoning-preserve` to honor its
  `clear_thinking=true` default. Keep its own GGUF template. Ordinary V4,
  Qwen and GLM profiles are unchanged. [Sources and limits](docs/llama-b10901-audit.md).
  The **v5.4.7** b10930 re-check found PR #28696 still open, so the V4.1
  block named b10930; the **v5.4.8** b10948, **v5.5.0** b10977 and
  **v5.5.1** b11030, **v5.5.4** b11063, **v5.5.5** b11105, **v5.5.6** b11160,
  **v5.5.7** b11195 and **v5.5.9** b11249 re-checks found it still open (not
  merged, updated 2026-09-28); the block now names b11249. Conversion alone still does not provide an
  inference runtime.

- **b10760 coverage refresh:** Gemma 3 and Gemma 3n now retain their distinct
  128k/32k limits and multimodal caveats; Mistral Small 3.1/3.2 uses Mistral's
  low-temperature recommendation; Llama 4 uses Meta's temp 0.6/top-p 0.9 and
  lets GGUF metadata distinguish Scout's 10M, Maverick's 1M, and base 256k
  limits. PaddleOCR-VL gets deterministic OCR/table/formula/chart prompts, and
  Ornith 1.5 overrides the older 1.0 filename profile with its current
  general/coding contract. Kimi Linear is isolated from K2/K3 and, together
  with Ling 3 and Kimi-K3, requires b10749's corrected no-scan SSM tensors.
- **GLM-5.3** (non-Flash) deliberately extends the GLM-5.2 profile because
  both publish `glm-dsa`, 1M context, temp 1.0/top-p 0.95, IndexShare, and a
  NextN head. **GLM-5.3-Flash** is separate: it is a 320B-A18B multimodal
  KDA/DSA hybrid (`glm5next`) with different placement/KV behavior. Its
  upstream llama.cpp PRs were still open when this profile shipped, so use a
  matching patched build. The preview profile currently caps automatic context
  at 262,140 (while documenting the model's native 1M metadata); current
  development caveats and the IQ3+ recommendation are recorded in the profile.
- **Hy4 Preview** uses Tencent's official temp 0.9/top-p 1.0 sampling and
  1M native-context metadata, while AutoTuner still caps the actual context to
  each selected quant/backend/system. Stock b10813+ supports the official
  `hy_v4` GGUF format; older `hyv4` GGUFs still need their community patch.
  Architecture metadata disambiguates these formats even with identical filenames.
  That conversion drops the native MTP layer, so the profile does not claim
  embedded speculative decoding.
- **Qwen3.8** keeps a separate profile from Qwen3.5/3.6 because its official
  coding evaluations use the thinking defaults (`temp 1.0`, `top_p 0.95`,
  `top_k 20`) instead of Qwen3.6's older coding temperature. The 27B variant
  is multimodal and can disable thinking; the 2.4T-A95B variant is text-only
  and always thinks. Both retain the existing `qwen35`/`qwen35moe` GGUF
  architecture family, so filename patterns deliberately take precedence
  without claiming those ambiguous architecture fallbacks.
- **Qwen3.8 Flash Next** is the separate `qwen4exp` architecture merged in
  llama.cpp (loader landed in b10660; AutoTuner gates at b10666 for validated
  QSA graph-memory coefficients). AutoTuner excludes its ~26.8 GiB read-lazy PLE n-gram
  table from splittable layer weights, budgets only its measured active-row
  residency against physical RAM while still displaying the full file-backed
  mapping, and budgets the extra QSA indexer KV. It defaults to Safe/ubatch 64;
  Balanced/Throughput remain available at 128/256 because graph buffers scale
  with context × ubatch.
- **Nemotron 3.5 Lightning** uses NVIDIA's temp 1.0/top-p 0.95 contract and
  b10665's DSpark variant. `draft_max: 7` plus `draft_p_min: 0.0` supports
  checkpoints that deliberately omit the confidence head.
- **Nanbeige 4.2 3B** uses its 256k context and separate official reasoning
  (temp 0.6) versus agent/tool (temp 1.0) sampling defaults.
- **Mellum 2** is a code-focused MoE (64 experts, 8 active; 12B total /
  2.5B active; 128k ctx). `ngram_method` is deliberately set to
  `ngram-map-k4v` (MTP-compatible) so it survives whether or not llama.cpp's
  `mellum` loader executes JetBrains' MTP head.
- **EXAONE 4.5** is a dense 33B VLM. Its integrated MTP/NextN tail blocks are
  *loaded but not executed* by llama.cpp (b9500, same as `exaone-moe`), so the
  profile uses draftless `ngram-mod`, not `draft-mtp`. License is
  non-commercial (research/academic only).
- **Step 3.5 / 3.7-Flash** both load under `step35` and both carry a real
  MTP-3 head (`num_nextn_predict_layers`), so the profile pairs
  `draft-mtp` + `ngram-map-k4v` with `draft_max: 3`. ⚠️ At ~196–198B MoE
  these exceed this machine's 48 GB (VRAM+RAM) — realistically need heavy
  `--n-cpu-moe` offload or are not runnable; the profile is for
  correctness/future smaller builds.
- **Granite 4.2** adds IBM's dense 3B/8B/30B reasoning family with Jinja tool
  calling and the required temp 1.0/top-p 0.95 sampling in every mode. All
  checkpoint configs expose 131,072 native tokens; IBM separately advertises a
  512k long-context extension only for 30B, so AutoTuner stays at the published
  native serving limit until an explicit GGUF scaling recipe is validated. The
  profile is filename-gated because its `granite` architecture is shared with
  Granite 4.1.
- **Granite Embedding R2** is an *embedding* model — it runs as an embedding
  endpoint (`--embeddings --pooling cls`, set via `extra_args`), not a
  chat/completion model. Sampling/draft fields are inert in that mode.

## Earlier llama.cpp audit summaries


The **b11538** (`79e2e74eb`, `0.6.0-dev`) [audit](docs/llama-b11538-audit.md)
covers 167 commits after b11371. The only CLI addition is
**`--moe-cache-mib`** (418 names / 331 long options); the default port moved
to 9931, which does not matter because AutoTuner always passes `--port`. New
for AutoTuner: **EschaLabs W2** GGUFs are refused on runtimes without the
Escha loader and start with their MTP head on the Escha fork (issue #7),
same-architecture MTP heads are recognised by their tensors whatever their
file name, **K2 Horizon** runs on mainline, **pplx-decider / LiquidAI d1** use
the decision profile with per-type build floors, **EmbeddingGemma 2** has an
embedding profile, Clef images need a 2026-10-05+ GGUF, and DFlash drafts with
embeddings but no head get a warning on b11132–b11511. Qwen3.8 Flash Next
`draft-mtp` still aborts upstream and stays off. The Vulkan MoE prompt
regression is gone (upstream revert in b11423): stock b11538 matches the
opt-in [RDNA4 build](docs/rdna4-moe-workaround.md), which is no longer
needed. See [validation](docs/v5.6.2-validation.md).

The previous **b11249** (`6d78fb072`, `0.5.0-dev`) [audit](docs/llama-b11249-audit.md)
covers 54 commits after b11195. The only CLI change is **`--rpc`**, which is
now listed on every build (PR #29537) and rejected at parse time by builds
without RPC; AutoTuner never plans RPC, and a user-supplied `--rpc` extra is
no longer pruned silently but fails with llama.cpp's own message (416 names /
329 long options). New for AutoTuner: **Muse Glimmer** `json_schema` output
needs b11249+ (PR #29615, measured) and **Nemotron 3 Puzzle** runs its
96-channel Mamba-2 scan on the GPU with HIP b11249+ (PR #28717). Speculative
decoding, mtmd and server batches moved to `llama_batch_ext` (PR #29385);
text DFlash2, MTP, lazy PLE and Ling 3.0 plans pass on both backends. The
Qwen vision/DFlash2, DeepSeek-V4.1, Xing, Prism and ROCmFPX safeguards
remain; the ROCmFPX pin stays because the fork's new main does not compile
with MSVC. See [validation](docs/v5.5.9-validation.md).

The previous **b11195** (`d834d44e6`, `0.5.0-dev`) [audit](docs/llama-b11195-audit.md)
covers 35 commits after b11160. Option set **and** `--help` text are
unchanged (415 names / 328 long options). New for AutoTuner: separate
**MiMo-V2 MTP heads** need b11195+ (PR #29294) and are refused as drafts on
older builds. The new **tiled CPU matmul** (PR #27851) is on by default and
has no server option; on this AVX2 CPU it is slower than b11160 for CPU-only
IQ4_XS prompt processing (−14…−27 %), while hybrid GPU plans are unaffected.
The Qwen vision/DFlash2, DeepSeek-V4.1, Xing, Prism and ROCmFPX safeguards
remain. See [validation](docs/v5.5.7-validation.md).

The previous **b11160** (`70c4e1582`, `0.5.0-dev`) [audit](docs/llama-b11160-audit.md)
covers 55 commits after b11105. The option set **and** the `--help` text are
unchanged (415 names / 328 long options); only `--version` reports the 0.5.0
bump, which AutoTuner does not parse. New: **Ling-3.0-flash-VL** (b11156+,
own profile plus an M-RoPE metadata gate), a b11132 floor for **Gemma 4
DSpark** drafts, and the Vulkan **int8 cooperative-matrix MMQ** path for
RDNA3/RDNA4 (on by default, no setting). The Qwen vision/DFlash2,
DeepSeek-V4.1, Xing, Prism and ROCmFPX safeguards remain. See
[validation](docs/v5.5.6-validation.md).

The previous **b11105** (`348f853b7`) [audit](docs/llama-b11105-audit.md) covers
42 commits after b11063. The CLI option set is unchanged (415 names / 328
long options); new sampling environment defaults and multi-address `--host`
change help/behaviour, not option names. AutoTuner now emits presence penalty
**even at zero**, preventing an inherited environment value from overriding
the chosen setting. The default remains single-loopback binding.

New settings cover **MiMo-V2.6 Flash/Pro RL** (b11102+, official 1.0/0.95
sampling, source/unit coverage without local weights) and **FastContext 4B
SFT/RL** (local GGUF sampling, 262k ceiling). **Xing4.0** has no mainline
loader; **VoxCPM2 BaseLM** is a Voice Lab TTS component: both are recognized
but ordinary chat launch is blocked. Muse Glimmer's first-token tool-call
parser is fixed in b11100; the existing `--jinja` flag selects it. Old
issue states and local reproductions are recorded separately: stale-closed
MTP/ngram-mod is not a proven fix, and the Qwen vision/DFlash2, DeepSeek-V4.1,
Prism and ROCmFPX safeguards remain. See [validation](docs/v5.5.5-validation.md)
for the two-backend runtime checks and release gates.

The previous **b11063** (`3d82ef62d`) [audit](docs/llama-b11063-audit.md) checks the
21 commits after b11042. No option was added or removed (415 names / 328
long options, all 170 profile/mode commands parse on both backends); the
only new `--help` byte is the timestamped `llama_server: initializing ...`
line that PR #29125 prints before argument parsing, which the manifest
strips before hashing (AutoTuner detects readiness through `/health`, not
log parsing). The behavioural change for a bundled profile is the
**dedicated Ling 3.0 (Bailing V3) chat parser** (PR #28682): the Ling 3.0
template pre-opens `<think>`, so a tool call could arrive before `</think>`
and was swallowed into `reasoning_content`; llama-server now selects the
specialised parser from the template's `<role>` markers whenever `--jinja`
is set, which `ling-3.yaml` has always passed. Live on both b11063 trees the
local Ling-3.0-flash IQ3_M returns `tool_calls` with thinking on (auto and
required tool choice), the profile note and all nine packs now recommend
b11063+ for tool use, and `min_llama_build` stays at b10749. The standard
HIP and Vulkan chain (runtime, tool call, DFlash2, vision, lazy PLE, Gemma 4
MTP, Maple, Nemotron) was repeated on b11063; the vision + DFlash2 request
still fails with HTTP 500 on both backends (upstream #27408), so that gate
now says "verified through b11063", and the DeepSeek-V4.1 block names
b11063 (PR #28696 still a draft). Mainline still ends at ggml type 43: the
PrismML (Bonsai 2) and ROCmFPX (Agnes) files stay fork-only and were
re-refused live on the b11063 trees. The previous **b11042** (`ec9281505`)
[audit](docs/llama-b11042-audit.md) checks the
twelve commits after b11030 (no server, argument, speculative or loader
change: `--help` is byte-identical to b10948–b11030, 415 names / 328 long
options, all 168 profile/mode commands parse on both backends) and repeats
the HIP and Vulkan runtime, tool-call, DFlash2, vision, lazy-PLE, Gemma 4
MTP, Maple and Nemotron runs on the b11042 trees. Its new subject is the
**PrismML Ternary-Bonsai 2 27B** pair (`Ternary-Bonsai-2-27B-PTQ1_0.gguf`,
`-PQ2_0.gguf`, BF16 mmproj): the Ternary/Bonsai recipes now pin the fork
release `prism-b10687-5d80cff` (`2b_b10687_{vulkan,hip}_llama.cpp`), the
new `bonsai-2-27b.yaml` profile carries the official thinking sampling,
`--jinja`, 262k context and the fork requirement, and a metadata-driven gate
(`prism.hadamard.*` in the GGUF, no `prism.hadamard` loader in the selected
llama library) refuses those files on mainline before launch. Live on this
workstation: both packings run fully on the GPU on **HIP** (PQ2_0 is the
fastest), **Vulkan** runs PTQ1_0 on the GPU but has no PQ2_0 kernels (the
weights stay CPU-mapped), mainline b11042 rejects both files and the
previous fork pin (prism-b10660) rejects PTQ1_0. The previous **b11030**
(`bdcbaaf6e`) [audit](docs/llama-b11030-audit.md) repeated the
actual HIP and Vulkan inference, tool-call, multi-GPU DFlash2, vision, lazy
PLE, Gemma 4 MTP and Maple runs on the local trees built by the unchanged
recipes and adds a Nemotron 3 Nano Omni (`nemotron_h_moe`) run. b10977→b11030
(53 commits, 168 files) again touches neither `common/arg.cpp` nor the server
option table: the `--help` text is byte-identical on both backends and to
b10948/b10977 (415 names / 328 long options), so no CLI migration is
required. What changed underneath: the new `hrm_text` architecture (PR
#27625, DFM Mimir 1B, profile `dfm-mimir.yaml`), the Nemotron MTP graph with
optional latent projections (PR #29018, Nemotron 3 Super's MTP block now
loads, gated to b11025+ in AutoTuner) and the Nemotron-H `layer_norm_epsilon`
fallback (PR #28989), Vulkan sparse flash attention for the qwen4exp /
MiniMax-M3 indexer path (PR #28105), the qwen4exp hyper-connection ops (PRs
#28901/#28988), the Vulkan `mul_mat_id` row-id hoisting raised from 256 to
1024 experts (PR #28501, Qwen3.8 Flash-Next's 512 experts leave the slow
path), CUDA/HIP graphs for MTP draft graphs (PR #28549), the `--fit`
auto-context with unified KV now sized for every sequence (PR #28849; AutoTuner
always passes `-c` and `--fit off`, so unaffected), the GGUF data-section
alignment relative to an embedded start offset (PR #28993; plain files are
unchanged and `scanner.py` reads them the same way) and the DeepSeek V3.2/V4
chat parser's message delimiters for server context checkpoints (PR #29008).
Qwen3.8 Flash-Next (qwen4exp, sparse FA + hc ops on Vulkan), Gemma 4 12B + MTP
drafter, Qwen3.8 + DFlash2, Maple TQ2_0 and Nemotron 3 Nano Omni were run
live on both backends for those. The previous **b10977** (`0ecb159c9`)
[audit](docs/llama-b10977-audit.md) covered the CPU work-buffer sizing fix
(PR #28882), the Gemma 4 / Step 3.5 / MiMo2 sliding-window array loaders
(PR #28868), the qwen4exp norm reshape (PR #28896) and the recurrent-state
context reuse check (PR #28749); **b10948** (`5f436dddb`)
[audit](docs/llama-b10948-audit.md) covered the `-j/--json-schema` help
rewording (PR #28736), the empty-schema "any object" default and the
`LOG_JSON` records under `--log-jsonl`.
**Still broken upstream:** Qwen3.5/3.8 (`qwen35`) + DFlash2 + vision fails
image requests with HTTP 500 on b11030 exactly as on b10901–b10977 (PR
#28587's skipped image rows leave a position gap in the recurrent DFlash2
draft memory; PR #28715 in b10906 did not change that, and nothing in
b10930→b11030 touches that path). AutoTuner gates this combination on
**every build from b10896 on**; disable Draft for images or Vision for
text-only DFlash2. The previous **b10930** (`56381e407`)
[audit](docs/llama-b10930-audit.md) established that gate. b10907 (PR #28630) also
stops MTP draft contexts on `deepseek2`/`glm4moe`/`cohere2moe` from allocating
KV for every trunk layer; AutoTuner's plan stays conservative there. The
previous **b10901** (`28ff09582`) [audit](docs/llama-b10901-audit.md) added the
qwen4exp indexer-V reserve note; that reserve is unchanged.

The previous **b10878** (`4850c7727`) source/help audit is recorded in
[`docs/llama-b10878-audit.md`](docs/llama-b10878-audit.md). The old mmap/mlock/
Direct-I/O switches have been removed; AutoTuner uses `--load-mode` and migrates
legacy Extras. Giant lazy tables explicitly use `--lazy-mode on`, because
upstream `auto` is now device-dependent. Failed or partial help probes are not
used to prune commands. The HIP build helper chooses `GGML_CUDA_FA_QUANTS=all`
for new source trees and retains the older option for stable/pinned forks.
External-draft placement, Q8-first KV, text logging and conservative graph
reserves remain unchanged. See the audit for exact live-test boundaries.
AutoTuner still quarantines the
upstream NextN regression in b10741-b10748 and points affected users to b10749+
rather than letting llama-server abort during model or draft-context loading. The following
`llama-server` features are supported (verified against `llama-server --help` /
`tools/server/README.md`; the detailed historical source audit remains below):

## llama.cpp review notes (b9334 – b10679)

### Review b10666 → b10679

Reviewed all **13 upstream commits** from exact tag **b10666** (`4e97ac86e`)
through **b10679** (`50f068fff`). Full source/build evidence is in
[`docs/llama-b10679-audit.md`](docs/llama-b10679-audit.md).

- No emitted `llama-server` option changed; no additional AutoTuner command
  pruning is required.
- b10675 hoists Vulkan MoE row IDs/expert counts and b10677 fixes graph
  reordering across aliased views. Rebuilding is sufficient to receive both.
- b10678 reduces Qwen3.8 Flash Next (`qwen4exp`) graph splits without changing
  its GGUF/CLI/allocation contract, so the measured planner coefficients stand.
- b10679 renames only the C API's lazy-read enum/field to `llama_lazy_mode` /
  `lazy_mode`; `--tensor-read-lazy` is unchanged and is newly accepted by
  `llama-bench`. AutoTuner links no C API, so no launch-code migration is needed.
- Windows Vulkan/HIP outputs are now unambiguous backend siblings, and
  backend-neutral profile hints retain the backend selected in the GUI/TUI.

### Review b10590 → b10666

Reviewed the exact upstream range from `b10590` (`6657ded4`) through `b10666`
(`4e97ac86e`). Full source, command-matrix, model-load, and memory evidence is
in [`docs/llama-b10666-audit.md`](docs/llama-b10666-audit.md).

- **No emitted server option broke.** AutoTuner never used the removed CLI-only
  `-no-cnv`, and b10666 accepts every generated normal-server option. Removed
  legacy draft/ngram spellings are not emitted.
- **Mainline promotions:** DFlash2 merged in b10658 and Qwen3.8 Flash Next
  (`qwen4exp`) in b10660. Both dedicated PR build recipes were removed; the
  qwen4exp loader landed in b10660; the profile now enforces b10666+ so its
  measured QSA allocator coefficients are applicable. DFlash2 preflight accepts b10658+
  while still recognizing an explicitly selected reviewed legacy PR binary.
- **Qwen3.8 Flash Next memory corrected from real loads:** the 67.55 GiB IQ1_S
  GGUF contains a 26.82 GiB `per_layer_token_embd.weight` PLE table that
  llama.cpp keeps in `CPU_Mapped` read-lazy storage. The remaining ~40.74 GiB
  are placement weights. AutoTuner now also includes the second QSA indexer KV
  cache and measured context×ubatch graph buffers. Safe ubatch 64 reduced a 90k
  graph from ~5.46 to ~0.39 GiB device compute and ~17.56 to ~1.10 GiB host
  compute while decode remained above 31 tok/s.
- **New profile coverage:** Qwen3.8 Flash Next is mainline-gated, Nemotron 3.5
  Lightning carries b10665 DSpark settings, and Nanbeige 4.2 3B receives its
  official 256k/sampling profile. No other new architecture was added in this
  exact range.

### Review b10549 → b10590

Reviewed all **41 upstream commits** from exact tag **b10549** (`b2e5e9b2`)
through **b10590** (`6657ded4`). Full command-matrix, source, and real-model
evidence is in [`docs/llama-b10590-audit.md`](docs/llama-b10590-audit.md).

- **No emitted server option broke:** every normal profile and representative
  Vision, OCR, MTP, DFlash, ngram, tools, and diffusion command survived the
  b10590 `--help` capability pass without one removed flag/value.
- **Transparent rebuild benefits:** MTP+embedding context initialization,
  BailingMoE3 DSpark rollback, Dots3-Note language/vision/audio, WebP MTMD,
  stream-aware fit accounting, and Vulkan/CUDA/OpenCL/SYCL fixes require no
  changed AutoTuner command.
- **DFlash2 is not in stock b10590:** Qwen3.8's 81-tensor DFlash2 sidecars need
  open PR #27342; stock b10590 creates the old 58-tensor DFlash graph and was
  reproduced failing with `expected 81, got 58`. v5.2.6 detects this before
  launch, automatically uses a compatible sibling PR build when present,
  selects trained n-max 7/p-min 0.0, accepts the reviewed PR build, and
  originally included a pinned PR build recipe. That obsolete recipe was
  removed in v5.2.9 after DFlash2 merged into b10658. A real PR-build request
  loaded and generated successfully at the time.
- **Qwen3.8 Flash Next preview:** `settings/qwen3_8_flash_next.yaml` matches the
  supplied `qwen4exp` metadata (48 hybrid blocks, full attention every fourth
  block, 512/10 experts, 262,144 native context, official 1.0/0.95/top-k-20
  sampling, and PLE metadata). The profile deliberately does not pretend that
  PLE is generic MTP or that YAML can add a missing model loader: use a
  llama.cpp build/fork that explicitly supports `qwen4exp`; upstream support
  was tracked in issue #27741.

### Review b10441 → b10549

Reviewed all **108 upstream commits** from exact tag **b10441**
(`0177dcc7`) through **b10549** (`b2e5e9b2`). Full source/build evidence is in
[`docs/llama-b10549-audit.md`](docs/llama-b10549-audit.md).

- **Integrated:** b10541 `--mmproj-device` keeps MTMD on the same exact GPU that
  owns AutoTuner's projector VRAM budget, including visibility-remapped
  dual-GPU launches; old binaries lose the complete option/value pair through
  help-based pruning.
- **Current model support:** Kimi-K3's b10448 mainline **text** loader is now
  build-gated without falsely claiming K3 vision; Ling 3.0 Flash/Tiny use the
  native b10460 `bailingmoe3` loader, official sampling, hybrid KV count, and
  integrated/sidecar MTP detection.
- **Automatic rebuild benefits:** DSpark speculator formats/LFM2 targets,
  b10549 LFM2/LFM2MoE tensor split, Granite SWA/MoE SWA metadata, repeated MTMD
  prompt caching, DeepSeek-OCR, Vulkan Q8-KV/FA, HIP, CUDA, Metal, SYCL,
  OpenCL, and server/router fixes need no additional launch controls.
- **Build-number mismatch fixed:** master can legitimately be commits ahead of
  the newest release (`build 10548` while b10545 is still latest). The recipe
  now uses the embedded full-history commit count, accepts only an exact HEAD
  tag, marks untagged builds `_dev_<commit>`, and verifies the compiled
  `--version`; no hard-coded offset or waiting is required.
- **Not promoted:** router preset-only `dedup-cache-models` remains a preset
  concern, not a normal AutoTuner performance control.

### Review b10329 → b10441

Reviewed all **112 upstream commits** from exact tag **b10329**
(`18f7ad7f`) through **b10441** (`0177dcc7`). The source, official-package,
backend, architecture, and memory-accounting evidence is documented in
[`docs/llama-b10441-audit.md`](docs/llama-b10441-audit.md).

- **No emitted flag broke:** the complete AutoTuner server surface remains in
  b10441. `--load-mode auto` is now the upstream default; the new
  value-bearing `--reasoning-effort` is safe through Extra CLI flags.
- **Backend identity is now exact:** `CUDA`, `HIP`, `Vulkan`, `SYCL`, `Metal`,
  and `OPENVINO` device prefixes from the selected binary are retained.
  CUDA/SYCL no longer inherit Vulkan indices or selectors.
- **Unified memory is one pool:** Apple Silicon and confirmed integrated GPUs
  use live available memory, with CPU/GPU allocations counted once. Full-GPU
  KV no longer receives an impossible host-RAM supplement.
- **Architecture updates:** existing Muse Glimmer and Granite Switch profiles
  cover their new native loaders; MiniMax-Text-01/MiniMax-M1 gains a
  `minimax-01` hybrid-MoE profile; PocketTTS remains a dedicated TTS workflow,
  not a normal text-chat claim.
- **Official packages expanded:** b10441 publishes Windows x64 Vulkan, ROCm,
  SYCL, OpenVINO, CUDA 12.4 and CUDA 13.3 builds, plus native macOS arm64 and
  Ubuntu backend packages. AutoTuner launches these external binaries rather
  than linking to one backend.

### Review b10151 → b10329

Reviewed all **178 upstream commits** from exact tag **b10151**
(`8e8681e0`) through **b10329** (`18f7ad7f`). The complete evidence and
scope decisions are documented in
[`docs/llama-b10329-audit.md`](docs/llama-b10329-audit.md).

- **Integrated:** DSpark sidecar/tensor detection and `draft-dspark` build
  gating; Unlimited-OCR's b10285 multi-row batching + b10287 32-tile fix;
  deterministic OCR profile; complete GUI/TUI PDF/Office/image workflow;
  b10329 value-bearing flag handling including `--tools-runtime`; explicit
  Flash Attention off; F16 KV profile opt-in; selected-server alias/process
  verification before any document upload.
- **Automatic benefits:** new speculative counters appear in the already-enabled
  `/metrics`; Qwen3-Next/DeepSeek V3.2/GLM MTP loaders, EAGLE-3 v3, model/router,
  Vulkan/ROCm/Metal/CUDA/SYCL, tokenizer, and MTMD fixes require only a rebuilt
  llama.cpp binary.
- **Not automatic:** Docker tool isolation and MCP configs remain trusted Extra
  CLI choices. The separate `llama-tts` breaking changes do not affect
  AutoTuner's server runners. b10329 only announces a *future* upstream port
  change; AutoTuner continues to pass explicit port 1234.

### Review b10107 → b10151

Reviewed all **44 upstream commits** from tag **b10107** (`c0bc859`) through
**b10151** (`8e8681e`). The only new/expanded launcher inputs are the split
`--load-mode` semantics and two experimental stdio-MCP configuration flags.

- **Integrated:** Expert model-load dropdown for `none`, `mmap`, non-mmap
  `mlock`, `mmap+mlock`, and `dio`; compatibility adaptation for pre-b10151
  binaries; current-build GPU locking no longer hits the old unconditional
  veto.
- **Not promoted to tuning controls:** `--mcp-servers-config PATH` and
  `--mcp-servers-json JSON` spawn external processes and change CORS behavior.
  They are server integrations rather than performance knobs, remain available
  through **Extra CLI flags**, and should only be used with trusted configs.
- **Automatic upstream benefits:** explicit `-md` now wins over discovered
  draft sidecars; reasoning budgets recognize multiple end sequences;
  MiniMax-M3/GLM indexer and backend/KV fixes require no new AutoTuner setting.
- **No change required:** KV types/cache sizing, context/batching, GPU-layer
  offload, tensor split, and speculative token-count flags did not gain new
  controls or incompatible defaults in this range.

### Review b9963 → b10056

Reviewed all **93 upstream commits** through tag **b10056** (`b85833e`).
No AutoTuner-emitted server flag was removed, renamed, or changed incompatibly.
Changes integrated in this review:

- **Vision prompt caching:** current mtmd state handling can reuse repeated
  image prompts. AutoTuner enables `--cache-ram` for b10045+ and keeps older
  or unprobeable builds on the safe `--cache-ram 0` path. A real b10058
  Gemma 4 + mmproj test returned `cached_tokens` 0 → 279 and reduced the
  repeated request from 3.13 s to 0.30 s.
- **`/slots` toggle fixed:** b10056 defaults `/slots` on, so AutoTuner now
  emits `--slots` or `--no-slots` explicitly instead of treating omission as
  off.
- **Reasoning history:** `--reasoning-preserve` is available in the Expert
  panel and persists with the other per-model Expert settings.
- **Hy3/Hy-MT2:** `hy_v3` + MTP support merged in b9993 (PR #25395), so the
  profile no longer tells users to select a PR fork.
- **New optional upstream surface:** `--cors-origins`, `--cors-methods`,
  `--cors-headers`, and `--cors-credentials` landed in b10010. They remain
  available through Extra CLI flags; a dedicated four-field UI is unnecessary
  while AutoTuner binds `127.0.0.1` by default.

Transparent rebuild benefits include Minimax2 EAGLE-3 support, Vulkan native
MXFP4/NVFP4 conversions, prompt-cache/checkpoint fixes, DeepSeek V4 graph
optimisations, mtmd fixes, and CUDA/HIP/SYCL backend improvements.

### Review b9840 → b9888

Reviewed mainline up to **b9888** (`cb295bf`, CUDA FlashAttention K/V cache-type validation). No AutoTuner flag was removed or renamed upstream. Changes made for v4.7.9:

- **Terminal throughput visibility asserted:** AutoTuner emits `--perf` for normal `llama-server`, `llama-diffusion-cli`, and `llama-diffusion-gemma-server`, so fork defaults cannot hide prompt/eval timings and tokens/s. Current mainline defaults these timings on; `--metrics` remains enabled for machine-readable monitoring.
- **NVIDIA CUDA safety:** b9888 validates V-cache types for CUDA FlashAttention too. Since default CUDA builds have `GGML_CUDA_FA_ALL_QUANTS=OFF`, AutoTuner keeps automatic KV choices symmetric on NVIDIA (high- and low-VRAM) while preserving AMD/Vulkan asymmetric K/V choices for extra context. Expert-mode manual K/V pins still pass through unchanged.
- **Tracked settings removed:** `autotuner_settings.json` is now only local user state (already gitignored) and is removed from Git tracking for GitHub releases.

Relevant upstream commits in this range are backend/runtime fixes (CUDA Gemma E4B MTP FA, stale tensor-split params for draft models, tensor-parallel + `--n-cpu-moe`, Vulkan integer overflow, UI/MCP fixes). They do not require new AutoTuner flags beyond the `--perf` verbosity fix above.

### Review b9625 → b9840

Reviewed the range up to **b9840**. Every server flag and `--spec-type`
value in use was verified against the current `tools/server/README.md`
(`--help` table) and `docs/speculative.md`. **No existing flag was removed
or renamed** — the AutoTuner's flag surface is unchanged and still valid.
Changes this round are AutoTuner-side additions and fork-build fixes:

- **EAGLE-3 speculative decoding (PR #18039; Qwen3.5/3.6 since PR #24593 /
  b9723).** `--spec-type draft-eagle3` is now emitted automatically when the
  paired drafter GGUF declares `general.architecture = eagle3` (a one-layer
  transformer that reads the target's hidden states — higher acceptance
  than a plain draft of the same size). Sibling files named `*-eagle3*` are
  auto-paired like any draft; `scanner.py` reclassifies an `eagle3`-arch GGUF
  into the draft pool (never listed as a choosable model).
- **DFlash speculative decoding (PR #22105).** `--spec-type draft-dflash` is
  emitted automatically when the paired drafter declares
  `general.architecture = dflash` (block-diffusion; emits a whole block per
  step). If auto-pairing misses a custom filename, pick the DFlash GGUF in
  the GUI's **draft** dropdown; it is labelled `[DFlash]` and remembered per
  model.
- **Fork discovery hardened.** Versioned fork dirs (`2b_b8840_llama.cpp`,
  `tq_b9632_llama.cpp`, …) now resolve correctly — a profile hint like
  `2b_llama/llama-server` matches the on-disk `2b_b8840_llama.cpp` after
  normalizing the `_b<NUM>` version segment. The 1-bit (`1b_`) and
  2-bit/Ternary (`2b_`) Bonsai families stay distinct. Forks that match the
  name pattern but have no built `llama-server` binary are now reported in
  the `llama_cpp` debug category (instead of vanishing silently), and the
  terminal launcher now finds `L:/LAB/ai-local` (the documented workspace)
  even without `LLAMA_CPP_DIR` set.
- **`bonsai-ternary.yaml`** corrected: `server_binary` now points to
  `2b_llama` (2-bit/Ternary fork), not `1b_llama` (1-bit Bonsai).
- **Build scripts** (historically `*_build.txt`, now executable `*.ps1`) probe BOTH UI layouts — pre-b9174
  `tools/server/webui/` and post-b9174 `tools/ui/` — and fall back to the
  HF prebuilt UI when neither exists. The Bonsai (b8840-basis) build adds
  `-DLLAMA_OPENSSL=OFF` to work around the cpp-httplib 0.40.0 / OpenSSL 3.2+
  `C2440` const error.

Everything else AutoTuner emits is **unchanged and still valid at b9840**:
`--fit [on|off]`, `-fa [on|off|auto]`, `--cache-ram`/`-cram`, `--metrics`,
`--n-cpu-moe`/`-ncmoe`, `--tensor-split`, `--main-gpu`, YaRN, KV-cache types,
`--reasoning`/`-rea`, `--reasoning-budget`, `--chat-template-kwargs`,
`--jinja`, `--mlock`/`--no-mmap`, and the full speculative set.

### Review b9500 → b9625

Reviewed the range up to **b9625**. Every server flag and `--spec-type`
value in use was verified against the b9625 `common/arg.cpp` and
`common/speculative.cpp`. **One breaking change affected the AutoTuner and
is fixed this round:**

- **`--think-budget` renamed to `--reasoning-budget` (CLI).** At b9625 the
  reasoning token-budget flag is `{"--reasoning-budget"} "N"` (`-1`
  unrestricted / `0` immediate end / `N>0` budget); the old `--think-budget`
  spelling is **gone — not kept as an alias** (the env var stays
  `LLAMA_ARG_THINK_BUDGET`, and the short reasoning toggle gained a `-rea`
  alias). The Expert panel's spin-box now emits `--reasoning-budget`, and
  `_parse_reasoning_from_extras` reads **both** the new and the legacy name
  so older `autotuner_settings.json` files still restore the spin-box
  correctly. A sibling `--reasoning-budget-message MESSAGE` was also added
  (text injected before the end-of-thinking tag when the budget is
  exhausted) — not emitted by AutoTuner.

Everything else AutoTuner emits is **unchanged and still valid at b9625**,
re-confirmed against the source: `--fit [on|off]`, `-fa [on|off|auto]`,
`--cache-ram`/`-cram` (`-1` no-limit / `0` disable), `--metrics`,
`--n-cpu-moe`/`-ncmoe`, `--tensor-split`, `--main-gpu`, `--rope-scaling
yarn` + `--rope-scale`, `--numa`, `--mlock`/`--no-mmap`,
`--no-context-shift`, `--parallel`, `--jinja`, `--reasoning`/`-rea`,
`--chat-template-kwargs`, `--mmproj`, the sampler flags, and the full
speculative set — `--spec-type` with the tokens `draft-mtp`, `ngram-mod`,
`ngram-map-k`, `ngram-map-k4v`, `ngram-simple`, `ngram-cache`, plus
`--spec-draft-ngl/-n-max/-p-min`, `--spec-ngram-mod-n-match/-n-min/-n-max`,
and `--spec-ngram-map-k4v-size-n/-size-m/-min-hits`. The `-md` external
drafter still enables the draft path without an explicit `--spec-type`.



Full review of all 58 commits between b9442 and b9500. **Result: no
functional AutoTuner changes to existing flags needed** — every server flag
and `--spec-type` value in use was verified against the b9500
`common/arg.cpp` and `common/speculative.cpp` and is unchanged and still
valid. Specifically re-confirmed present at b9500: `--spec-type`,
`--spec-draft-ngl/-n-max/-p-min`, `--spec-ngram-mod-n-match/-n-min/-n-max`,
`--spec-ngram-map-k4v-size-n/-size-m/-min-hits`, and the spec-type tokens
`draft-mtp`, `ngram-mod`, `ngram-map-k`, `ngram-map-k4v`, `ngram-simple`,
`ngram-cache`. Relevant points:

- **Speculative: `draft-simple` auto-enable removed (#23988).** The server
  no longer auto-enables a `draft-simple` path; the `common/arg.cpp` diff was
  whitespace-only (no flag renamed/removed). **No impact** — AutoTuner always
  emits `--spec-type` explicitly and never relied on auto-enabling. A new
  `draft-eagle3` spec-type also exists now (EAGLE3 drafters); AutoTuner now
  emits it when an `eagle3`-arch drafter is paired (see b9625→b9840 review
  below).
  AutoTuner.
- **Gemma 4 "unified" runtime + 12B (#24077, #24082, #24088, #24025).**
  Vision/audio (mtmd) fixes for the encoder-free "unified" Gemma 4 and the
  new arch enums `gemma4uv`/`gemma4ua`. The **12B `gemma-4-12b-it` is the
  unified variant**, but its language-model GGUF still loads under
  `general.architecture` = **`gemma4`** (verified in the b9500 converter:
  `Gemma4UnifiedModel` → `MODEL_ARCH.GEMMA4`); `gemma4uv`/`gemma4ua` are only
  projector types on the separate mmproj file. → scanner, KV-sizing and
  `match_profile` treat the 12B exactly like the rest of the family. The
  `gemma-4.yaml` profile was updated for accuracy (12B added to the
  context-tier comment and the multimodal/audio notes; throughput-vs-dense
  caveat clarified) — no code change required for it to work.
- **Qwen3.5 MTP post-norm (#24025).** Qwen35 now uses the post-norm hidden
  state for MTP, internal rename `pre_norm` → `nextn`. Runtime correction
  only; no CLI flag or metadata-key change → the tri-state MTP scanner over
  `{arch}.nextn_predict_layers` stays valid.
- **New architectures (profiles added this round).** `mellum` (JetBrains
  Mellum2-12B-A2.5B, MoE, #23966), `exaone4` (EXAONE 4.5 33B VLM, #21733),
  `step35` (StepFun Step 3.5 + Step 3.7-Flash, MoE+MTP-3, #23274/#23845), and
  `modern-bert` (IBM Granite Embedding Multilingual R2 97m/311m, #22716). See
  *Adding profiles for new models* — these are profile additions, not forced
  by any flag rename. The arch is read dynamically from the GGUF metadata.
- **Vulkan performance (transparent).** Device mutex no longer held while
  compiling pipelines (#23641), reduced host-memory lock contention (#23376),
  Q3_K/Q6_K block-load on 32-bit ints (#23056). Benefits from the rebuild
  alone, no flag change; relevant to the Vulkan backend on the R9700 /
  RX 9070 XT (faster server start / pipeline warmup).
- **Library API (not CLI).** `llama_set_warmup` deprecated (#24009),
  `llama_context` max-outputs limited (#23861), CUDA reserves quantized-KV
  space at startup (#23907). No effect on the flags AutoTuner emits.

**Scanner fix shipped this round (AutoTuner-side):** `scanner.py`'s
`_ROPE_SCALE_SUPPORTED_ARCHS` matched only the `qwen2` prefix, so the newer
`qwen3*` / `qwen35*` arch strings (Qwen3/3.5/3.6) fell through and were
excluded from automatic YaRN (they could only get RoPE-scaling via an
explicit `rope_scale.enabled: true` in the profile). Broadened the prefix
`qwen2` → `qwen` (matched via `startswith`, so it now covers
`qwen`/`qwen2*`/`qwen3*`/`qwen35*` and stays correct for future Qwen archs).

### Review b9409 → b9442

Reviewed the releases up to **b9442** (`d4c8e2c`, a vocab/tokenizer commit
adding jina-embeddings-v2-base-zh). **Result: no functional AutoTuner
changes to existing flags needed** — every server flag in use (`--fit off`,
`--metrics`, `--cache-ram`, `--spec-type` with `draft-mtp` / `ngram-mod` /
`ngram-map-k4v`, `--spec-draft-*`, `-fa on`, `--n-cpu-moe`, `--tensor-split`,
YaRN, KV-cache types) was verified against the b9442 `common/arg.cpp` and
`common/speculative.cpp` and is unchanged and still valid. The changes added
in this round are AutoTuner-side, not forced by any flag rename:

- **mmproj detection** now matches the `mmproj` marker **anywhere** in a
  filename and also picks up `.mmproj`-extension projectors, so the
  MXFP4 MoE pair (`…-mxfp4-moe-mmproj-f16.gguf`) is paired correctly.
- **GGUF `general.sampling.*`** is now read and used to fill any sampler
  value a matched profile leaves unspecified — fixing repetition loops and
  broken tool-calls on models without a tailored profile.
- **MoE multi-GPU spread** switched from priority-weighting to
  **capacity-fill**, so both GPUs are packed with expert layers instead of
  stranding VRAM on the secondary card.

### Review b9371 → b9409

Reviewed the releases up to **b9409** (`fe12e42`, a pure `sync : ggml`
commit). **Result: no functional AutoTuner changes to existing flags
needed** — all server flags in use (`--fit off`, `--metrics`,
`--cache-ram`, `--spec-type`, `--spec-draft-*`, `-fa on`, `--n-cpu-moe`,
YaRN, KV-cache types) were unchanged and still valid. Added this round (not
forced by a b9409 flag rename, but as a feature):

- **`--cache-ram` prompt caching** is now actively emitted (previously not
  at all). At the time this review was written it was conservatively limited
  to non-vision models; the b10056 review above adds build-gated Vision support.
- **Multi-server port assignment** (1234, 1235, … with a reset on exit) and
  **live-VRAM load-balancing** onto the emptier GPU before starting a
  second/third model — purely GUI/launcher-side, no new server flags.

### Review b9334 → b9371 (37 commits)

Full review of all 37 commits between b9334 and b9371. **Result: no
functional AutoTuner changes needed** — all server flags in use (`--fit
off`, `--metrics`, `--spec-type`, `--spec-draft-*`, `-fa on`, `--n-cpu-moe`,
YaRN, KV-cache types) were unchanged. Relevant points:

- **Env rename (#23778):** llama.cpp moved several environment variables to
  the unified `LLAMA_ARG_` prefix: `LLAMA_LOG_FILE/COLORS/VERBOSITY/PREFIX/TIMESTAMPS`
  → `LLAMA_ARG_LOG_*`, `LLAMA_OFFLINE` → `LLAMA_ARG_OFFLINE`,
  `LLAMA_CHAT_TEMPLATE_KWARGS` → `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`. **The CLI
  flags themselves stay the same.** AutoTuner sets only
  `HIP_VISIBLE_DEVICES` / `GGML_VK_VISIBLE_DEVICES` as env overrides (GGML
  vars, not affected); `LLAMA_ARG_FIT` already carries the prefix → no impact.
  ⚠️ *If llama log/offline env overrides are added in the future, use the
  `LLAMA_ARG_` prefix from b9371 on.*
- **Vulkan performance:** several transparent backend optimisations
  (MUL_MAT_VEC 4 K/iteration for F16/F32 #22887, conv2d + coopmat1 #22620,
  REPEAT f16→f16 #23298). Benefits from the rebuild alone, no flag change.
  The AMD UMA transfer-queue fix (#22455) affects only integrated GPUs/APUs,
  not the dedicated R9700 / RX 9070 XT.
- **New model/conversion support (convert-side, not server runtime):**
  `Gemma4ForCausalLM` conversion (#23682), MiniCPM5 tokenizer (#23384),
  talkie-1930-13b (#22596), Mistral3-NVFP4 weight scales (#23629). Profile
  maintenance only on adoption — the arch is read dynamically from the
  metadata in the tuner.
- **Server code:** cosmetic only (SSL log message #23393, cpp-httplib 0.46.0 #23650).
