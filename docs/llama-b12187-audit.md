# llama.cpp b12187 (0.9.0-dev) audit — AutoTuner v5.6.4

Checked 2026-11-07 against the locally built Windows **b12187** HIP and
Vulkan servers, commit `cf659f240` (`0.9.0-dev`), on Core Ultra 9 285K, AI
PRO R9700 32 GB + RX 9070 XT 16 GB. Both trees were built by the user with
the repository recipes. The
[b11669…b12187 comparison](https://github.com/ggml-org/llama.cpp/compare/b11669...b12187)
contains **518 commits**; it spans two stable releases, **v0.8.0 = b11960**
and **v0.9.0 = b12110**. See the
[binary/CLI manifest](llama-b12187-server-flags.json) and the
[validation record](v5.6.4-validation.md).

## CLI and planning contracts

Both backends expose **423 option names / 336 long options**: the b11669 set
plus three options; nothing was removed and no default printed by `--help`
changed. All profile Extra CLI flags are still advertised (including the new
`--reasoning off` in two profiles). Device order is unchanged. AutoTuner still
owns context/offload placement and sends `--fit off`.

- **`--connect` / `--connect-code CODE`** ([#28277](https://github.com/ggml-org/llama.cpp/pull/28277),
  b11744) open an experimental peer-to-peer tunnel to the server from outside
  the LAN through the separate `llama-connect` tool (CMake option
  `LLAMA_CONNECT`, default OFF, which downloads a prebuilt binary). AutoTuner
  never sends them. `--connect-code` is a value flag now (pruned with its
  value on older builds) and, because the code grants access to the server,
  it is redacted like `--api-key` in the GUI launch log, the OCR workflow log
  and the diagnostics report.
- **`--models-memory-margin N`** ([#21231](https://github.com/ggml-org/llama.cpp/pull/21231),
  b11930) applies only to the router server (`--models-dir`), which AutoTuner
  does not use; known as a value flag.
- **`--model` or `-hf` is required** ([#31242](https://github.com/ggml-org/llama.cpp/pull/31242),
  b11929): AutoTuner always passes `-m`.
- **Windows argv** ([#31417](https://github.com/ggml-org/llama.cpp/pull/31417),
  b12014) dropped llama.cpp's own reconstruction of the Windows command line.
  A model path with umlauts and CJK characters
  (`Modelle-äöü\Modell-ß-测试.gguf`) and a German prompt tokenize identically
  on b11669 and b12187, HIP and Vulkan, when passed the way AutoTuner starts
  the server (argument list, `CreateProcessW`).
- **`-sm tensor`** is refused for architectures with MoE amax rescaling
  ([#31081](https://github.com/ggml-org/llama.cpp/pull/31081)); AutoTuner
  never sends `-sm`.
- **`--fit` and NextN** ([#31569](https://github.com/ggml-org/llama.cpp/pull/31569),
  b12136): the loader counts NextN layers in `-ngl`, so `--fit on` used to
  leave layer 0 on the CPU. AutoTuner sends `--fit off` and its own `-ngl`;
  every full-offload run of the live chain reports all layers offloaded
  (for example 66/66 for Qwen3.8-27B with its NextN block).
- **ggml type 43 = `F8_E4M3`** ([#27512](https://github.com/ggml-org/llama.cpp/pull/27512),
  b12000, file type 42). Tensor sizes come from offsets, so the scanner needs
  no table change; `F8_E4M3`, `FP8` and `NVFP4` are now recognised as quant
  tags when pairing mmproj and draft files. The ROCmFPX gate is unaffected
  (mainline now rejects `type 100` with `should be in [0, 44)`).

## Models that run on mainline now

### Xing 4.0 (was blocked)

[#31589](https://github.com/ggml-org/llama.cpp/pull/31589) (b12156) adds the
`xing4_0` loader with MLA, the DeepSeek-V4 hyper-connection operators, MoE and
the MTP head. AutoTuner had blocked the architecture since v5.5.5. The block
is now a build floor (b12156) and the profile is runnable. The b11669 trees
still stop with "unknown model architecture"; AutoTuner names b12156 there.

A first direct test on the R9700 (Vulkan, 8K context, one 400-token prompt,
llama.cpp's default p-min) found one draft token best: 61.9 tok/s without
MTP, 76.5 with one, 74.7 with two and 53.7 with three draft tokens. Through
AutoTuner's own plan (262K context) the usual confidence gate p-min 0.75
removed most of that gain, so the mirrored A/B (nospec, p-min 0.75, p-min 0,
twice each, 400 tokens, temperature 0) decided the profile:

| Backend | Prompt | No MTP | MTP, p-min 0.75 | MTP, p-min 0 |
|---|---|---:|---:|---:|
| Vulkan | code | 82.1 | 81.5 | **91.7** |
| Vulkan | German prose | 82.0 | 78.0 | **89.9** |
| HIP | code | 64.5 | 75.8 | **77.1** |
| HIP | German prose | 64.5 | 67.6 | **74.2** |

The profile uses the embedded MTP head with one draft token and no gate
(`draft_max: 1`, `draft_p_min: 0.0`): +10–12 % on Vulkan, +15–20 % on HIP.
Math, tool call and thinking checks pass on both backends.

### DeepSeek-V4.1-Flash (was blocked)

[#31086](https://github.com/ggml-org/llama.cpp/pull/31086) (b11870) adds the
`deepseek41` architecture: its own DSv4 KV cache for CSA2 (Full / Reindex /
Reuse layers sharing KV), the Engram table, single-pass mHC, the
`deepseek41v` image projector and a V4.1 chat renderer. It has no MTP graph;
DSpark speculation uses a separate DFLASH draft (`draft-dspark`). The
conversion-only PR #28696 that AutoTuner's block referred to was closed.

The profile is now runnable from b11870 and claims its architecture
authoritatively (`arch_authoritative: true`), so a V4.1 file named
"DeepSeek-V4-…" cannot fall into the V4 profile. GGUFs of the old
conversion layout (`deepseek_v41*`, or `deepseek4` with Engram keys) stay
refused with a message that names b11870. AutoTuner budgets KV for every
layer, which overestimates the shared CSA2 cache; no V4.1 GGUF fits this
test system (552B backbone), so the plan is unmeasured.

### Kolibri 1

[#30673](https://github.com/ggml-org/llama.cpp/pull/30673) (b11709) adds the
`kolibri1` loader. The profile keeps its runtime-marker check instead of a
build floor because the hob-llama.cpp fork that served the model first
reports b11434; mainline b11709+ contains the marker and passes.

The local `Kolibri-1-Q4_K_M.gguf` was converted with that hob patch and did
**not** load on mainline: `unknown pre-tokenizer type: 'kolibri1'`. The patch
registers the pre-tokenizer name `kolibri1` as an alias of the qwen2 split
regex and writes expert gating type 5 (`sigmoid_logit_add`: select on logits
+ bias, weight with the sigmoid of the logits); mainline's converter writes
`qwen2` and gating type 2, and mainline's kolibri1 graph performs that
selection itself. For an `arch kolibri1` GGUF with either hob value, AutoTuner
now adds `--override-kv tokenizer.ggml.pre=str:qwen2,kolibri1.expert_gating_func=int:2`
on mainline b11709+ (never on the hob runtime, never on mainline
conversions). Through AutoTuner's plan on b12187 (both GPUs plus 3 CPU expert
layers, 197K context) the file answers correctly: math, tool call and thinking
pass, **67.5 tok/s on Vulkan and 63.0 on HIP**. (llama.cpp warns that a
repeated `--override-kv` keeps only the last value; the handler appends, so
both work, but AutoTuner sends one comma-separated value anyway.)

## New architecture, decision-type and projector floors

A new table `_ARCH_MIN_BUILD` names the first build for every architecture
added after b11669; an older build would stop with "unknown model
architecture". The table is checked from GGUF metadata, independent of the
filename.

| Architecture | Build | PR |
|---|---|---|
| `muse-spark` (Muse Spark Open) | b11671 | #30616 |
| `beam` (Reflection Beam 1.0) | b11701 | #30670 |
| `sensenova_u1` | b11801 | #28919 |
| `flint` (Recursive Flint) | b11803 | #30928 |
| `deepseek41` | b11870 | #31086 |
| `lfm3` (LiquidAI LFM3) | b11938 | #31252 |
| `limite` (Limite 1B Violetto) | b11988 | #29433 |
| `mistral-large4` | b11998 | #31376 |
| `clover_moe` (Clover 35B-A3B) | b12060 | #31306 |
| `metis16` (Metis 1.6) | b12089 | #31134 |
| `xing4_0` | b12156 | #31589 |
| `ricursive` (Ripple 65B) | b12158 | #31692 |
| `kimi-k31` (Kimi K3.1) | b12180 | #31779 |

Decision types (`<arch>.decision.type`): `granite-switch` b11707
(#30676/#30693, 256 options), `h2o-lightning` b11773 (#30361),
`openbmb-minicpm-ocr-jev` b11928 (#30980), `lfm3-d1` and `lfm3-d1-omni`
b11945 (#31275). None of them needs the one-batch rule; MiniCPM-OCR-Jev's PR
shows identical scores with `-ub 128`. Granite Switch 4.1 is a chat model
with a decision adapter; llama.cpp keeps its prompt cache off like for every
decision type, and AutoTuner follows that.

Multimodal projectors (`clip.projector_type`; the message now says
"multimodal projector" because several are audio): `muse-spark-audio`
b11671, `voxtral_rt` b11675, `sensenova_u1` b11801, `minicpm_ocr_1_0` b11826,
`soprano` b11850, `deepseek41v` b11870, `lfm3vl` b11938, `lfm3a` /
`lfm3a_gen` b11941, `nemotron_transcribe` b12087, `chatterbox` /
`chatterbox_spkenc` b12097, `drama3_gen` b12116.

## New profiles

All of these fell back to `_default.yaml` (8K context, no `--jinja`) or, for
Kimi K3.1, to the K3 profile with the b10749 floor. Sampling comes from the
model cards or the shipped generation config.

| Profile | Models | Floor | Sampling (chat) |
|---|---|---|---|
| `kimi-k3_1.yaml` | Kimi K3.1 (Pro), 2.8T-A104B, 1M | b12180 | 1.0 / top-p 0.95 |
| `lfm3.yaml` | LFM3-1.7B, LFM3-VL-2B, LFM3-Audio-2B, LFM3-Omni-2B | b11938 (audio b11941) | 0.1 / top-k 50 / rep 1.1 |
| `muse-spark-open.yaml` | Muse Spark Open 816B-A42B, text/image/audio, 1M | b11671 | 1.0 / top-p 1.0 / top-k 64 |
| `mistral-large-4.yaml` | Mistral Large 4 1T-A52B, image, 1M | b11998 | 1.0 / top-p 0.95 |
| `clover.yaml` | DiffuSpace Clover-35B-A3B-Preview (AR mode), 128K | b12060 | 1.0 / top-p 1.0 / presence 0.9 / rep 1.05 |
| `ripple.yaml` | Ricursive Ripple-65B Preview, 1M | b12158 | llama.cpp defaults (none published) |
| `limite.yaml` | Limite 1B Violetto (fixed system prompt) | b11988 | 0.6 / top-p 0.95 |
| `flint.yaml` | Recursive Flint 40B-A2.5B, 256K | b11803 | 1.0 / top-p 0.95 / top-k 20 |
| `minicpm-ocr-1_0.yaml` | MiniCPM-OCR 1.0 (`--reasoning off`) | b11826 | greedy |
| `nemotron-3_5-transcribe.yaml` | Nemotron 3.5 Transcribe 30B-A4B (`--reasoning off`), 64K | b12087 | greedy |
| `nemotron-embed.yaml` | Nemotron-3 / 3.5-Embed 1B / 8B (`--embeddings`) | b11833 | — |

The decision profile now also matches LiquidAI d1-2B, H2O-Lightning-4B and
MiniCPM-OCR-Jev. MiniCPM-OCR 1.0 and Nemotron 3.5 Transcribe write their
answer into the thinking block when reasoning is on (both PRs say so), hence
`--reasoning off`. No profile for Metis 1.6 (weights announced, no model card
found), Beam 1.0 (weights not public) and SenseNova U1.5 (only community
GGUFs, mainly an image-generation model); their architectures are covered by
the floor table. The TTS additions (Chatterbox, Drama-3, Soprano, Pocket TTS)
are served by `llama-tts` / `POST /tts` and are not chat profiles.

Hybrid/MoE tables: `lfm3`, `metis16`, `clover_moe`, `kimi-k31` and
`deepseek41` follow upstream's `llm_arch_is_hybrid`; Kimi K3.1 uses the K3
ratio (24 gated-DSA of 93 layers), Clover writes `attention.recurrent_layers`,
DeepSeek-V4.1 keeps full-layer KV accounting like V4. The MoE fallback list
knows the new MoE architectures for GGUFs without `expert_count`.

## Local models added since b11669

Every new local GGUF ran through AutoTuner's own plan on both backends with
a math answer (reasoning separated), a tool call, a 400-token decode and,
with an mmproj, three image-colour questions. All passed:

| Model | Profile | Vulkan decode | HIP decode |
|---|---|---:|---:|
| Xing 4.0 29B-A4B IQ4_NL (+ MTP) | `xing-4_0.yaml` | 82.8 | 72.8 |
| Kolibri 1 Q4_K_M (hob GGUF, overrides) | `kolibri-1.yaml` | 67.5 | 63.0 |
| Nemotron 3.5 Bolt UD-Q3_K_XL + mmproj | `nemotron-3_5-bolt.yaml` | 45.4 | 35.3 |
| MiniCPM-V 4.7 35B-A3B Q6_K + mmproj | `minicpm-v-4_7.yaml` | 118.2 | 85.0 |
| LFM2.5-24B-A2B Q6_K | `lfm2.yaml` | 196.1 | 157.9 |
| Underdog-Saluki-27B IQ2 mix + mmproj | `qwen3_5-3_6.yaml` | 44.3 | 41.0 |

(Xing row: plan default at that time, p-min 0.75; see the A/B above.)
`Underdog-Saluki-27B-1.0-IQ2-mix.gguf` is a renamed ISTA-DASLab GSQ quant of
Qwen3.8-27B (`general.basename`); by file name it uses the Qwen3.5/3.6
profile, whose chat sampling equals the Qwen3.8 profile's (1.0 / top-k 20 / top-p 0.95).
Profile matching stays filename-first.

## Pairing fix found during the audit

Recognising `NVFP4` as a quant tag shortened the base name of
`dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf` enough that the
half-prefix rule bound it to `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-…` in the
same folder ("nvidia-nemotron-3" is shared). A DFlash/DSpark/MTP head is
trained for one target, so a draft whose size label differs from the
target's (`74b-a9b` vs `30b-a3b`) is no longer paired or marked compatible;
names without a size label are not judged. All 86 local models keep exactly
the mmproj and draft they had at HEAD.

## Other upstream changes checked

- **Speculative decoding:** probabilistic DFlash2 drafts
  ([#30663](https://github.com/ggml-org/llama.cpp/pull/30663), b11834;
  +0–4 % at temperature > 0 in the PR, unchanged at 0), MTP hidden state
  saved/restored with checkpoints ([#30141](https://github.com/ggml-org/llama.cpp/pull/30141)),
  checkpoint restore for drafts longer than `n_rs_seq`
  ([#31350](https://github.com/ggml-org/llama.cpp/pull/31350)), logprobs
  with speculation ([#27196](https://github.com/ggml-org/llama.cpp/pull/27196)),
  Gemma 4 assistant follows the target's devices and FA setting
  ([#31635](https://github.com/ggml-org/llama.cpp/pull/31635)). AutoTuner's
  draft flags stay as they are; the draft paths pass the live chain.
- **Vulkan on RDNA4:** large-tile split-K ([#30280](https://github.com/ggml-org/llama.cpp/pull/30280)),
  small-n split-K ([#30146](https://github.com/ggml-org/llama.cpp/pull/30146)),
  ReBAR detection ([#31658](https://github.com/ggml-org/llama.cpp/pull/31658)),
  large coopmat matmul re-enabled on AMD driver 26.10.1+
  ([#31587](https://github.com/ggml-org/llama.cpp/pull/31587)).
- **Gemma 4:** graph optimisations for E2B/E4B prefill
  ([#31356](https://github.com/ggml-org/llama.cpp/pull/31356)), images resized
  without letterbox ([#30922](https://github.com/ggml-org/llama.cpp/pull/30922)).
- **MoE cache:** SLRU and then LRFU/MQ eviction
  ([#30637](https://github.com/ggml-org/llama.cpp/pull/30637),
  [#31233](https://github.com/ggml-org/llama.cpp/pull/31233)).
  `--moe-cache-mib` stays opt-in (v5.6.2 measurement); not re-measured.
- **Server:** realtime conversation/transcription and `POST /tts` endpoints,
  reasoning-effort levels exposed from the chat template, typed rerank input.
  Not used by AutoTuner.
- **Build:** the only new CMake options are `LLAMA_CONNECT` (OFF) and a SYCL
  option. The user-built b12187 trees match the recipe flags, and the HIP
  tree now links OpenSSL (the v5.6.3 recipe fix). Recipes unchanged; the
  stable recipe already resolves the latest release (v0.9.0).

## Draft heads: one fix, two defaults changed

**LiLiCorr drafter ran as MTP.** NVIDIA's DFlash/LiLiCorr drafter for
Nemotron 3.5 Bolt (`dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf`, now
local) stores its lattice layers as `blk.5..7` beyond `dflash.block_count = 5`.
The tensor scan reported that as an MTP block, so AutoTuner sent
`--spec-type draft-mtp` and llama-server stopped with "failed to create MTP
context" (it would have done so on v5.6.3 too). EAGLE-3/DFlash/DSpark
architectures are drafts by definition now; the head runs as `draft-dflash`
(326/510 and 332/467 accepted drafts on Vulkan/HIP).

**Paired drafters that cost more than they give** stay in the draft menu but
are no longer selected by default (new profile key `auto_draft: false`, used
by GUI and CLI; a remembered choice still wins):

- *Nemotron 3.5 Bolt + LiLiCorr* on 48 GB: the draft (0.93 GB) and the
  recurrent rollback snapshots (+1.85 GB) cut the planned context from
  167,936 to 16,384 tokens on Vulkan and from 262,144 to 23,552 on HIP;
  decode 45.4 → 48.1 tok/s on Vulkan, 35.3 → 26.5 on HIP.
- *MiniCPM-V 4.7 35B-A3B + DSpark* (mirrored, decode tok/s):

  | Backend / prompt | No draft | n=16 (old default) | n=7 | n=3 |
  |---|---:|---:|---:|---:|
  | Vulkan code | 117.9 | 135.2 | **159.9** | 157.9 |
  | Vulkan German prose | **119.1** | 42.2 | 58.2 | 72.6 |
  | HIP code | **84.3** | 45.3 | 62.5 | 53.9 |
  | HIP German prose | **83.3** | 7.6 | 21.9 | 25.6 |

  Only code on Vulkan gains; the profile now uses 7 draft tokens when the
  drafter is chosen by hand.

## Rechecked gates

- **Qwen3.8 Flash Next + separate MTP head** still aborts with
  `GGML_ASSERT(buffer)` on b12187 (HIP and Vulkan), even with #31443 and
  #30141; the gate stays, verified build now b12187. Without speculation the
  plan answers normally (22–28 tok/s).
- **Prism PQ2_0/PTQ1_0** (4 loads) and **ROCmFPX** Agnes files (4 loads) are
  still refused by b12187 and by AutoTuner before launch.
- **Image + DFlash2** (Qwen3.8-27B): 152/168 accepted drafts on both
  backends, as on b11538 and b11669.
- **Ling-3.0-flash AD-IQ3_M on HIP** now passes the integer-list bench as
  well (it ended after "11," on b11538/b11669); thinking, tool and text checks
  pass on both backends.
- **Unicode model paths** after #31417: identical results on b11669 and
  b12187.

## Upstream status

PR #28696 (V4.1 conversion) closed in favour of #31086. Still open: issue
#29058 (Prism PQ2_0/PTQ1_0 types), PR #24185 (ROCmFP4 CPU support), issue
#27408 (image chunks in the draft KV; AutoTuner's range gate is unaffected
since b11319). ROCmFPX issue #26 (RDNA4 MMQ `J_best=0`) is fixed on the
ROCmFPX main branch with the user's patch; the fork pin `fpx_b11544` with
the local patch keeps working and was not rebuilt.
