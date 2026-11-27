# GGUF Modelle - Metadaten Übersicht

## Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-COHERENT.gguf
**Pfad:** `Agnes-AI\Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-COHERENT.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| Fehler | Konnte Metadaten nicht lesen: np.uint32(100) is not a valid GGMLQuantizationType |

---

## Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-STRIX_LEAN.gguf
**Pfad:** `Agnes-AI\Agnes-3.0-Flash-Preview-MTP-imatrix-Q4_0-ROCmFP4-STRIX_LEAN.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| Fehler | Konnte Metadaten nicht lesen: np.uint32(101) is not a valid GGMLQuantizationType |

---

## mmproj-Agnes-3.0-Flash-Preview-BF16.gguf
**Pfad:** `Agnes-AI\mmproj-Agnes-3.0-Flash-Preview-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[27]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 65 103 110 101 115  45  51  46  48  45  70 108  97 115 104  45  80 114
 101 118 105 101 119]
```

---

## Kolibri-1-Q4_K_M.gguf
**Pfad:** `Aleph-Alpha\Kolibri-1-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[903]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[107 111 108 105  98 114 105  49]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[128]` |
| `general.sampling.top_p` | `[0.97]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 75 111 108 105  98 114 105  32  49]` |
| `general.version` | `[49]` |
| `general.basename` | `[107 111 108 105  98 114 105]` |
| `general.size_label` | `[ 51  56  52 120  50  46  56  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 75 111 108 105  98 114 105  32  49  32  66  70  49  54]` |
| `general.base_model.0.organization` | `[ 65 108 101 112 104  32  65 108 112 104  97]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[101 110]` |
| `kolibri1.block_count` | `[50]` |
| `kolibri1.context_length` | `[262144]` |
| `kolibri1.embedding_length` | `[2560]` |
| `kolibri1.attention.head_count` | `[48]` |
| `kolibri1.attention.head_count_kv` | `[4]` |
| `kolibri1.rope.freq_base` | `[10000.]` |
| `kolibri1.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `kolibri1.expert_count` | `[384]` |
| `kolibri1.expert_used_count` | `[6]` |
| `kolibri1.attention.key_length` | `[128]` |
| `kolibri1.attention.value_length` | `[128]` |
| `kolibri1.expert_feed_forward_length` | `[512]` |
| `kolibri1.expert_shared_feed_forward_length` | `[512]` |
| `kolibri1.attention.sliding_window` | `[513]` |
| `kolibri1.attention.sliding_window_pattern` | `[False]` |
| `kolibri1.expert_shared_count` | `[1]` |
| `kolibri1.expert_weights_norm` | `[False]` |
| `kolibri1.expert_gating_func` | `[5]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[107 111 108 105  98 114 105  49]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 55 57 57 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[101 115  32  97  99]` |
| `tokenizer.ggml.eos_token_id` | `[127906]` |
| `tokenizer.ggml.padding_token_id` | `[127901]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  65 108 101 112 104  45  65 108 112 104  97  47  75
 111 108 105  98 114 105  45  49  45  66  70  49  54]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf
**Pfad:** `Alibaba\Qwen3\mmproj-Qwen3-Omni-30B-A3B-Instruct-bf16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[860]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.size_label` | `[49 46 49 66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.languages` | `[101 110]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[2048]` |
| `clip.audio.embedding_length` | `[1280]` |
| `clip.audio.feed_forward_length` | `[5120]` |
| `clip.audio.block_count` | `[32]` |
| `clip.audio.attention.head_count` | `[20]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `clip.vision.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.audio.projector_type` | `[113 119 101 110  51  97]` |
| `general.quantization_version` | `[2]` |

---

## Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf
**Pfad:** `Alibaba\Qwen3\Qwen3-Omni-30B-A3B-Instruct-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[579]` |
| `GGUF.kv_count` | `[34]` |
| `general.architecture` | `[113 119 101 110  51 118 108 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.size_label` | `[ 49  50  56 120  49  46  56  66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.languages` | `[101 110]` |
| `qwen3vlmoe.block_count` | `[48]` |
| `qwen3vlmoe.context_length` | `[65536]` |
| `qwen3vlmoe.embedding_length` | `[2048]` |
| `qwen3vlmoe.feed_forward_length` | `[768]` |
| `qwen3vlmoe.attention.head_count` | `[32]` |
| `qwen3vlmoe.attention.head_count_kv` | `[4]` |
| `qwen3vlmoe.rope.dimension_sections` | `[0]` |
| `qwen3vlmoe.rope.freq_base` | `[1.e+06]` |
| `qwen3vlmoe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3vlmoe.expert_count` | `[128]` |
| `qwen3vlmoe.expert_used_count` | `[8]` |
| `qwen3vlmoe.attention.key_length` | `[128]` |
| `qwen3vlmoe.attention.value_length` | `[128]` |
| `qwen3vlmoe.expert_feed_forward_length` | `[768]` |
| `qwen3vlmoe.expert_shared_feed_forward_length` | `[0]` |
| `qwen3vlmoe.n_deepstack_layers` | `[0]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 50 48 54 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.bos_token_id` | `[151645]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mmproj-Qwen3.6-27B-MTP-BF16.gguf
**Pfad:** `Alibaba\Qwen3.6\mmproj-Qwen3.6-27B-MTP-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

---

## mmproj-Qwen3.6-35B-A3B-MTP-UD_BF16.gguf
**Pfad:** `Alibaba\Qwen3.6\mmproj-Qwen3.6-35B-A3B-MTP-UD_BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.finetune` | `[51 53 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

---

## Qwen3.6-27B-MTP-IQ4_XS.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-27B-MTP-IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  50  55  66  46 116 120 116]
```

---

## Qwen3.6-27B-MTP-UD_Q6_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-27B-MTP-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  50  55  66  46 116 120 116]
```

---

## Qwen3.6-35B-A3B-MTP-UD_Q6_K.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-35B-A3B-MTP-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[77]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  46
 116 120 116]
```

---

## Qwen3.6-35B-A3B-MTP-UD_Q8_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-35B-A3B-MTP-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[77]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  46
 116 120 116]
```

---

## Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[26]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[77]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  46
 116 120 116]
```

---

## mmproj-Qwen3.8-27B-BF16.gguf
**Pfad:** `Alibaba\Qwen3.8\mmproj-Qwen3.8-27B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[35]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

---

## mmproj-Qwen3.8-Flash-Next-BF16.gguf
**Pfad:** `Alibaba\Qwen3.8\mmproj-Qwen3.8-Flash-Next-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[31]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101 120 116]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101 120 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[52 52 57 77]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[113 119 101 110  45  99 111 109 109 117 110 105 116 121  45  49  46  48]` |
| `general.license.link` | `[76 73 67 69 78 83 69]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2560]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

---

## mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf
**Pfad:** `Alibaba\Qwen3.8\mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[34]` |
| `GGUF.kv_count` | `[51]` |
| `general.architecture` | `[113 119 101 110  52 101 120 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 67 107 112 116  95  81  51  56]` |
| `general.size_label` | `[ 53  49  50 120  49  46  52  66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[113 119 101 110  45  99 111 109 109 117 110 105 116 121  45  49  46  48]` |
| `general.license.link` | `[76 73 67 69 78 83 69]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen4exp.block_count` | `[49]` |
| `qwen4exp.context_length` | `[262144]` |
| `qwen4exp.embedding_length` | `[2560]` |
| `qwen4exp.attention.head_count` | `[24]` |
| `qwen4exp.attention.head_count_kv` | `[2]` |
| `qwen4exp.rope.dimension_sections` | `[0]` |
| `qwen4exp.rope.freq_base` | `[1.e+07]` |
| `qwen4exp.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen4exp.expert_count` | `[512]` |
| `qwen4exp.expert_used_count` | `[10]` |
| `qwen4exp.attention.key_length` | `[256]` |
| `qwen4exp.attention.value_length` | `[256]` |
| `qwen4exp.expert_feed_forward_length` | `[640]` |
| `qwen4exp.expert_shared_feed_forward_length` | `[640]` |
| `qwen4exp.nextn_predict_layers` | `[1]` |
| `qwen4exp.ssm.conv_kernel` | `[4]` |
| `qwen4exp.ssm.state_size` | `[128]` |
| `qwen4exp.ssm.group_count` | `[16]` |
| `qwen4exp.ssm.time_step_rank` | `[48]` |
| `qwen4exp.ssm.inner_size` | `[6144]` |
| `qwen4exp.full_attention_interval` | `[4]` |
| `qwen4exp.rope.dimension_count` | `[64]` |
| `qwen4exp.hyper_connection.count` | `[4]` |
| `qwen4exp.hyper_connection.low_rank` | `[320]` |
| `qwen4exp.attention.indexer.head_count` | `[4]` |
| `qwen4exp.attention.indexer.key_length` | `[128]` |
| `qwen4exp.attention.indexer.top_k` | `[2048]` |
| `qwen4exp.attention.compress_ratios` | `[0]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.8-27B-DFlash2-Q4_K_M.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-DFlash2-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[81]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66  45  68  70 108  97 115 104   50]` |
| `general.author` | `[ 73 110  99 111  32  65  73]` |
| `general.organization` | `[122  45 108  97  98]` |
| `general.finetune` | `[ 68  70 108  97 115 104  50]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.source.url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[5120]` |
| `dflash.feed_forward_length` | `[17408]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.attention.causal` | `[False]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[256]` |
| `dflash.selector_top_k` | `[16]` |
| `dflash.target_layers` | `[62]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[248070]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `dflash.rope.dimension_sections` | `[0]` |

**general.source.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  45 108  97  98  47  81 119 101 110  51  46  56
  45  50  55  66  45  68  70 108  97 115 104  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.8-27B-UD-IQ4_XS.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q4_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q4_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q6_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[67]` |
| `general.architecture` | `[113 119 101 110  52 101 120 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  32  70 108  97 115 104  32  78 101 120 116]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[ 53  49  50 120  53  54  66]` |
| `qwen4exp.block_count` | `[48]` |
| `qwen4exp.context_length` | `[262144]` |
| `qwen4exp.embedding_length` | `[2560]` |
| `qwen4exp.attention.head_count` | `[24]` |
| `qwen4exp.attention.head_count_kv` | `[2]` |
| `qwen4exp.rope.dimension_sections` | `[0]` |
| `qwen4exp.rope.freq_base` | `[1.e+07]` |
| `qwen4exp.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen4exp.expert_count` | `[512]` |
| `qwen4exp.expert_used_count` | `[10]` |
| `qwen4exp.attention.key_length` | `[256]` |
| `qwen4exp.attention.value_length` | `[256]` |
| `qwen4exp.expert_feed_forward_length` | `[640]` |
| `qwen4exp.expert_shared_feed_forward_length` | `[640]` |
| `qwen4exp.ssm.conv_kernel` | `[4]` |
| `qwen4exp.ssm.state_size` | `[128]` |
| `qwen4exp.ssm.group_count` | `[16]` |
| `qwen4exp.ssm.time_step_rank` | `[48]` |
| `qwen4exp.ssm.inner_size` | `[6144]` |
| `qwen4exp.full_attention_interval` | `[4]` |
| `qwen4exp.rope.dimension_count` | `[64]` |
| `qwen4exp.hyper_connection.count` | `[4]` |
| `qwen4exp.hyper_connection.low_rank` | `[320]` |
| `qwen4exp.attention.indexer.head_count` | `[4]` |
| `qwen4exp.attention.indexer.key_length` | `[128]` |
| `qwen4exp.attention.indexer.top_k` | `[2048]` |
| `qwen4exp.attention.compress_ratios` | `[4]` |
| `qwen4exp.ple.layers` | `[1]` |
| `qwen4exp.ple.ngram_size` | `[3]` |
| `qwen4exp.ple.heads_per_ngram` | `[8]` |
| `qwen4exp.ple.conv_kernel` | `[4]` |
| `qwen4exp.ple.eos_token_id` | `[248044]` |
| `qwen4exp.embedding_length_per_layer_input` | `[160]` |
| `qwen4exp.ple.layer_multipliers` | `[8052911324071]` |
| `qwen4exp.ple.head_offsets` | `[300001275]` |
| `qwen4exp.ple.head_vocab_sizes` | `[20000171]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `qwen4exp.ple.image_token_id` | `[248056]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[10]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[926]` |
| `quantize.imatrix.chunks_count` | `[45]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

**general.description:**
```jinja
[ 65  32  80 114 101 118 105 101 119  32 111 102  32 116 104 101  32  81
 119 101 110  52  32  65 114  99 104 105 116 101  99 116 117 114 101]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101 120 116
  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108
 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101
 120 116  46 116 120 116]
```

---

## Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[503]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

---

## Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[721]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

---

## Qwen-AgentWorld-35B-A3B-UD_Q6_K.gguf
**Pfad:** `Alibaba\QwenAgentWorld\Qwen-AgentWorld-35B-A3B-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[1]` |
| `general.dataset.0.name` | `[ 65 103 101 110 116  87 111 114 108 100  66 101 110  99 104]` |
| `general.dataset.0.organization` | `[ 81 119 101 110]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.name:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.basename:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66  47  98 108
 111  98  47 109  97 105 110  47  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  32  65 103 101 110 116  87 111 114 108 100  32  51  53
  66  32  65  51  66]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  65 103 101 110 116  87 111 114
 108 100  66 101 110  99 104]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45  51  53
  66  45  65  51  66  45  71  71  85  70  47 105 109  97 116 114 105 120
  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45
  51  53  66  45  65  51  66  46 116 120 116]
```

---

## mmproj-Unlimited-OCR-BF16.gguf
**Pfad:** `Baidu\mmproj-Unlimited-OCR-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[476]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 85 110 108 105 109 105 116 101 100  32  79  67  82]` |
| `general.size_label` | `[52 48 49 77]` |
| `general.license` | `[109 105 116]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 85 110 108 105 109 105 116 101 100  32  79  67  82]` |
| `general.base_model.0.organization` | `[ 66  97 105 100 117]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1280]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[64]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[100 101 101 112 115 101 101 107 111  99 114]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.projector.scale_factor` | `[1]` |
| `clip.vision.window_size` | `[14]` |
| `clip.vision.preproc_min_tiles` | `[2]` |
| `clip.vision.preproc_max_tiles` | `[32]` |
| `clip.vision.sam.block_count` | `[12]` |
| `clip.vision.sam.embedding_length` | `[768]` |
| `clip.vision.sam.head_count` | `[12]` |
| `general.quantization_version` | `[2]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  98  97 105 100 117  47  85 110 108 105 109 105 116
 101 100  45  79  67  82]
```

---

## Unlimited-OCR-BF16.gguf
**Pfad:** `Baidu\Unlimited-OCR-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[155]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[100 101 101 112 115 101 101 107  50  45 111  99 114]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 85 110 108 105 109 105 116 101 100  32  79  67  82]` |
| `general.size_label` | `[ 54  52 120  53  53  48  77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `deepseek2-ocr.block_count` | `[12]` |
| `deepseek2-ocr.context_length` | `[32768]` |
| `deepseek2-ocr.embedding_length` | `[1280]` |
| `deepseek2-ocr.feed_forward_length` | `[6848]` |
| `deepseek2-ocr.attention.head_count` | `[10]` |
| `deepseek2-ocr.attention.head_count_kv` | `[10]` |
| `deepseek2-ocr.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `deepseek2-ocr.expert_used_count` | `[6]` |
| `deepseek2-ocr.expert_group_count` | `[1]` |
| `deepseek2-ocr.expert_group_used_count` | `[1]` |
| `general.file_type` | `[32]` |
| `deepseek2-ocr.leading_dense_block_count` | `[1]` |
| `deepseek2-ocr.vocab_size` | `[129280]` |
| `deepseek2-ocr.expert_feed_forward_length` | `[896]` |
| `deepseek2-ocr.expert_count` | `[64]` |
| `deepseek2-ocr.expert_shared_count` | `[2]` |
| `deepseek2-ocr.rope.dimension_count` | `[0]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[100 101 101 112 115 101 101 107  45 118  51]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 57 50 55 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 165 194 177 196 173  32 195 166 194 170 196 178]` |
| `tokenizer.ggml.bos_token_id` | `[0]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.padding_token_id` | `[2]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |

**tokenizer.chat_template:**
```jinja
[123  37  32 102 111 114  32 109  32 105 110  32 109 101 115 115  97 103
 101 115  32  37 125 123 123 109  91  39  99 111 110 116 101 110 116  39
  93 125 125 123  37  32 101 110 100 102 111 114  32  37 125]
```

---

## mmproj-PaddleOCR-VL-1.6.gguf
**Pfad:** `Baidu\PaddleOCR-VL-1.6\mmproj-PaddleOCR-VL-1.6.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[443]` |
| `GGUF.kv_count` | `[31]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 80  97 100 100 108 101  79  67  82  32  86  76  32  49  46  54]` |
| `general.version` | `[49 46 54]` |
| `general.basename` | `[ 80  97 100 100 108 101  79  67  82  45  86  76]` |
| `general.size_label` | `[52 51 57 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 80  97 100 100 108 101  79  67  82  32  86  76  32  49  46  53]` |
| `general.base_model.0.version` | `[49 46 53]` |
| `general.base_model.0.organization` | `[ 80  97 100 100 108 101  80  97 100 100 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1024]` |
| `clip.vision.image_size` | `[1001]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[112  97 100 100 108 101 111  99 114]` |
| `clip.vision.image_max_pixels` | `[1003520]` |
| `clip.vision.image_min_pixels` | `[112896]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  80  97 100 100 108 101  80  97 100 100 108 101  47
  80  97 100 100 108 101  79  67  82  45  86  76  45  49  46  53]
```

---

## PaddleOCR-VL-1.6.gguf
**Pfad:** `Baidu\PaddleOCR-VL-1.6\PaddleOCR-VL-1.6.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[165]` |
| `GGUF.kv_count` | `[42]` |
| `general.architecture` | `[112  97 100 100 108 101 111  99 114]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 80  97 100 100 108 101  79  67  82  32  86  76  32  49  46  54]` |
| `general.version` | `[49 46 54]` |
| `general.basename` | `[ 80  97 100 100 108 101  79  67  82  45  86  76]` |
| `general.size_label` | `[52 54 55 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 80  97 100 100 108 101  79  67  82  32  86  76  32  49  46  53]` |
| `general.base_model.0.version` | `[49 46 53]` |
| `general.base_model.0.organization` | `[ 80  97 100 100 108 101  80  97 100 100 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `paddleocr.block_count` | `[18]` |
| `paddleocr.context_length` | `[131072]` |
| `paddleocr.embedding_length` | `[1024]` |
| `paddleocr.feed_forward_length` | `[3072]` |
| `paddleocr.attention.head_count` | `[16]` |
| `paddleocr.attention.head_count_kv` | `[2]` |
| `paddleocr.rope.dimension_sections` | `[0]` |
| `paddleocr.rope.freq_base` | `[500000.]` |
| `paddleocr.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `paddleocr.attention.key_length` | `[128]` |
| `paddleocr.attention.value_length` | `[128]` |
| `general.file_type` | `[32]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[108 108  97 109  97]` |
| `tokenizer.ggml.pre` | `[100 101 102  97 117 108 116]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 48 51 52 50 51 93]` |
| `tokenizer.ggml.scores` | `[-10000.]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.seperator_token_id` | `[100272]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[100274]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  80  97 100 100 108 101  80  97 100 100 108 101  47
  80  97 100 100 108 101  79  67  82  45  86  76  45  49  46  53]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf
**Pfad:** `bottlecapAI\mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[25]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | *Siehe Code-Block unten* |
| `general.size_label` | `[52 54 49 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 51 100  97  57  98 101  51  57  57  97  49  57  53  49  97  99  57 102
  99  99  54 100  49  98  55  54  97 102  57  53  54  53  99 101 100  50
  56  53 102  49]
```

**general.finetune:**
```jinja
[ 51 100  97  57  98 101  51  57  57  97  49  57  53  49  97  99  57 102
  99  99  54 100  49  98  55  54  97 102  57  53  54  53  99 101 100  50
  56  53 102  49]
```

---

## ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf
**Pfad:** `bottlecapAI\ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 67 107 112 116  50  49  48  48]` |
| `general.size_label` | `[50 55 66]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## North-Mini-Code-1.0-UD_Q6_K_XL.gguf
**Pfad:** `CohereLabs\North-Mini-Code-1.0-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[442]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[ 99 111 104 101 114 101  50 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.version` | `[49 46 48]` |
| `general.finetune` | `[ 67 111 100 101]` |
| `general.basename` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 77 105 110 105]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 78 111 114 116 104  32  77 105 110 105  32  67 111 100 101  32  49  46   48]` |
| `general.base_model.0.version` | `[49 46 48]` |
| `general.base_model.0.organization` | `[ 67 111 104 101 114 101  76  97  98 115]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 103 101 110 116]` |
| `cohere2moe.block_count` | `[49]` |
| `cohere2moe.context_length` | `[500000]` |
| `cohere2moe.embedding_length` | `[2048]` |
| `cohere2moe.feed_forward_length` | `[3072]` |
| `cohere2moe.attention.head_count` | `[32]` |
| `cohere2moe.attention.head_count_kv` | `[4]` |
| `cohere2moe.rope.freq_base` | `[50000.]` |
| `cohere2moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `cohere2moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `cohere2moe.expert_count` | `[128]` |
| `cohere2moe.expert_used_count` | `[8]` |
| `cohere2moe.attention.key_length` | `[128]` |
| `cohere2moe.attention.value_length` | `[128]` |
| `cohere2moe.logit_scale` | `[1.]` |
| `cohere2moe.attention.sliding_window` | `[4096]` |
| `cohere2moe.attention.sliding_window_pattern` | `[False]` |
| `cohere2moe.vocab_size` | `[262144]` |
| `cohere2moe.expert_feed_forward_length` | `[768]` |
| `cohere2moe.leading_dense_block_count` | `[1]` |
| `cohere2moe.expert_weights_norm` | `[False]` |
| `cohere2moe.expert_gating_func` | `[2]` |
| `cohere2moe.rope.dimension_count` | `[128]` |
| `cohere2moe.rope.scaling.type` | `[110 111 110 101]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 105 110 121  95  97 121  97]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 54 50 49 52 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  98 114  32  97 110 105]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[255001]` |
| `tokenizer.ggml.unknown_token_id` | `[4]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[391]` |
| `quantize.imatrix.chunks_count` | `[200]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  67 111 104 101 114 101  76  97  98 115  47  78 111
 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46  48]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46
  48  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115
 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45
  49  46  48  46 116 120 116]
```

---

## maple-preview-TQ2_0-head-Q4_K.gguf
**Pfad:** `DeepGrove\maple-preview-TQ2_0-head-Q4_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[291]` |
| `GGUF.kv_count` | `[40]` |
| `general.architecture` | `[109  97 112 108 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 77  97 112 108 101  32  80 114 101 118 105 101 119]` |
| `general.size_label` | `[ 50  53  54 120  57  54  50  77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[101 110]` |
| `maple.block_count` | `[24]` |
| `maple.context_length` | `[131072]` |
| `maple.embedding_length` | `[2048]` |
| `maple.feed_forward_length` | `[4096]` |
| `maple.attention.head_count` | `[16]` |
| `maple.attention.head_count_kv` | `[4]` |
| `maple.rope.freq_base` | `[10000.]` |
| `maple.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `maple.expert_count` | `[256]` |
| `maple.expert_used_count` | `[8]` |
| `maple.attention.key_length` | `[128]` |
| `maple.attention.value_length` | `[128]` |
| `maple.vocab_size` | `[151936]` |
| `maple.rope.dimension_count` | `[64]` |
| `maple.attention.sliding_window` | `[512]` |
| `maple.attention.sliding_window_pattern` | `[False]` |
| `maple.expert_feed_forward_length` | `[512]` |
| `maple.expert_weights_scale` | `[1.]` |
| `maple.expert_weights_norm` | `[ True]` |
| `maple.expert_gating_func` | `[1]` |
| `maple.swiglu_clamp_exp` | `[7.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151643]` |
| `tokenizer.ggml.bos_token_id` | `[151643]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[37]` |

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## dots.ocr-Q8_0.gguf
**Pfad:** `dots-studio\dots.ocr\dots.ocr-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[339]` |
| `GGUF.kv_count` | `[25]` |
| `general.architecture` | `[113 119 101 110  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.size_label` | `[49 46 56 66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `qwen2.block_count` | `[28]` |
| `qwen2.context_length` | `[131072]` |
| `qwen2.embedding_length` | `[1536]` |
| `qwen2.feed_forward_length` | `[8960]` |
| `qwen2.attention.head_count` | `[12]` |
| `qwen2.attention.head_count_kv` | `[2]` |
| `qwen2.rope.freq_base` | `[1.e+06]` |
| `qwen2.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `general.file_type` | `[7]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151643]` |
| `tokenizer.ggml.padding_token_id` | `[151680]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**tokenizer.chat_template:**
```jinja
[123  37  45  32 102 111 114  32 109  32 105 110  32 109 101 115 115  97
 103 101 115  32  37 125  10  32  32  32  32 123  37  45  32 105 102  32
 109  46 114 111 108 101  32  61  61  32  39 115 121 115 116 101 109  39
  32  37 125  10  32  32  32  32  32  32  32  32 123 123  45  32  39  60
 124 115 121 115 116 101 109 124  62  39  32  43  32 109  46  99 111 110
 116 101 110 116  32  43  32  39  60 124 101 110 100 111 102 115 121 115
 116 101 109 124  62  92 110  39  32 125 125  10  32  32  32  32 123  37
  45  32 101 108 105 102  32 109  46 114 111 108 101  32  61  61  32  39
 117 115 101 114  39  32  37 125  10  32  32  32  32  32  32  32  32 123
 123  45  32  39  60 124 117 115 101 114 124  62  39  32  43  32 109  46
  99 111 110 116 101 110 116  32  43  32  39  60 124 101 110 100 111 102
 117 115 101 114 124  62  39  32 125 125  10  32  32  32  32 123  37  45
  32 101 108 105 102  32 109  46 114 111 108 101  32  61  61  32  39  97
 115 115 105 115 116  97 110 116  39  32  37 125  10  32  32  32  32  32
  32  32  32 123 123  45  32  39  60 124  97 115 115 105 115 116  97 110
 116 124  62  39  32  43  32 109  46  99 111 110 116 101 110 116  32 125
 125  10  32  32  32  32  32  32  32  32 123  37  45  32 105 102  32 110
 111 116  32 108 111 111 112  46 108  97 115 116  32  37 125  10  32  32
  32  32  32  32  32  32  32  32  32  32 123 123  45  32  39  60 124 101
 110 100 111 102  97 115 115 105 115 116  97 110 116 124  62  39  32 125
 125  10  32  32  32  32  32  32  32  32 123  37  45  32 101 110 100 105
 102  32  37 125  10  32  32  32  32 123  37  45  32 101 110 100 105 102
  32  37 125  10 123  37  45  32 101 110 100 102 111 114  32  37 125  10
 123  37  45  32 105 102  32 109 101 115 115  97 103 101 115  91  45  49
  93  46 114 111 108 101  32  33  61  32  39  97 115 115 105 115 116  97
 110 116  39  32  37 125  10  32  32  32  32 123 123  45  32  39  60 124
  97 115 115 105 115 116  97 110 116 124  62  39  32 125 125  10 123  37
  45  32 101 110 100 105 102  32  37 125]
```

---

## mmproj-dots.ocr-Q8_0.gguf
**Pfad:** `dots-studio\dots.ocr\mmproj-dots.ocr-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[304]` |
| `GGUF.kv_count` | `[24]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.size_label` | `[49 46 51 66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `general.file_type` | `[7]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1536]` |
| `clip.vision.image_size` | `[0]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1536]` |
| `clip.vision.feed_forward_length` | `[4224]` |
| `clip.vision.block_count` | `[42]` |
| `clip.vision.attention.head_count` | `[12]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[100 111 116 115  95 111  99 114]` |
| `clip.vision.image_min_pixels` | `[3136]` |
| `clip.vision.image_max_pixels` | `[11289600]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `clip.use_silu` | `[ True]` |
| `general.quantization_version` | `[2]` |

---

## VibeCoder-20b-RL1.0-MOE-MXFP4.gguf
**Pfad:** `EpistemeAI\VibeCoder-20b-RL1.0-MOE-MXFP4.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[459]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 112 116  45 111 115 115]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 73 110]` |
| `general.size_label` | `[ 51  50 120  50  46  52  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[48 46 48 50]` |
| `general.base_model.0.organization` | `[ 69 112 105 115 116 101 109 101  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[1]` |
| `general.dataset.0.name` | *Siehe Code-Block unten* |
| `general.dataset.0.organization` | `[ 69 112 105 115 116 101 109 101  65  73]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[103 112 116  95 111 115 115]` |
| `general.languages` | `[101 110]` |
| `gpt-oss.block_count` | `[24]` |
| `gpt-oss.context_length` | `[131072]` |
| `gpt-oss.embedding_length` | `[2880]` |
| `gpt-oss.feed_forward_length` | `[2880]` |
| `gpt-oss.attention.head_count` | `[64]` |
| `gpt-oss.attention.head_count_kv` | `[8]` |
| `gpt-oss.rope.scaling.type` | `[121  97 114 110]` |
| `gpt-oss.rope.scaling.factor` | `[32.]` |
| `gpt-oss.rope.scaling.original_context_length` | `[4096]` |
| `gpt-oss.rope.scaling.yarn_beta_fast` | `[32.]` |
| `gpt-oss.rope.scaling.yarn_beta_slow` | `[1.]` |
| `gpt-oss.rope.freq_base` | `[150000.]` |
| `gpt-oss.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `gpt-oss.expert_count` | `[32]` |
| `gpt-oss.expert_used_count` | `[4]` |
| `gpt-oss.attention.key_length` | `[64]` |
| `gpt-oss.attention.value_length` | `[64]` |
| `gpt-oss.attention.sliding_window` | `[128]` |
| `gpt-oss.expert_feed_forward_length` | `[2880]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 112 116  45  52 111]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 48 49 48 56 55 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  99 111  99 111  32 115]` |
| `tokenizer.ggml.bos_token_id` | `[199998]` |
| `tokenizer.ggml.eos_token_id` | `[200002]` |
| `tokenizer.ggml.padding_token_id` | `[200017]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[38]` |

**general.base_model.0.name:**
```jinja
[ 86 105  98 101  67 111 100 101 114  32  50  48  98  32  48  46  48  50
  32  68 101  98 117 103 103 101 114]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  69 112 105 115 116 101 109 101  65  73  47  86 105
  98 101  67 111 100 101 114  45  50  48  98  45  48  46  48  50  45  68
 101  98 117 103 103 101 114]
```

**general.dataset.0.name:**
```jinja
[ 86 105  98 101  32  67 111 100 101 114  32  80  97 114 116  32  68 101
  98 117 103]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  69 112 105 115 116 101 109 101  65  73  47 118 105
  98 101  45  99 111 100 101 114  45 112  97 114 116  45 100 101  98 117
 103]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  35 125]
```

---

## Clef-Flash-Q4_K_M.gguf
**Pfad:** `ggml-org\Clef-Flash-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[587]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[ 99 108 101 102]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 67 108 101 102  45  70 108  97 115 104]` |
| `general.size_label` | `[57 46 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  53  32  57  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `clef.block_count` | `[32]` |
| `clef.context_length` | `[262144]` |
| `clef.embedding_length` | `[4096]` |
| `clef.feed_forward_length` | `[12288]` |
| `clef.attention.head_count` | `[16]` |
| `clef.attention.head_count_kv` | `[4]` |
| `clef.rope.dimension_sections` | `[0]` |
| `clef.rope.freq_base` | `[1.e+07]` |
| `clef.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `clef.attention.key_length` | `[256]` |
| `clef.attention.value_length` | `[256]` |
| `clef.ssm.conv_kernel` | `[4]` |
| `clef.ssm.state_size` | `[128]` |
| `clef.ssm.group_count` | `[16]` |
| `clef.ssm.time_step_rank` | `[32]` |
| `clef.ssm.inner_size` | `[4096]` |
| `clef.attention.recurrent_layers` | `[False]` |
| `clef.full_attention_interval` | `[4]` |
| `clef.rope.dimension_count` | `[64]` |
| `clef.decision.type` | `[ 99 108 101 102]` |
| `clef.decision.routing_block_count` | `[2]` |
| `clef.decision.block_count` | `[4]` |
| `clef.decision.head_count` | `[16]` |
| `clef.attention.layer_norm_epsilon` | `[1.e-05]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  57  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**tokenizer.chat_template.systemone:**
```jinja
[123 123  32 ...  32 125 125]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## Julia-1-Q8_0.gguf
**Pfad:** `ggml-org\Julia-1-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[165]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[109 111 100 101 114 110  45  98 101 114 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 74 117 108 105  97  45  49]` |
| `general.size_label` | `[49 52 52 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[109 109  66  69  82  84  32  83 109  97 108 108]` |
| `general.base_model.0.organization` | `[ 74 104 117  32  67 108 115 112]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[114 111 117 116 105 110 103]` |
| `modern-bert.block_count` | `[24]` |
| `modern-bert.context_length` | `[8192]` |
| `modern-bert.embedding_length` | `[384]` |
| `modern-bert.feed_forward_length` | `[1536]` |
| `modern-bert.attention.head_count` | `[6]` |
| `modern-bert.rope.freq_base` | `[160000.]` |
| `modern-bert.rope.freq_base_swa` | `[160000.]` |
| `modern-bert.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `modern-bert.attention.layer_norm_epsilon` | `[1.e-05]` |
| `modern-bert.attention.causal` | `[False]` |
| `modern-bert.classifier.pooling_type` | `[1]` |
| `modern-bert.attention.sliding_window` | `[128]` |
| `modern-bert.attention.sliding_window_pattern` | `[3]` |
| `modern-bert.rope.scaling.type` | `[110 111 110 101]` |
| `modern-bert.vocab_size` | `[256000]` |
| `modern-bert.hidden_activation` | `[103 101 108 117]` |
| `modern-bert.decision.type` | `[108  97 121  97]` |
| `modern-bert.decision.block_count` | `[2]` |
| `modern-bert.decision.max_head_tokens` | `[256]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[109 109  98 101 114 116]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  57  57  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[232 170 172 230 152 142  32 230 155 184]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.seperator_token_id` | `[1]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.token_type_count` | `[3]` |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 106 104 117  45  99 108 115 112  47 109 109  66  69
  82  84  45 115 109  97 108 108]
```

**tokenizer.chat_template.systemone:**
```jinja
[ 60  98 111 115  62 123 123  32 116 121 112 101  32 125 125  32 113 117
 101 115 116 105 111 110  58  32 123 123  32 105 110 115 116 114 117  99
 116 105 111 110 115  32 105 102  32 105 110 115 116 114 117  99 116 105
 111 110 115  32 105 115  32 115 116 114 105 110 103  32 101 108 115 101
  32 105 110 115 116 114 117  99 116 105 111 110 115  32 124  32 116 111
 106 115 111 110  32 125 125  60 101 111 115  62 123  37  32 102 111 114
  32 111  32 105 110  32 111 112 116 105 111 110 115  32  37 125  60 109
  97 115 107  62  32 123  37  32 105 102  32 111  46 100 101 115  99 114
 105 112 116 105 111 110  32  37 125 123 123  32 111  46 100 101 115  99
 114 105 112 116 105 111 110  32 105 102  32 111  46 100 101 115  99 114
 105 112 116 105 111 110  32 105 115  32 115 116 114 105 110 103  32 101
 108 115 101  32 111  46 100 101 115  99 114 105 112 116 105 111 110  32
 124  32 116 111 106 115 111 110  32 125 125 123  37  32 101 108 115 101
  32  37 125 123 123  32 111  46 107 101 121  32 125 125 123  37  32 101
 110 100 105 102  32  37 125 123  37  32 101 110 100 102 111 114  32  37
 125  60 101 111 115  62 123 123  32 115 116  97 116 101  32 105 102  32
 115 116  97 116 101  32 105 115  32 115 116 114 105 110 103  32 101 108
 115 101  32 115 116  97 116 101  32 124  32 116 111 106 115 111 110  32
 125 125  60 101 111 115  62]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## Kev-4B-Q4_K_M.gguf
**Pfad:** `ggml-org\Kev-4B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[428]` |
| `GGUF.kv_count` | `[90]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 75 101 118  45  52  66]` |
| `general.size_label` | `[52 46 50 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  53  32  52  66  32  66  97 115 101]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[13]` |
| `general.dataset.0.name` | `[ 66  97 110 107 105 110 103  55  55]` |
| `general.dataset.0.organization` | `[ 76 101 103  97  99 121  32  68  97 116  97 115 101 116 115]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.1.name` | `[ 66 111 111 108 113]` |
| `general.dataset.1.organization` | `[ 71 111 111 103 108 101]` |
| `general.dataset.1.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.2.name` | `[ 65 103  95  78 101 119 115]` |
| `general.dataset.2.organization` | `[ 70  97 110  99 121 122 104 120]` |
| `general.dataset.2.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.3.name` | `[ 77 117 108 116 105  95  78 108 105]` |
| `general.dataset.3.organization` | `[ 78 121 117  32  77 108 108]` |
| `general.dataset.3.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.4.name` | `[ 83 115 116  53]` |
| `general.dataset.4.organization` | `[ 83 101 116  70 105 116]` |
| `general.dataset.4.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.5.name` | `[ 89 101 108 112  95  82 101 118 105 101 119  95  70 117 108 108]` |
| `general.dataset.5.organization` | `[ 89 101 108 112]` |
| `general.dataset.5.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.6.name` | `[ 84 114 101  99]` |
| `general.dataset.6.organization` | `[ 67 111 103  67 111 109 112]` |
| `general.dataset.6.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.7.name` | `[ 68  98 112 101 100 105  97  95  49  52]` |
| `general.dataset.7.organization` | `[ 70  97 110  99 121 122 104 120]` |
| `general.dataset.7.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.8.name` | *Siehe Code-Block unten* |
| `general.dataset.8.organization` | `[ 83 101 116  70 105 116]` |
| `general.dataset.8.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.9.name` | `[ 73 109 100  98]` |
| `general.dataset.9.organization` | `[ 83 116  97 110 102 111 114 100 110 108 112]` |
| `general.dataset.9.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.10.name` | `[ 67 111 109 109 105 116 112  97  99 107 102 116]` |
| `general.dataset.10.organization` | `[ 66 105 103  99 111 100 101]` |
| `general.dataset.10.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.11.name` | *Siehe Code-Block unten* |
| `general.dataset.11.version` | `[50 46 48]` |
| `general.dataset.11.organization` | `[ 78 118 105 100 105  97]` |
| `general.dataset.11.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.12.name` | *Siehe Code-Block unten* |
| `general.dataset.12.organization` | `[ 68  97 118 105 100 104 101 105 110 101 109  97 110]` |
| `general.dataset.12.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45  99 108  97 115 115 105 102 105  99  97 116 105 111  110]` |
| `general.languages` | `[101 110]` |
| `qwen35.block_count` | `[32]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[2560]` |
| `qwen35.feed_forward_length` | `[9216]` |
| `qwen35.attention.head_count` | `[16]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[32]` |
| `qwen35.ssm.inner_size` | `[4096]` |
| `qwen35.attention.recurrent_layers` | `[False]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.decision.type` | `[107 101 118]` |
| `qwen35.embedding_length_out` | `[512]` |
| `qwen35.decision.temperature.choice` | `[2.40605]` |
| `qwen35.decision.temperature.score` | `[2.40605]` |
| `qwen35.decision.temperature.noul` | `[2.40605]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248044]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  52  66  45  66  97 115 101]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 108 101 103  97  99 121  45 100  97 116  97 115 101
 116 115  47  98  97 110 107 105 110 103  55  55]
```

**general.dataset.1.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47  98 111 111 108 113]
```

**general.dataset.2.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 102  97 110  99 121 122 104 120  47  97 103  95 110
 101 119 115]
```

**general.dataset.3.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 110 121 117  45 109 108 108  47 109 117 108 116 105
  95 110 108 105]
```

**general.dataset.4.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  83 101 116  70 105 116  47 115 115 116  53]
```

**general.dataset.5.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  89 101 108 112  47 121 101 108 112  95 114 101 118
 105 101 119  95 102 117 108 108]
```

**general.dataset.6.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  67 111 103  67 111 109 112  47 116 114 101  99]
```

**general.dataset.7.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 102  97 110  99 121 122 104 120  47 100  98 112 101
 100 105  97  95  49  52]
```

**general.dataset.8.name:**
```jinja
[ 65 109  97 122 111 110  95  82 101 118 105 101 119 115  95  77 117 108
 116 105  95  69 110]
```

**general.dataset.8.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  83 101 116  70 105 116  47  97 109  97 122 111 110
  95 114 101 118 105 101 119 115  95 109 117 108 116 105  95 101 110]
```

**general.dataset.9.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 115 116  97 110 102 111 114 100 110 108 112  47 105
 109 100  98]
```

**general.dataset.10.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  98 105 103  99 111 100 101  47  99 111 109 109 105
 116 112  97  99 107 102 116]
```

**general.dataset.11.name:**
```jinja
[ 65 101 103 105 115  32  65  73  32  67 111 110 116 101 110 116  32  83
  97 102 101 116 121  32  68  97 116  97 115 101 116  32  50  46  48]
```

**general.dataset.11.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 110 118 105 100 105  97  47  65 101 103 105 115  45
  65  73  45  67 111 110 116 101 110 116  45  83  97 102 101 116 121  45
  68  97 116  97 115 101 116  45  50  46  48]
```

**general.dataset.12.name:**
```jinja
[ 67 111 110 115 117 109 101 114  32  70 105 110  97 110  99 101  32  67
 111 109 112 108  97 105 110 116 115  32  76  97 114 103 101]
```

**general.dataset.12.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 100  97 118 105 100 104 101 105 110 101 109  97 110
  47  99 111 110 115 117 109 101 114  45 102 105 110  97 110  99 101  45
  99 111 109 112 108  97 105 110 116 115  45 108  97 114 103 101]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**tokenizer.chat_template.systemone:**
```jinja
[ 60 124 102 105 109  95 112 114 101 102 105 120 124  62 123 123  32 115
 116  97 116 101  32 125 125  60 124 102 105 109  95 109 105 100 100 108
 101 124  62 123 123  32 105 110 115 116 114 117  99 116 105 111 110 115
  32 125 125 123  37  32 102 111 114  32 111  32 105 110  32 111 112 116
 105 111 110 115  32  37 125  60 124  98 111 120  95 115 116  97 114 116
 124  62 123  37  32 105 102  32 116 121 112 101  32  61  61  32  39 115
  99 111 114 101  39  32  37 125 123  37  32 105 102  32 111  46 100 101
 115  99 114 105 112 116 105 111 110  32  37 125 123 123  32 111  46 100
 101 115  99 114 105 112 116 105 111 110  32 125 125 123  37  32 101 110
 100 105 102  32  37 125 123  37  32 101 108 115 101  32  37 125 123  37
  32 105 102  32 116 121 112 101  32  33  61  32  39 110 111 117 108  39
  32  37 125 123 123  32 111  46 107 101 121  32 125 125 123  37  32 101
 108 105 102  32 111  46 107 101 121  32  61  61  32  39 116 114 117 101
  39  32  37 125 121 101 115 123  37  32 101 108 115 101  32  37 125 110
 111 123  37  32 101 110 100 105 102  32  37 125 123  37  32 105 102  32
 111  46 100 101 115  99 114 105 112 116 105 111 110  32  37 125  58  32
 123 123  32 111  46 100 101 115  99 114 105 112 116 105 111 110  32 125
 125 123  37  32 101 110 100 105 102  32  37 125 123  37  32 101 110 100
 105 102  32  37 125  60 124  98 111 120  95 101 110 100 124  62 123  37
  32 101 110 100 102 111 114  32  37 125  60 124 102 105 109  95 115 117
 102 102 105 120 124  62]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## Laya-Q8_0.gguf
**Pfad:** `ggml-org\Laya-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[201]` |
| `GGUF.kv_count` | `[53]` |
| `general.architecture` | `[109 111 100 101 114 110  45  98 101 114 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 76  97 121  97]` |
| `general.size_label` | `[52 50 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45  99 108  97 115 115 105 102 105  99  97 116 105 111  110]` |
| `modern-bert.block_count` | `[30]` |
| `modern-bert.context_length` | `[8192]` |
| `modern-bert.embedding_length` | `[1024]` |
| `modern-bert.feed_forward_length` | `[4096]` |
| `modern-bert.attention.head_count` | `[16]` |
| `modern-bert.rope.freq_base` | `[160000.]` |
| `modern-bert.rope.freq_base_swa` | `[10000.]` |
| `modern-bert.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `modern-bert.attention.layer_norm_epsilon` | `[1.e-05]` |
| `modern-bert.attention.causal` | `[False]` |
| `modern-bert.classifier.pooling_type` | `[1]` |
| `modern-bert.attention.sliding_window` | `[128]` |
| `modern-bert.attention.sliding_window_pattern` | `[3]` |
| `modern-bert.rope.scaling.type` | `[110 111 110 101]` |
| `modern-bert.vocab_size` | `[50368]` |
| `modern-bert.hidden_activation` | `[103 101 108 117]` |
| `modern-bert.decision.type` | `[108  97 121  97]` |
| `modern-bert.decision.block_count` | `[2]` |
| `modern-bert.decision.max_head_tokens` | `[192]` |
| `modern-bert.decision.temperature.choice` | `[1.636903]` |
| `modern-bert.decision.temperature.score` | `[1.25143]` |
| `modern-bert.decision.temperature.noul` | `[1.9833995]` |
| `modern-bert.decision.temperature.choice.3_5` | `[1.7601519]` |
| `modern-bert.decision.temperature.choice.6_10` | `[1.0000159]` |
| `modern-bert.decision.temperature.score.3_5` | `[1.25143]` |
| `modern-bert.decision.temperature.noul.2` | `[1.9833995]` |
| `modern-bert.decision.temperature.choice.11` | `[0.10058281]` |
| `modern-bert.decision.temperature.choice.2` | `[1.9063563]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[109 111 100 101 114 110  45  98 101 114 116]` |
| `tokenizer.ggml.tokens` | `[ 91 117 110 117 115 101 100  56  50  93]` |
| `tokenizer.ggml.token_type` | `[4]` |
| `tokenizer.ggml.merges` | `[196 160  79 117 116  32  99 111 109 101 115]` |
| `tokenizer.ggml.bos_token_id` | `[50281]` |
| `tokenizer.ggml.eos_token_id` | `[50282]` |
| `tokenizer.ggml.unknown_token_id` | `[50280]` |
| `tokenizer.ggml.seperator_token_id` | `[50282]` |
| `tokenizer.ggml.padding_token_id` | `[50283]` |
| `tokenizer.ggml.mask_token_id` | `[50284]` |
| `tokenizer.ggml.token_type_count` | `[3]` |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template.systemone:**
```jinja
[ 91  67  76  83  93 123 123  32 116 121 112 101  32 125 125  32 113 117
 101 115 116 105 111 110  58  32 123 123  32 105 110 115 116 114 117  99
 116 105 111 110 115  32 105 102  32 105 110 115 116 114 117  99 116 105
 111 110 115  32 105 115  32 115 116 114 105 110 103  32 101 108 115 101
  32 105 110 115 116 114 117  99 116 105 111 110 115  32 124  32 116 111
 106 115 111 110  32 125 125  91  83  69  80  93 123  37  32 102 111 114
  32 111  32 105 110  32 111 112 116 105 111 110 115  32  37 125  91  77
  65  83  75  93  32 123  37  32 105 102  32 116 121 112 101  32  61  61
  32  39  99 104 111 105  99 101  39  32  37 125 123 123  32 111  46 107
 101 121  32 125 125 123  37  32 105 102  32 111  46 100 101 115  99 114
 105 112 116 105 111 110  32  37 125  58  32 123 123  32 111  46 100 101
 115  99 114 105 112 116 105 111 110  32 105 102  32 111  46 100 101 115
  99 114 105 112 116 105 111 110  32 105 115  32 115 116 114 105 110 103
  32 101 108 115 101  32 111  46 100 101 115  99 114 105 112 116 105 111
 110  32 124  32 116 111 106 115 111 110  32 125 125 123  37  32 101 110
 100 105 102  32  37 125 123  37  32 101 108 105 102  32 116 121 112 101
  32  61  61  32  39 115  99 111 114 101  39  32  37 125 108 101 118 101
 108  32 123 123  32 111  46 107 101 121  32 125 125  58  32 123 123  32
 111  46 100 101 115  99 114 105 112 116 105 111 110  32 105 102  32 111
  46 100 101 115  99 114 105 112 116 105 111 110  32 105 115  32 115 116
 114 105 110 103  32 101 108 115 101  32 111  46 100 101 115  99 114 105
 112 116 105 111 110  32 124  32 116 111 106 115 111 110  32 125 125 123
  37  32 101 108 115 101  32  37 125 123 123  32 111  46 107 101 121  32
 125 125  58  32 123  37  32 105 102  32 111  46 100 101 115  99 114 105
 112 116 105 111 110  32  37 125 123 123  32 111  46 100 101 115  99 114
 105 112 116 105 111 110  32 105 102  32 111  46 100 101 115  99 114 105
 112 116 105 111 110  32 105 115  32 115 116 114 105 110 103  32 101 108
 115 101  32 111  46 100 101 115  99 114 105 112 116 105 111 110  32 124
  32 116 111 106 115 111 110  32 125 125 123  37  32 101 108 105 102  32
 111  46 107 101 121  32  61  61  32  39 116 114 117 101  39  32  37 125
 121 101 115  44  32 116 104 101  32 115 116  97 116 101 109 101 110 116
  32 104 111 108 100 115 123  37  32 101 108 115 101  32  37 125 110 111
  44  32 116 104 101  32 115 116  97 116 101 109 101 110 116  32 100 111
 101 115  32 110 111 116  32 104 111 108 100 123  37  32 101 110 100 105
 102  32  37 125 123  37  32 101 110 100 105 102  32  37 125 123  37  32
 101 110 100 102 111 114  32  37 125  91  83  69  80  93 123 123  32 115
 116  97 116 101  32 105 102  32 115 116  97 116 101  32 105 115  32 115
 116 114 105 110 103  32 101 108 115 101  32 115 116  97 116 101  32 124
  32 116 111 106 115 111 110  32 125 125  91  83  69  80  93]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## mmproj-Clef-Flash-Q8_0.gguf
**Pfad:** `ggml-org\mmproj-Clef-Flash-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[27]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 67 108 101 102  45  70 108  97 115 104]` |
| `general.size_label` | `[52 53 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  53  32  57  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[7]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  57  66]
```

---

## embeddinggemma-2-Q8_0.gguf
**Pfad:** `Google\embeddinggemma-2\embeddinggemma-2-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[413]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  45 101 109  98 101 100 100 105 110 103  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[101 109  98 101 100 100 105 110 103 103 101 109 109  97  45  50]` |
| `general.size_label` | `[50 55 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[102 101  97 116 117 114 101  45 101 120 116 114  97  99 116 105 111 110]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `gemma-embedding2.block_count` | `[24]` |
| `gemma-embedding2.context_length` | `[262144]` |
| `gemma-embedding2.embedding_length` | `[512]` |
| `gemma-embedding2.feed_forward_length` | `[2048]` |
| `gemma-embedding2.attention.head_count` | `[4]` |
| `gemma-embedding2.attention.head_count_kv` | `[1]` |
| `gemma-embedding2.rope.freq_base` | `[1.e+06]` |
| `gemma-embedding2.rope.freq_base_swa` | `[10000.]` |
| `gemma-embedding2.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma-embedding2.attention.key_length` | `[512]` |
| `gemma-embedding2.attention.value_length` | `[512]` |
| `gemma-embedding2.attention.sliding_window` | `[1024]` |
| `gemma-embedding2.attention.shared_kv_layers` | `[0]` |
| `gemma-embedding2.embedding_length_per_layer_input` | `[512]` |
| `gemma-embedding2.attention.sliding_window_pattern` | `[False]` |
| `gemma-embedding2.attention.key_length_swa` | `[256]` |
| `gemma-embedding2.attention.value_length_swa` | `[256]` |
| `gemma-embedding2.rope.dimension_count` | `[512]` |
| `gemma-embedding2.rope.dimension_count_swa` | `[256]` |
| `gemma-embedding2.embedding_length_out` | `[768]` |
| `gemma-embedding2.attention.causal` | `[False]` |
| `gemma-embedding2.pooling_type` | `[1]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

---

## diffusiongemma-26B-A4B-it-Q8_0.gguf
**Pfad:** `Google\Gemma 4 Diffusion\diffusiongemma-26B-A4B-it-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[692]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[100 105 102 102 117 115 105 111 110  45 103 101 109 109  97]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 68 103  95  82  99  48  80  49  95  80  97 116  99 104 101 100]` |
| `general.size_label` | `[ 49  50  56 120  50  46  54  66]` |
| `diffusion-gemma.block_count` | `[30]` |
| `diffusion-gemma.context_length` | `[262144]` |
| `diffusion-gemma.embedding_length` | `[2816]` |
| `diffusion-gemma.feed_forward_length` | `[2112]` |
| `diffusion-gemma.attention.head_count` | `[16]` |
| `diffusion-gemma.attention.head_count_kv` | `[2]` |
| `diffusion-gemma.rope.freq_base` | `[1.e+06]` |
| `diffusion-gemma.rope.freq_base_swa` | `[10000.]` |
| `diffusion-gemma.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `diffusion-gemma.expert_count` | `[128]` |
| `diffusion-gemma.expert_used_count` | `[8]` |
| `diffusion-gemma.attention.key_length` | `[512]` |
| `diffusion-gemma.attention.value_length` | `[512]` |
| `diffusion-gemma.final_logit_softcapping` | `[30.]` |
| `diffusion-gemma.attention.sliding_window` | `[1024]` |
| `diffusion-gemma.attention.shared_kv_layers` | `[0]` |
| `diffusion-gemma.embedding_length_per_layer_input` | `[0]` |
| `diffusion-gemma.attention.sliding_window_pattern` | `[False]` |
| `diffusion-gemma.attention.key_length_swa` | `[256]` |
| `diffusion-gemma.attention.value_length_swa` | `[256]` |
| `diffusion-gemma.expert_feed_forward_length` | `[704]` |
| `diffusion-gemma.rope.dimension_count` | `[512]` |
| `diffusion-gemma.rope.dimension_count_swa` | `[256]` |
| `diffusion-gemma.attention.causal` | `[False]` |
| `diffusion.canvas_length` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

---

## gemma-4-12b-it-v2-UD-Q8_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-12b-it-v2-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[667]` |
| `GGUF.kv_count` | `[58]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[49 50 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  98  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4.block_count` | `[48]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[3840]` |
| `gemma4.feed_forward_length` | `[15360]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[1]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.suppress_tokens` | `[258882]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[328]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  98  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  49  50  98  45 105 116  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  49  50  98  45 105 116  46 116
 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## gemma-4-26B-A4B-it-v2-UD-Q8_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-26B-A4B-it-v2-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[658]` |
| `GGUF.kv_count` | `[60]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 54 66 45 65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[30]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[2816]` |
| `gemma4.feed_forward_length` | `[2112]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[2]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.expert_count` | `[128]` |
| `gemma4.expert_used_count` | `[8]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.expert_feed_forward_length` | `[704]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[295]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45 105 116
  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108
 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45
 105 116  46 116 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## gemma-4-31B-it-v2-UD-Q5_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-31B-it-v2-UD-Q5_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[833]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[60]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[5376]` |
| `gemma4.feed_forward_length` | `[21504]` |
| `gemma4.attention.head_count` | `[32]` |
| `gemma4.attention.head_count_kv` | `[4]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[17]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[410]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  51  49  66  45 105 116  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  51  49  66  45 105 116  46 116
 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## mmproj-gemma-4-12b-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-12b-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[11]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.finetune` | `[ 49  50  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[53 50 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  98  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[3840]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[3840]` |
| `clip.vision.feed_forward_length` | `[0]` |
| `clip.vision.block_count` | `[0]` |
| `clip.vision.attention.head_count` | `[1]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[3840]` |
| `clip.audio.embedding_length` | `[640]` |
| `clip.audio.feed_forward_length` | `[0]` |
| `clip.audio.block_count` | `[0]` |
| `clip.audio.attention.head_count` | `[1]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 117 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.audio.projector_type` | `[103 101 109 109  97  52 117  97]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  98  45 105 116]
```

---

## mmproj-gemma-4-26B-A4B-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-26B-A4B-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[30]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.finetune` | `[ 50  54  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.size_label` | `[65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2816]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

---

## mmproj-gemma-4-31b-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-31b-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[30]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.finetune` | `[ 51  49  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.size_label` | `[53 55 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5376]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

---

## mtp-gemma-4-12b-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-12b-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 49  50  98  45 105 116  45  97 115 115 105 115 116  97 110 116]` |
| `general.basename` | `[103 101 109 109  97  45  52]` |
| `general.size_label` | `[52 50 51 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[1]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[3840]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |

**general.name:**
```jinja
[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116  32  65 115 115
 105 115 116  97 110 116]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-26B-A4B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-26B-A4B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[50 54 66 32 65 52 66]` |
| `general.finetune` | `[50 54 66]` |
| `general.size_label` | `[65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[2]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[2816]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-31B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-31B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[51 49 66]` |
| `general.finetune` | `[51 49 66]` |
| `general.size_label` | `[52 55 48 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[32]` |
| `gemma4-assistant.attention.head_count_kv` | `[4]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[5376]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## gemma-4-12b-it-v2-qat-q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-12b-it-v2-qat-q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[667]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[49 50 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4.block_count` | `[48]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[3840]` |
| `gemma4.feed_forward_length` | `[15360]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[1]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.suppress_tokens` | `[258882]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## gemma-4-26B-it-v2-mmproj.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-26B-it-v2-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[28]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 55 51 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2816]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

---

## gemma-4-26B-it-v2_q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-26B-it-v2_q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[658]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[ 49  50  56 120  50  46  54  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[30]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[2816]` |
| `gemma4.feed_forward_length` | `[2112]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[2]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.expert_count` | `[128]` |
| `gemma4.expert_used_count` | `[8]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.expert_feed_forward_length` | `[704]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## gemma-4-31B-it-v2-mmproj.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-31B-it-v2-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[28]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 55 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5376]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

---

## gemma-4-31B-it-v2_q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-31B-it-v2_q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[833]` |
| `GGUF.kv_count` | `[49]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[51 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[60]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[5376]` |
| `gemma4.feed_forward_length` | `[21504]` |
| `gemma4.attention.head_count` | `[32]` |
| `gemma4.attention.head_count_kv` | `[4]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## mmproj-gemma-4-12b-it-v2-qat-q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\mmproj-gemma-4-12b-it-v2-qat-q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[11]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 50 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[3840]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[3840]` |
| `clip.vision.feed_forward_length` | `[0]` |
| `clip.vision.block_count` | `[0]` |
| `clip.vision.attention.head_count` | `[0]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[3840]` |
| `clip.audio.embedding_length` | `[640]` |
| `clip.audio.feed_forward_length` | `[0]` |
| `clip.audio.block_count` | `[0]` |
| `clip.audio.attention.head_count` | `[0]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 117 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.audio.projector_type` | `[103 101 109 109  97  52 117  97]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  66  45 105 116]
```

---

## mtp-gemma-4-12b-it-v2-qat.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\mtp-gemma-4-12b-it-v2-qat.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 49  50  98  45 105 116  45  97 115 115 105 115 116  97 110 116]` |
| `general.basename` | `[103 101 109 109  97  45  52]` |
| `general.size_label` | `[52 50 51 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[1]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[3840]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |

**general.name:**
```jinja
[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116  32  65 115 115
 105 115 116  97 110 116]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-26B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\mtp-gemma-4-26B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[50 54 66 32 65 52 66]` |
| `general.finetune` | `[50 54 66]` |
| `general.size_label` | `[65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[2]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[2816]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-31B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\mtp-gemma-4-31B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[51 49 66]` |
| `general.finetune` | `[51 49 66]` |
| `general.size_label` | `[52 55 48 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[32]` |
| `gemma4-assistant.attention.head_count_kv` | `[4]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[5376]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## granite-4.2-30b-Q4_K_M.gguf
**Pfad:** `IBM\granite-4.2-30b-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[579]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 114  97 110 105 116 101  32  52  46  50  32  51  48  98]` |
| `general.basename` | `[103 114  97 110 105 116 101  45  52  46  50]` |
| `general.size_label` | `[51 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `granite.block_count` | `[64]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[32768]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[5.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[1.]` |
| `granite.residual_scale` | `[1.]` |
| `granite.logit_scale` | `[1.]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 114  97 110 105 116 101  45 100 111  99 108 105 110 103]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100283]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100257]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.name:**
```jinja
[ 71 114  97 110 105 116 101  32  52  46  49  32  51  48  98  32  66  97
 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  51  48  98  45  98  97 115
 101]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## granite-4.2-30b-Q6_K.gguf
**Pfad:** `IBM\granite-4.2-30b-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[579]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 114  97 110 105 116 101  32  52  46  50  32  51  48  98]` |
| `general.basename` | `[103 114  97 110 105 116 101  45  52  46  50]` |
| `general.size_label` | `[51 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `granite.block_count` | `[64]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[32768]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[5.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[1.]` |
| `granite.residual_scale` | `[1.]` |
| `granite.logit_scale` | `[1.]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 114  97 110 105 116 101  45 100 111  99 108 105 110 103]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100283]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100257]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**general.base_model.0.name:**
```jinja
[ 71 114  97 110 105 116 101  32  52  46  49  32  51  48  98  32  66  97
 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  51  48  98  45  98  97 115
 101]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## granite-4.2-8b-bf16.gguf
**Pfad:** `IBM\granite-4.2-8b-bf16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 114  97 110 105 116 101  32  52  46  50  32  56  98]` |
| `general.basename` | `[103 114  97 110 105 116 101  45  52  46  50]` |
| `general.size_label` | `[56 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 114  97 110 105 116 101  32  52  46  49  32  56  98  32  66  97 115  101]` |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `granite.block_count` | `[40]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[12800]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[1.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `general.file_type` | `[32]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[1.]` |
| `granite.residual_scale` | `[1.]` |
| `granite.logit_scale` | `[1.]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 114  97 110 105 116 101  45 100 111  99 108 105 110 103]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100283]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100257]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  56  98  45  98  97 115 101]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## granite-4.2-8b-Q8_0.gguf
**Pfad:** `IBM\granite-4.2-8b-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 114  97 110 105 116 101  32  52  46  50  32  56  98]` |
| `general.basename` | `[103 114  97 110 105 116 101  45  52  46  50]` |
| `general.size_label` | `[56 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 114  97 110 105 116 101  32  52  46  49  32  56  98  32  66  97 115  101]` |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `granite.block_count` | `[40]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[12800]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[1.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[1.]` |
| `granite.residual_scale` | `[1.]` |
| `granite.logit_scale` | `[1.]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 114  97 110 105 116 101  45 100 111  99 108 105 110 103]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100283]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100257]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  56  98  45  98  97 115 101]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## K2-Horizon-MoVA-36B-A4B-Q8_0.gguf
**Pfad:** `IFM\K2-Horizon-MoVA-36B-A4B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[798]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[107  50  45 104 111 114 105 122 111 110]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 67 104 101  99 107 112 111 105 110 116  95  48  48  48  50  53  48  48]` |
| `general.size_label` | `[ 57  49 120  51  46  55  66]` |
| `k2-horizon.block_count` | `[48]` |
| `k2-horizon.context_length` | `[524288]` |
| `k2-horizon.embedding_length` | `[2560]` |
| `k2-horizon.feed_forward_length` | `[6144]` |
| `k2-horizon.attention.head_count` | `[32]` |
| `k2-horizon.attention.head_count_kv` | `[8]` |
| `k2-horizon.rope.freq_base` | `[1.e+07]` |
| `k2-horizon.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `k2-horizon.expert_count` | `[100]` |
| `k2-horizon.expert_used_count` | `[8]` |
| `k2-horizon.attention.key_length` | `[128]` |
| `k2-horizon.attention.value_length` | `[128]` |
| `k2-horizon.attention.group_norm_groups` | `[2]` |
| `k2-horizon.rope.dimension_count` | `[128]` |
| `k2-horizon.expert_feed_forward_length` | `[768]` |
| `k2-horizon.leading_dense_block_count` | `[3]` |
| `k2-horizon.moe_every_n_layers` | `[1]` |
| `k2-horizon.expert_shared_count` | `[1]` |
| `k2-horizon.expert_weights_norm` | `[ True]` |
| `k2-horizon.expert_shared_feed_forward_length` | `[768]` |
| `k2-horizon.expert_weights_scale` | `[2.5]` |
| `k2-horizon.expert_gating_func` | `[2]` |
| `k2-horizon.attention.value_expert_count` | `[64]` |
| `k2-horizon.attention.value_expert_used_count` | `[4]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[107  50  45 104 111 114 105 122 111 110]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 38  32 115 121 110  99]` |
| `tokenizer.ggml.bos_token_id` | `[0]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.seperator_token_id` | `[1]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[ True]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.tokens:**
```jinja
[114 101 115 101 114 118 101 100  95 115 112 101  99 105  97 108  95 116
 111 107 101 110  95  53  55  51]
```

**tokenizer.chat_template:**
```jinja
[123 123  45 ...  37 125  10]
```

---

## Ling-3.0-flash-AD-IQ3_M-00001-of-00002.gguf
**Pfad:** `inclusionAI\Ling-3.0-flash-AD-IQ3_M-00001-of-00002.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[682]` |
| `GGUF.kv_count` | `[58]` |
| `general.architecture` | `[ 98  97 105 108 105 110 103 109 111 101  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 76 105 110 103  32  51  46  48  32  70 108  97 115 104]` |
| `general.size_label` | `[ 53  49  50 120  51  46  57  66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `bailingmoe3.block_count` | `[42]` |
| `bailingmoe3.context_length` | `[131072]` |
| `bailingmoe3.embedding_length` | `[2560]` |
| `bailingmoe3.feed_forward_length` | `[6144]` |
| `bailingmoe3.attention.head_count` | `[32]` |
| `bailingmoe3.attention.head_count_kv` | `[1]` |
| `bailingmoe3.rope.freq_base` | `[6.e+06]` |
| `bailingmoe3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `bailingmoe3.expert_count` | `[512]` |
| `bailingmoe3.expert_used_count` | `[8]` |
| `bailingmoe3.expert_group_count` | `[8]` |
| `bailingmoe3.expert_group_used_count` | `[4]` |
| `bailingmoe3.expert_gating_func` | `[2]` |
| `bailingmoe3.attention.key_length` | `[576]` |
| `bailingmoe3.attention.value_length` | `[512]` |
| `bailingmoe3.vocab_size` | `[157184]` |
| `bailingmoe3.ssm.conv_kernel` | `[4]` |
| `bailingmoe3.kda.head_dim` | `[128]` |
| `bailingmoe3.attention.kv_lora_rank` | `[512]` |
| `bailingmoe3.attention.key_length_mla` | `[192]` |
| `bailingmoe3.attention.value_length_mla` | `[128]` |
| `bailingmoe3.rope.dimension_count` | `[64]` |
| `bailingmoe3.leading_dense_block_count` | `[2]` |
| `bailingmoe3.expert_feed_forward_length` | `[768]` |
| `bailingmoe3.expert_shared_feed_forward_length` | `[768]` |
| `bailingmoe3.expert_shared_count` | `[1]` |
| `bailingmoe3.expert_weights_scale` | `[2.5]` |
| `bailingmoe3.expert_weights_norm` | `[ True]` |
| `bailingmoe3.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[ 98  97 105 108 105 110 103 109 111 101  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 55 49 56 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  97 110  32 108  97 121]` |
| `tokenizer.ggml.bos_token_id` | `[156891]` |
| `tokenizer.ggml.eos_token_id` | `[156895]` |
| `tokenizer.ggml.padding_token_id` | `[156892]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `bailingmoe3.kda.gate_lower_bound` | `[-5.]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[27]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[573]` |
| `quantize.imatrix.chunks_count` | `[522]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[917]` |
| `split.count` | `[2]` |
| `bailingmoe3.swiglu_clamp_exp` | `[4.]` |
| `bailingmoe3.swiglu_clamp_shexp` | `[7.]` |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 114 111 111 116  47 119 111 114 107  47 108 105 110 103  51  45 105
 109  97 116 114 105 120  45  98 102  49  54  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 114 111 111 116  47 119 111 114 107  47  99  97 108 105  98  47  99
  97 108 105  98  95 116 114  97 105 110  46 116 120 116]
```

---

## Ling-3.0-flash-AD-IQ3_M-00002-of-00002.gguf
**Pfad:** `inclusionAI\Ling-3.0-flash-AD-IQ3_M-00002-of-00002.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[235]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[917]` |
| `split.count` | `[2]` |

---

## dflash-raon-1-mini-preview-Q8_0.gguf
**Pfad:** `KRAFTON\dflash-raon-1-mini-preview-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[81]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.size_label` | `[54 55 56 77]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[2560]` |
| `dflash.feed_forward_length` | `[8192]` |
| `dflash.attention.head_count` | `[20]` |
| `dflash.attention.head_count_kv` | `[5]` |
| `dflash.attention.causal` | `[False]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `general.file_type` | `[7]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[256]` |
| `dflash.selector_top_k` | `[16]` |
| `dflash.target_layers` | `[22]` |
| `dflash.target_hidden_size` | `[10240]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[114  97 111 110]` |
| `tokenizer.ggml.tokens` | `[ 60 124 114 101 115 101 114 118 101 100  95  54  50  52 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[116 196 160  98 101  32 196 160 116 104  97 116]` |
| `tokenizer.ggml.eos_token_id` | `[259202]` |
| `tokenizer.ggml.padding_token_id` | `[259200]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `tokenizer.ggml.mask_token_id` | `[259839]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## raon-1-mini-preview-Q6_K.gguf
**Pfad:** `KRAFTON\raon-1-mini-preview-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[731]` |
| `GGUF.kv_count` | `[61]` |
| `general.architecture` | `[114  97 111 110  49]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 82  97 111 110  32  49  32  77 105 110 105  32  80 114 101 118 105 101  119]` |
| `general.basename` | `[ 82  97 111 110  45  49]` |
| `general.size_label` | `[109 105 110 105]` |
| `raon1.block_count` | `[25]` |
| `raon1.context_length` | `[262144]` |
| `raon1.embedding_length` | `[2560]` |
| `raon1.feed_forward_length` | `[12032]` |
| `raon1.attention.head_count` | `[40]` |
| `raon1.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `raon1.expert_count` | `[128]` |
| `raon1.expert_used_count` | `[6]` |
| `raon1.expert_gating_func` | `[2]` |
| `raon1.attention.key_length` | `[576]` |
| `raon1.attention.value_length` | `[512]` |
| `raon1.vocab_size` | `[259840]` |
| `raon1.attention.head_count_kv` | `[1]` |
| `raon1.ssm.conv_kernel` | `[4]` |
| `raon1.ssm.time_step_rank` | `[128]` |
| `raon1.kda.head_dim` | `[128]` |
| `raon1.kda.gate_lower_bound` | `[-5.]` |
| `raon1.attention.q_lora_rank` | `[576]` |
| `raon1.attention.kv_lora_rank` | `[512]` |
| `raon1.rope.dimension_count` | `[64]` |
| `raon1.attention.key_length_mla` | `[192]` |
| `raon1.attention.value_length_mla` | `[128]` |
| `raon1.attention.latent_norm_rms_epsilon` | `[1.e-06]` |
| `raon1.expert_feed_forward_length` | `[1088]` |
| `raon1.expert_shared_count` | `[1]` |
| `raon1.leading_dense_block_count` | `[1]` |
| `raon1.expert_weights_scale` | `[1.]` |
| `raon1.expert_weights_norm` | `[ True]` |
| `raon1.activation.situ_beta` | `[4.]` |
| `raon1.activation.situ_linear_beta` | `[25.]` |
| `raon1.hyper_connection.count` | `[4]` |
| `raon1.hyper_connection.epsilon` | `[1.e-05]` |
| `raon1.hyper_connection.magnitude` | `[2.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[114  97 111 110]` |
| `tokenizer.ggml.tokens` | `[ 60 124 114 101 115 101 114 118 101 100  95  54  50  52 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[116 196 160  98 101  32 196 160 116 104  97 116]` |
| `tokenizer.ggml.eos_token_id` | `[259202]` |
| `tokenizer.ggml.padding_token_id` | `[259200]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[432]` |
| `quantize.imatrix.chunks_count` | `[99]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.finetune` | `[112 114 101 118 105 101 119]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.sampling.temp` | `[1.]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |

**quantize.imatrix.file:**
```jinja
[114  97 111 110  45  49  45 109 105 110 105  45 112 114 101 118 105 101
 119  46 105 109  97 116 114 105 120]
```

**quantize.imatrix.dataset:**
```jinja
[ 99  97 108 105  98 114  97 116 105 111 110  95 109 105 120 101 100  95
 118  49  46 116 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## raon-1-nano-F16_MTP.gguf
**Pfad:** `KRAFTON\raon-1-nano-F16_MTP.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[520]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[114  97 111 110  49 110  97 110 111]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 82  97 111 110  45  49  32 110  97 110 111]` |
| `general.size_label` | `[51 46 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `raon1nano.block_count` | `[25]` |
| `raon1nano.context_length` | `[262144]` |
| `raon1nano.embedding_length` | `[1536]` |
| `raon1nano.feed_forward_length` | `[6144]` |
| `raon1nano.attention.head_count` | `[32]` |
| `raon1nano.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `raon1nano.attention.key_length` | `[576]` |
| `raon1nano.attention.value_length` | `[512]` |
| `general.file_type` | `[1]` |
| `raon1nano.vocab_size` | `[259840]` |
| `raon1nano.attention.head_count_kv` | `[1]` |
| `raon1nano.ssm.conv_kernel` | `[4]` |
| `raon1nano.ssm.time_step_rank` | `[128]` |
| `raon1nano.kda.head_dim` | `[128]` |
| `raon1nano.kda.gate_lower_bound` | `[-5.]` |
| `raon1nano.attention.q_lora_rank` | `[512]` |
| `raon1nano.attention.kv_lora_rank` | `[512]` |
| `raon1nano.rope.dimension_count` | `[64]` |
| `raon1nano.attention.key_length_mla` | `[192]` |
| `raon1nano.attention.value_length_mla` | `[128]` |
| `raon1nano.attention.latent_norm_rms_epsilon` | `[1.e-06]` |
| `raon1nano.embedding_length_per_layer_input` | `[256]` |
| `raon1nano.hyper_connection.count` | `[4]` |
| `raon1nano.hyper_connection.epsilon` | `[1.e-05]` |
| `raon1nano.hyper_connection.magnitude` | `[1.]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[114  97 111 110]` |
| `tokenizer.ggml.tokens` | `[ 60 124 114 101 115 101 114 118 101 100  95  54  50  52 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[116 196 160  98 101  32 196 160 116 104  97 116]` |
| `tokenizer.ggml.eos_token_id` | `[259202]` |
| `tokenizer.ggml.padding_token_id` | `[259200]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `raon1nano.nextn_predict_layers` | `[1]` |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## raon-1-nano-Q8_0_MTP.gguf
**Pfad:** `KRAFTON\raon-1-nano-Q8_0_MTP.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[520]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[114  97 111 110  49 110  97 110 111]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 82  97 111 110  45  49  32 110  97 110 111]` |
| `general.size_label` | `[51 46 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `raon1nano.block_count` | `[25]` |
| `raon1nano.context_length` | `[262144]` |
| `raon1nano.embedding_length` | `[1536]` |
| `raon1nano.feed_forward_length` | `[6144]` |
| `raon1nano.attention.head_count` | `[32]` |
| `raon1nano.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `raon1nano.attention.key_length` | `[576]` |
| `raon1nano.attention.value_length` | `[512]` |
| `raon1nano.vocab_size` | `[259840]` |
| `raon1nano.attention.head_count_kv` | `[1]` |
| `raon1nano.ssm.conv_kernel` | `[4]` |
| `raon1nano.ssm.time_step_rank` | `[128]` |
| `raon1nano.kda.head_dim` | `[128]` |
| `raon1nano.kda.gate_lower_bound` | `[-5.]` |
| `raon1nano.attention.q_lora_rank` | `[512]` |
| `raon1nano.attention.kv_lora_rank` | `[512]` |
| `raon1nano.rope.dimension_count` | `[64]` |
| `raon1nano.attention.key_length_mla` | `[192]` |
| `raon1nano.attention.value_length_mla` | `[128]` |
| `raon1nano.attention.latent_norm_rms_epsilon` | `[1.e-06]` |
| `raon1nano.embedding_length_per_layer_input` | `[256]` |
| `raon1nano.hyper_connection.count` | `[4]` |
| `raon1nano.hyper_connection.epsilon` | `[1.e-05]` |
| `raon1nano.hyper_connection.magnitude` | `[1.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[114  97 111 110]` |
| `tokenizer.ggml.tokens` | `[ 60 124 114 101 115 101 114 118 101 100  95  54  50  52 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[116 196 160  98 101  32 196 160 116 104  97 116]` |
| `tokenizer.ggml.eos_token_id` | `[259202]` |
| `tokenizer.ggml.padding_token_id` | `[259200]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `raon1nano.nextn_predict_layers` | `[1]` |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## d1-3B-Q8_0.gguf
**Pfad:** `LiquidAI\d1-3B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[266]` |
| `GGUF.kv_count` | `[40]` |
| `general.architecture` | `[108 102 109  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[100  49  45  51  66]` |
| `general.basename` | `[100  49]` |
| `general.size_label` | `[51 66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[108 102 109  49  46  48]` |
| `general.license.link` | `[76 73 67 69 78 83 69]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[76 70 77 50 46 53 32 86 76 32 51 66]` |
| `general.base_model.0.organization` | `[ 76 105 113 117 105 100  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[118 105]` |
| `lfm2.block_count` | `[30]` |
| `lfm2.context_length` | `[32768]` |
| `lfm2.embedding_length` | `[2048]` |
| `lfm2.feed_forward_length` | `[10752]` |
| `lfm2.attention.head_count` | `[32]` |
| `lfm2.attention.head_count_kv` | `[0]` |
| `lfm2.rope.freq_base` | `[1.e+06]` |
| `lfm2.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `lfm2.vocab_size` | `[128000]` |
| `lfm2.shortconv.l_cache` | `[3]` |
| `lfm2.decision.type` | `[108 102 109  50  45 100  49]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 102 109  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 55 57 57 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[110  97  32  98 104 195 131 194 164 110 103 105 103]` |
| `tokenizer.ggml.bos_token_id` | `[124894]` |
| `tokenizer.ggml.eos_token_id` | `[124900]` |
| `tokenizer.ggml.padding_token_id` | `[124893]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  76 105 113 117 105 100  65  73  47  76  70  77  50
  46  53  45  86  76  45  51  66]
```

**tokenizer.chat_template:**
```jinja
[123 123  45 ...  37 125  10]
```

**tokenizer.chat_template.systemone:**
```jinja
[ 60 124 115 ...  32 125 125]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## d1-omni-600M-Q8_0.gguf
**Pfad:** `LiquidAI\d1-omni-600M-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[179]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[108 102 109  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[100  49  45 111 109 110 105  45  54  48  48  77]` |
| `general.basename` | `[100  49  45 111 109 110 105]` |
| `general.size_label` | `[54 48 48 77]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[108 102 109  49  46  48]` |
| `general.license.link` | `[76 73 67 69 78 83 69]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 76  70  77  50  46  53  32  69 110  99 111 100 101 114  32  51  53  48   77]` |
| `general.base_model.0.organization` | `[ 76 105 113 117 105 100  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `lfm2.block_count` | `[18]` |
| `lfm2.context_length` | `[128000]` |
| `lfm2.embedding_length` | `[1024]` |
| `lfm2.feed_forward_length` | `[4096]` |
| `lfm2.attention.head_count` | `[16]` |
| `lfm2.attention.head_count_kv` | `[16]` |
| `lfm2.rope.freq_base` | `[1.e+06]` |
| `lfm2.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `lfm2.vocab_size` | `[65536]` |
| `lfm2.shortconv.l_cache` | `[3]` |
| `lfm2.attention.layer_norm_epsilon` | `[1.e-05]` |
| `lfm2.attention.causal` | `[False]` |
| `lfm2.decision.type` | `[108 102 109  50  45 100  49  45 111 109 110 105]` |
| `lfm2.decision.block_count` | `[2]` |
| `lfm2.decision.temperature.choice.11` | `[1.3725151]` |
| `lfm2.decision.temperature.choice.2` | `[1.7465146]` |
| `lfm2.decision.temperature.choice.3_5` | `[1.3998982]` |
| `lfm2.decision.temperature.choice.6_10` | `[1.1751071]` |
| `lfm2.decision.temperature.noul.2` | `[1.6663224]` |
| `lfm2.decision.temperature.score.3_5` | `[1.7301133]` |
| `lfm2.decision.temperature.score.6_10` | `[1.]` |
| `lfm2.decision.temperature.choice` | `[1.]` |
| `lfm2.decision.temperature.score` | `[1.]` |
| `lfm2.decision.temperature.noul` | `[1.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 102 109  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 54 53 53 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  99 114  99  32  51  50]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[7]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[16]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.token_type_count` | `[3]` |
| `tokenizer.chat_template.systemone` | *Siehe Code-Block unten* |
| `tokenizer.chat_templates` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  76 105 113 117 105 100  65  73  47  76  70  77  50
  46  53  45  69 110  99 111 100 101 114  45  51  53  48  77]
```

**tokenizer.chat_template.systemone:**
```jinja
[123  37  32 ...  49 124  62]
```

**tokenizer.chat_templates:**
```jinja
[115 121 115 116 101 109 111 110 101]
```

---

## LFM2.5-24B-A2B-Q6_K.gguf
**Pfad:** `LiquidAI\LFM2.5-24B-A2B-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[428]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[108 102 109  50 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[50]` |
| `general.sampling.temp` | `[0.1]` |
| `general.sampling.penalty_repeat` | `[1.05]` |
| `general.name` | `[76 70 77 50 46 53 45 50 52 66 45 65 50 66]` |
| `general.size_label` | `[ 54  52 120  49  46  52  66]` |
| `lfm2moe.block_count` | `[40]` |
| `lfm2moe.context_length` | `[131072]` |
| `lfm2moe.embedding_length` | `[2048]` |
| `lfm2moe.feed_forward_length` | `[11776]` |
| `lfm2moe.attention.head_count` | `[32]` |
| `lfm2moe.attention.head_count_kv` | `[0]` |
| `lfm2moe.rope.freq_base` | `[1.e+07]` |
| `lfm2moe.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `lfm2moe.expert_count` | `[64]` |
| `lfm2moe.expert_used_count` | `[4]` |
| `lfm2moe.expert_feed_forward_length` | `[1536]` |
| `lfm2moe.leading_dense_block_count` | `[2]` |
| `lfm2moe.expert_gating_func` | `[2]` |
| `lfm2moe.vocab_size` | `[128000]` |
| `lfm2moe.shortconv.l_cache` | `[3]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 102 109  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 55 57 57 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[110  97  32  98 104 195 131 194 164 110 103 105 103]` |
| `tokenizer.ggml.bos_token_id` | `[124894]` |
| `tokenizer.ggml.eos_token_id` | `[124900]` |
| `tokenizer.ggml.padding_token_id` | `[124893]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**tokenizer.chat_template:**
```jinja
[123 123  45 ...  37 125  10]
```

---

## mmproj-d1-3B-Q8_0.gguf
**Pfad:** `LiquidAI\mmproj-d1-3B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[441]` |
| `GGUF.kv_count` | `[29]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[100  49  45  51  66]` |
| `general.finetune` | `[51 98]` |
| `general.basename` | `[100  49]` |
| `general.size_label` | `[52 50 54 77]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[76 70 77 50 46 53 32 86 76 32 51 66]` |
| `general.base_model.0.organization` | `[ 76 105 113 117 105 100  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[7]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[256]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[108 102 109  50]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.image_resize_algo` | `[ 98 105  99 117  98 105  99]` |
| `general.quantization_version` | `[2]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  76 105 113 117 105 100  65  73  47  76  70  77  50
  46  53  45  86  76  45  51  66]
```

---

## mmproj-d1-omni-600M-Q8_0.gguf
**Pfad:** `LiquidAI\mmproj-d1-omni-600M-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[854]` |
| `GGUF.kv_count` | `[40]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[100  49  45 111 109 110 105  45  54  48  48  77]` |
| `general.finetune` | `[ 54  48  48 109]` |
| `general.basename` | `[100  49  45 111 109 110 105]` |
| `general.size_label` | `[50 48 54 77]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[108 102 109  49  46  48]` |
| `general.license.link` | `[76 73 67 69 78 83 69]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 76  70  77  50  46  53  32  69 110  99 111 100 101 114  32  51  53  48   77]` |
| `general.base_model.0.organization` | `[ 76 105 113 117 105 100  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `general.file_type` | `[7]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1024]` |
| `clip.vision.image_size` | `[256]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[768]` |
| `clip.vision.feed_forward_length` | `[3072]` |
| `clip.vision.block_count` | `[12]` |
| `clip.vision.attention.head_count` | `[12]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[1024]` |
| `clip.audio.embedding_length` | `[512]` |
| `clip.audio.feed_forward_length` | `[2048]` |
| `clip.audio.block_count` | `[17]` |
| `clip.audio.attention.head_count` | `[8]` |
| `clip.vision.projector_type` | `[100  49 111 109 110 105  95 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `clip.use_gelu` | `[ True]` |
| `clip.audio.projector_type` | `[100  49 111 109 110 105  95  97]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-05]` |
| `general.quantization_version` | `[2]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  76 105 113 117 105 100  65  73  47  76  70  77  50
  46  53  45  69 110  99 111 100 101 114  45  51  53  48  77]
```

---

## dflash-Muse-Glimmer-30B-kquant.gguf
**Pfad:** `Meta\dflash-Muse-Glimmer-30B-kquant.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[58]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 72 102  95  77 117 115 101 103 108 105 109 109 101 114]` |
| `general.size_label` | `[50 46 54 66]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[131072]` |
| `dflash.embedding_length` | `[6656]` |
| `dflash.feed_forward_length` | `[19968]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.rope.freq_base` | `[500000.]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.block_size` | `[16]` |
| `dflash.target_layers` | `[50]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 108  97 109  97  52]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 40 103  32 101]` |
| `tokenizer.ggml.bos_token_id` | `[200000]` |
| `tokenizer.ggml.eos_token_id` | `[200001]` |
| `tokenizer.ggml.padding_token_id` | `[200018]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[200008]` |
| `tokenizer.ggml.mask_token_id` | `[201818]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124 114 101 115 101 114 118 101 100  95 115 112 101  99 105  97 108
  95 116 111 107 101 110  95  50  48  52  55 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

---

## mmproj-Muse-Glimmer-30B-BF16.gguf
**Pfad:** `Meta\mmproj-Muse-Glimmer-30B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[809]` |
| `GGUF.kv_count` | `[19]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 77 117 115 101  45  71 108 105 109 109 101 114  45  51  48  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[6656]` |
| `clip.vision.image_size` | `[896]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1536]` |
| `clip.vision.feed_forward_length` | `[8960]` |
| `clip.vision.block_count` | `[50]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[109 117 115 101  45 103 108 105 109 109 101 114]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

---

## Muse-Glimmer-30B-UD-Q5_K_XL.gguf
**Pfad:** `Meta\Muse-Glimmer-30B-UD-Q5_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[731]` |
| `GGUF.kv_count` | `[36]` |
| `general.architecture` | `[109 117 115 101  45 103 108 105 109 109 101 114]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 77 117 115 101  45  71 108 105 109 109 101 114  45  51  48  66]` |
| `general.size_label` | `[50 56 66]` |
| `muse-glimmer.block_count` | `[52]` |
| `muse-glimmer.context_length` | `[131072]` |
| `muse-glimmer.embedding_length` | `[6656]` |
| `muse-glimmer.feed_forward_length` | `[19968]` |
| `muse-glimmer.attention.head_count` | `[32]` |
| `muse-glimmer.attention.head_count_kv` | `[2]` |
| `muse-glimmer.rope.freq_base` | `[500000.]` |
| `muse-glimmer.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `muse-glimmer.attention.key_length` | `[128]` |
| `muse-glimmer.attention.value_length` | `[128]` |
| `muse-glimmer.final_logit_softcapping` | `[20.]` |
| `muse-glimmer.logit_scale` | `[0.19611613]` |
| `muse-glimmer.attention.sliding_window` | `[2048]` |
| `muse-glimmer.attention.sliding_window_pattern` | `[4]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 108  97 109  97  52]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 40 103  32 101]` |
| `tokenizer.ggml.bos_token_id` | `[200000]` |
| `tokenizer.ggml.eos_token_id` | `[200001]` |
| `tokenizer.ggml.padding_token_id` | `[200018]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[200008]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[17]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[416]` |
| `quantize.imatrix.chunks_count` | `[166]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124 114 101 115 101 114 118 101 100  95 115 112 101  99 105  97 108
  95 116 111 107 101 110  95  50  48  52  55 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 77 117 115 101  45  71 108 105 109 109 101 114  45 105 109  97 116 114
 105 120  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 77 117 115 101  45  71 108 105 109 109 101 114  45  99  97 108 105  98
 114  97 116 105 111 110  46 116 120 116]
```

---

## fastcontext-1.0-4b-rl-q8_0.gguf
**Pfad:** `Microsoft\fastcontext-1.0-4b-rl-q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[398]` |
| `GGUF.kv_count` | `[34]` |
| `general.architecture` | `[113 119 101 110  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.8]` |
| `general.sampling.temp` | `[0.7]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[82 76]` |
| `general.basename` | `[ 70  97 115 116  67 111 110 116 101 120 116  45  49  46  48]` |
| `general.size_label` | `[52 66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | *Siehe Code-Block unten* |
| `general.languages` | `[101 110]` |
| `qwen3.block_count` | `[36]` |
| `qwen3.context_length` | `[262144]` |
| `qwen3.embedding_length` | `[2560]` |
| `qwen3.feed_forward_length` | `[9728]` |
| `qwen3.attention.head_count` | `[32]` |
| `qwen3.attention.head_count_kv` | `[8]` |
| `qwen3.rope.freq_base` | `[5.e+06]` |
| `qwen3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3.attention.key_length` | `[128]` |
| `qwen3.attention.value_length` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151643]` |
| `tokenizer.ggml.bos_token_id` | `[151643]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[ 70  97 115 116  67 111 110 116 101 120 116  32  49  46  48  32  52  66
  32  82  76]
```

**general.tags:**
```jinja
[ 82 101 112 111 115 105 116 111 114 121  32  69 120 112 108 111 114  97
 116 105 111 110]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## FastContext-1.0-4B-SFT-Q8_0.gguf
**Pfad:** `Microsoft\FastContext-1.0-4B-SFT-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[398]` |
| `GGUF.kv_count` | `[39]` |
| `general.architecture` | `[113 119 101 110  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.8]` |
| `general.sampling.temp` | `[0.7]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[83 70 84]` |
| `general.basename` | `[ 70  97 115 116  67 111 110 116 101 120 116  45  49  46  48]` |
| `general.size_label` | `[52 66]` |
| `general.license` | `[109 105 116]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 48 55]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | *Siehe Code-Block unten* |
| `general.languages` | `[101 110]` |
| `qwen3.block_count` | `[36]` |
| `qwen3.context_length` | `[262144]` |
| `qwen3.embedding_length` | `[2560]` |
| `qwen3.feed_forward_length` | `[9728]` |
| `qwen3.attention.head_count` | `[32]` |
| `qwen3.attention.head_count_kv` | `[8]` |
| `qwen3.rope.freq_base` | `[5.e+06]` |
| `qwen3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3.attention.key_length` | `[128]` |
| `qwen3.attention.value_length` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151643]` |
| `tokenizer.ggml.bos_token_id` | `[151643]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[ 70  97 115 116  67 111 110 116 101 120 116  32  49  46  48  32  52  66
  32  83  70  84]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  51  32  52  66  32  73 110 115 116 114 117  99 116  32
  50  53  48  55]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  52  66
  45  73 110 115 116 114 117  99 116  45  50  53  48  55]
```

**general.tags:**
```jinja
[ 82 101 112 111 115 105 116 111 114 121  32  69 120 112 108 111 114  97
 116 105 111 110]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mmproj-Tess-4-27B-F16.gguf
**Pfad:** `migtissera\mmproj-Tess-4-27B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[23]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## mtp-Tess-4-27B-Q4_K_M.gguf
**Pfad:** `migtissera\mtp-Tess-4-27B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[18]` |
| `GGUF.kv_count` | `[43]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[51 46 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Tess-4-27B-Q6_K.gguf
**Pfad:** `migtissera\Tess-4-27B-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[851]` |
| `GGUF.kv_count` | `[35]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[50 55 66]` |
| `qwen35.block_count` | `[64]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Devstral-Small-2-24B-Instruct-2512-UD_Q4_K_XL.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\Devstral-Small-2-24B-Instruct-2512-UD_Q4_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[54]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 53 49 50]` |
| `general.finetune` | `[ 73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[393216]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[32768]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+08]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[48.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[8192]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[280]` |
| `quantize.imatrix.chunks_count` | `[75]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50  45  71
  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116
 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50
  45  50  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50
  46 116 120 116]
```

---

## Devstral-Small-2-24B-Instruct-2512-UD_Q8_K_XL.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\Devstral-Small-2-24B-Instruct-2512-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[54]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 53 49 50]` |
| `general.finetune` | `[ 73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[393216]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[32768]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+08]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[48.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[8192]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[280]` |
| `quantize.imatrix.chunks_count` | `[75]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50  45  71
  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116
 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50
  45  50  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50
  46 116 120 116]
```

---

## mmproj-Devstral-Small-2-24B-Instruct-2512-BF16.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\mmproj-Devstral-Small-2-24B-Instruct-2512-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 45 50 53 49 50]` |
| `general.finetune` | `[ 50  52  98  45  73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 83 109  97 108 108]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 45 50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[32768]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

---

## mmproj-Devstral-Small-2-24B-Instruct-2512-UD_F32.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\mmproj-Devstral-Small-2-24B-Instruct-2512-UD_F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 45 50 53 49 50]` |
| `general.finetune` | `[ 50  52  98  45  73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 83 109  97 108 108]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 45 50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[32768]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

---

## Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf
**Pfad:** `Mistral AI\Ministral 3\Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[36]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[51 46 48]` |
| `general.organization` | `[ 77 105 115 116 114  97 108  32  65  73]` |
| `general.finetune` | `[50 53 49 50]` |
| `general.basename` | `[ 77 105 110 105 115 116 114  97 108]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[49 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.url` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[109 105 115 116 114  97 108  45  99 111 109 109 111 110]` |
| `general.languages` | `[ 97 114]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[16384]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[109 105 110 105 115 116 114  97 108  45  49  52  66  45  73 110 115 116
 114 117  99 116  45  50  53  49  50]
```

**general.description:**
```jinja
[ 84 104 101  32  77 105 110 105 115 116 114  97 108  32  51  32 102  97
 109 105 108 121  32 105 115  32 100 101 115 105 103 110 101 100  32 102
 111 114  32 101 100 103 101  32 100 101 112 108 111 121 109 101 110 116
  44  32  99  97 112  97  98 108 101  32 111 102  32 114 117 110 110 105
 110 103  32 111 110  32  97  32 119 105 100 101  32 114  97 110 103 101
  32 111 102  32 104  97 114 100 119  97 114 101  46  32  84 104 105 115
  32 109 111 100 101 108  32 105 115  32 116 104 101  32  49  52  66  32
 105 110 115 116 114 117  99 116  32 112 111 115 116  45 116 114  97 105
 110 101 100  32 118 101 114 115 105 111 110  32 105 110  32  70  80  56
  44  32 102 105 110 101  45 116 117 110 101 100  32 102 111 114  32 105
 110 115 116 114 117  99 116 105 111 110  32 116  97 115 107 115  44  32
 109  97 107 105 110 103  32 105 116  32 105 100 101  97 108  32 102 111
 114  32  99 104  97 116  32  97 110 100  32 105 110 115 116 114 117  99
 116 105 111 110  32  98  97 115 101 100  32 117 115 101  32  99  97 115
 101 115  46]
```

**general.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.base_model.0.name:**
```jinja
[ 77 105 110 105 115 116 114  97 108  32  51  32  49  52  66  32  66  97
 115 101  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  66  97 115 101  45
  50  53  49  50]
```

---

## Ministral-3-14B-Instruct-2512-Q8_0.gguf
**Pfad:** `Mistral AI\Ministral 3\Ministral-3-14B-Instruct-2512-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[53]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[51 46 48]` |
| `general.organization` | `[ 77 105 115 116 114  97 108  32  65  73]` |
| `general.finetune` | `[50 53 49 50]` |
| `general.basename` | `[ 77 105 110 105 115 116 114  97 108]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[49 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.url` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[109 105 115 116 114  97 108  45  99 111 109 109 111 110]` |
| `general.languages` | `[ 97 114]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[262144]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[16384]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+09]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[16.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[16384]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[109 105 110 105 115 116 114  97 108  45  49  52  66  45  73 110 115 116
 114 117  99 116  45  50  53  49  50]
```

**general.description:**
```jinja
[ 84 104 101  32  77 105 110 105 115 116 114  97 108  32  51  32 102  97
 109 105 108 121  32 105 115  32 100 101 115 105 103 110 101 100  32 102
 111 114  32 101 100 103 101  32 100 101 112 108 111 121 109 101 110 116
  44  32  99  97 112  97  98 108 101  32 111 102  32 114 117 110 110 105
 110 103  32 111 110  32  97  32 119 105 100 101  32 114  97 110 103 101
  32 111 102  32 104  97 114 100 119  97 114 101  46  32  84 104 105 115
  32 109 111 100 101 108  32 105 115  32 116 104 101  32  49  52  66  32
 105 110 115 116 114 117  99 116  32 112 111 115 116  45 116 114  97 105
 110 101 100  32 118 101 114 115 105 111 110  32 105 110  32  70  80  56
  44  32 102 105 110 101  45 116 117 110 101 100  32 102 111 114  32 105
 110 115 116 114 117  99 116 105 111 110  32 116  97 115 107 115  44  32
 109  97 107 105 110 103  32 105 116  32 105 100 101  97 108  32 102 111
 114  32  99 104  97 116  32  97 110 100  32 105 110 115 116 114 117  99
 116 105 111 110  32  98  97 115 101 100  32 117 115 101  32  99  97 115
 101 115  46]
```

**general.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.base_model.0.name:**
```jinja
[ 77 105 110 105 115 116 114  97 108  32  51  32  49  52  66  32  66  97
 115 101  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  66  97 115 101  45
  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf
**Pfad:** `NVIDIA\dflash-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[153]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 76 105  53  95  83  87  65]` |
| `general.size_label` | `[49 46 49 66]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[4096]` |
| `dflash.feed_forward_length` | `[12288]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[2]` |
| `dflash.rope.freq_base` | `[10000.]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `general.file_type` | `[39]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[0]` |
| `dflash.selector_top_k` | `[0]` |
| `dflash.target_layers` | `[86]` |
| `dflash.attention.sliding_window` | `[1024]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `dflash.attention.causal` | `[False]` |
| `lilicorr.hidden_size` | `[1024]` |
| `lilicorr.candidate_topk` | `[8]` |
| `lilicorr.num_heads` | `[8]` |
| `lilicorr.num_layers` | `[2]` |
| `lilicorr.factor_dim` | `[1024]` |
| `lilicorr.mlp_hidden_size` | `[2048]` |
| `lilicorr.vector_eps` | `[0.0001]` |
| `lilicorr.logit_scale` | `[8.]` |
| `general.quantization_version` | `[2]` |
| `general.tensor_extra.name` | *Siehe Code-Block unten* |
| `general.tensor_extra.prec_a4` | `[False]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[112 105 120 116 114  97 108]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[11]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[990]` |

**general.tensor_extra.name:**
```jinja
[108 105 108 105  99 111 114 114  46 116 111 107 101 110  95 112 114 111
 106  46 119 101 105 103 104 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  32 ...  37 125  10]
```

---

## mmproj-NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.gguf
**Pfad:** `NVIDIA\mmproj-NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[390]` |
| `GGUF.kv_count` | `[24]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 51  48  98  45  82 101  97 115 111 110 105 110 103]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.size_label` | `[65 51 66]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2688]` |
| `clip.vision.image_size` | `[512]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1280]` |
| `clip.vision.feed_forward_length` | `[5120]` |
| `clip.vision.block_count` | `[32]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[110 101 109 111 116 114 111 110  95 118  50  95 118 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.basename:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

---

## mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16.gguf
**Pfad:** `NVIDIA\mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[519]` |
| `GGUF.kv_count` | `[25]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.size_label` | `[56 52 50 77]` |
| `clip.projector_type` | `[110 101 109 111 116 114 111 110  95 111 109 110 105]` |
| `general.file_type` | `[39]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[512]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1280]` |
| `clip.vision.feed_forward_length` | `[5120]` |
| `clip.vision.block_count` | `[32]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `clip.vision.nemotron_omni` | `[ True]` |
| `clip.vision.image_min_pixels` | `[262144]` |
| `clip.vision.image_max_pixels` | `[3407872]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 72 102  32  80  97  99 107 101 100  32  54  50  50  50  32  83 104  97
 114 101 100 116 112  49]
```

---

## NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD_Q6_K.gguf
**Pfad:** `NVIDIA\NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[401]` |
| `GGUF.kv_count` | `[56]` |
| `general.architecture` | `[110 101 109 111 116 114 111 110  95 104  95 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 82 101  97 115 111 110 105 110 103]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 48 66 45 65 51 66]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `nemotron_h_moe.block_count` | `[52]` |
| `nemotron_h_moe.context_length` | `[1048576]` |
| `nemotron_h_moe.embedding_length` | `[2688]` |
| `nemotron_h_moe.feed_forward_length` | `[1856]` |
| `nemotron_h_moe.attention.head_count` | `[32]` |
| `nemotron_h_moe.attention.head_count_kv` | `[0]` |
| `nemotron_h_moe.rope.freq_base` | `[10000.]` |
| `nemotron_h_moe.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.expert_used_count` | `[6]` |
| `nemotron_h_moe.expert_group_count` | `[1]` |
| `nemotron_h_moe.expert_group_used_count` | `[1]` |
| `nemotron_h_moe.vocab_size` | `[131072]` |
| `nemotron_h_moe.rope.dimension_count` | `[84]` |
| `nemotron_h_moe.ssm.conv_kernel` | `[4]` |
| `nemotron_h_moe.ssm.state_size` | `[128]` |
| `nemotron_h_moe.ssm.group_count` | `[8]` |
| `nemotron_h_moe.ssm.inner_size` | `[4096]` |
| `nemotron_h_moe.ssm.time_step_rank` | `[64]` |
| `nemotron_h_moe.rope.scaling.finetuned` | `[False]` |
| `nemotron_h_moe.attention.key_length` | `[128]` |
| `nemotron_h_moe.attention.value_length` | `[128]` |
| `nemotron_h_moe.expert_feed_forward_length` | `[1856]` |
| `nemotron_h_moe.expert_shared_feed_forward_length` | `[3712]` |
| `nemotron_h_moe.expert_count` | `[128]` |
| `nemotron_h_moe.expert_shared_count` | `[1]` |
| `nemotron_h_moe.expert_weights_norm` | `[ True]` |
| `nemotron_h_moe.expert_weights_scale` | `[2.5]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[112 105 120 116 114  97 108]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[11]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[999]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[185]` |
| `quantize.imatrix.chunks_count` | `[72]` |

**general.name:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.basename:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**tokenizer.chat_template:**
```jinja
[123  37  32 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103  45  71  71  85  70  47 105 109  97
 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45
  51  45  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51
  66  45  82 101  97 115 111 110 105 110 103  46 116 120 116]
```

---

## NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf
**Pfad:** `NVIDIA\NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-Q3_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[763]` |
| `GGUF.kv_count` | `[72]` |
| `general.architecture` | `[110 101 109 111 116 114 111 110  95 104  95 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.size_label` | `[ 51  52  53 120  56  46  49  66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.license.name` | `[111 112 101 110 109 100 119  45  49  46  49]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 78 118 105 100 105  97]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[2]` |
| `general.dataset.0.name` | *Siehe Code-Block unten* |
| `general.dataset.0.version` | `[118  51]` |
| `general.dataset.0.organization` | `[ 78 118 105 100 105  97]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.1.name` | *Siehe Code-Block unten* |
| `general.dataset.1.organization` | `[ 78 118 105 100 105  97]` |
| `general.dataset.1.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[106  97]` |
| `nemotron_h_moe.block_count` | `[88]` |
| `nemotron_h_moe.context_length` | `[262144]` |
| `nemotron_h_moe.embedding_length` | `[4096]` |
| `nemotron_h_moe.feed_forward_length` | `[2560]` |
| `nemotron_h_moe.attention.head_count` | `[32]` |
| `nemotron_h_moe.attention.head_count_kv` | `[0]` |
| `nemotron_h_moe.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.expert_count` | `[496]` |
| `nemotron_h_moe.expert_group_count` | `[1]` |
| `nemotron_h_moe.expert_group_used_count` | `[1]` |
| `nemotron_h_moe.expert_gating_func` | `[2]` |
| `nemotron_h_moe.vocab_size` | `[131072]` |
| `nemotron_h_moe.rope.dimension_count` | `[128]` |
| `nemotron_h_moe.ssm.conv_kernel` | `[4]` |
| `nemotron_h_moe.ssm.state_size` | `[128]` |
| `nemotron_h_moe.ssm.group_count` | `[8]` |
| `nemotron_h_moe.ssm.inner_size` | `[8192]` |
| `nemotron_h_moe.ssm.time_step_rank` | `[128]` |
| `nemotron_h_moe.rope.scaling.finetuned` | `[False]` |
| `nemotron_h_moe.attention.key_length` | `[128]` |
| `nemotron_h_moe.attention.value_length` | `[128]` |
| `nemotron_h_moe.expert_feed_forward_length` | `[2560]` |
| `nemotron_h_moe.expert_used_count` | `[9]` |
| `nemotron_h_moe.expert_shared_feed_forward_length` | `[5376]` |
| `nemotron_h_moe.expert_shared_count` | `[1]` |
| `nemotron_h_moe.expert_weights_norm` | `[ True]` |
| `nemotron_h_moe.expert_weights_scale` | `[5.]` |
| `nemotron_h_moe.moe_latent_size` | `[1024]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[11]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[384]` |
| `quantize.imatrix.chunks_count` | `[625]` |

**general.name:**
```jinja
[ 78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45  51  46
  53  45  66 111 108 116  45  55  52  66  45  65  57  66]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 111 112 101 110 109 100 119  46  97 105
  47 108 105  99 101 110 115 101  47  49  45  49  47]
```

**general.base_model.0.name:**
```jinja
[ 78  86  73  68  73  65  32  78 101 109 111 116 114 111 110  32  51  46
  53  32  66 111 108 116  32  55  52  66  32  65  57  66  32  66  70  49
  54]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 110 118 105 100 105  97  47  78  86  73  68  73  65
  45  78 101 109 111 116 114 111 110  45  51  46  53  45  66 111 108 116
  45  55  52  66  45  65  57  66  45  66  70  49  54]
```

**general.dataset.0.name:**
```jinja
[ 78 101 109 111 116 114 111 110  32  80 111 115 116  32  84 114  97 105
 110 105 110 103  32 118  51]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 110 118 105 100 105  97  47 110 101 109 111 116 114
 111 110  45 112 111 115 116  45 116 114  97 105 110 105 110 103  45 118
  51]
```

**general.dataset.1.name:**
```jinja
[ 78 101 109 111 116 114 111 110  32  80 114 101  32  84 114  97 105 110
 105 110 103  32  68  97 116  97 115 101 116 115]
```

**general.dataset.1.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 110 118 105 100 105  97  47 110 101 109 111 116 114
 111 110  45 112 114 101  45 116 114  97 105 110 105 110 103  45 100  97
 116  97 115 101 116 115]
```

**tokenizer.chat_template:**
```jinja
[123  37  32 ...  37 125  10]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**quantize.imatrix.file:**
```jinja
[ 78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45  51  46
  53  45  66 111 108 116  45  55  52  66  45  65  57  66  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45
  51  46  53  45  66 111 108 116  45  55  52  66  45  65  57  66  46 116
 120 116]
```

---

## gpt-oss-20b-UD_Q8_K_XL.gguf
**Pfad:** `OpenAI\gpt-oss-20b-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[459]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[103 112 116  45 111 115 115]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 71 112 116  45  79 115 115  45  50  48  66]` |
| `general.basename` | `[ 71 112 116  45  79 115 115  45  50  48  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `gpt-oss.block_count` | `[24]` |
| `gpt-oss.context_length` | `[131072]` |
| `gpt-oss.embedding_length` | `[2880]` |
| `gpt-oss.feed_forward_length` | `[2880]` |
| `gpt-oss.attention.head_count` | `[64]` |
| `gpt-oss.attention.head_count_kv` | `[8]` |
| `gpt-oss.rope.freq_base` | `[150000.]` |
| `gpt-oss.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `gpt-oss.expert_count` | `[32]` |
| `gpt-oss.expert_used_count` | `[4]` |
| `gpt-oss.attention.key_length` | `[64]` |
| `gpt-oss.attention.value_length` | `[64]` |
| `gpt-oss.attention.sliding_window` | `[128]` |
| `gpt-oss.expert_feed_forward_length` | `[2880]` |
| `gpt-oss.rope.scaling.type` | `[121  97 114 110]` |
| `gpt-oss.rope.scaling.factor` | `[32.]` |
| `gpt-oss.rope.scaling.original_context_length` | `[4096]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 112 116  45  52 111]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 48 49 48 56 55 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  99 111  99 111  32 115]` |
| `tokenizer.ggml.bos_token_id` | `[199998]` |
| `tokenizer.ggml.eos_token_id` | `[200002]` |
| `tokenizer.ggml.padding_token_id` | `[200017]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  35 125]
```

---

## dspark-MiniCPM-V-4_7-35B-A3B-F16.gguf
**Pfad:** `openbmb\dspark-MiniCPM-V-4_7-35B-A3B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[62]` |
| `GGUF.kv_count` | `[39]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 51  53  98  45  68  83 112  97 114 107]` |
| `general.basename` | `[ 77 105 110 105  67  80  77  45  86  45  52  46  55]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[2048]` |
| `dflash.feed_forward_length` | `[6144]` |
| `dflash.attention.head_count` | `[8]` |
| `dflash.attention.head_count_kv` | `[2]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[256]` |
| `dflash.attention.value_length` | `[256]` |
| `general.file_type` | `[32]` |
| `dflash.block_size` | `[15]` |
| `dflash.target_layers` | `[36]` |
| `dflash.rope.dimension_sections` | `[0]` |
| `dflash.sample_from_anchor` | `[ True]` |
| `dflash.has_confidence_head` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[ 60 124 116 117 114 110  95 101 111 115 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.bos_token_id` | `[248045]` |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.unknown_token_id` | `[248077]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[248076]` |

**general.name:**
```jinja
[ 77 105 110 105  67  80  77  32  86  32  52  46  55  32  51  53  66  32
  65  51  66  32  68  83 112  97 114 107]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## MiniCPM-V-4_7-35B-A3B-Q6_K.gguf
**Pfad:** `openbmb\MiniCPM-V-4_7-35B-A3B-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 51  53  97  51  95 104 102  95 118  52  55  95  48  57  50  51]` |
| `general.finetune` | `[ 51  53  97  51  95 104 102  95 118  52  55  95  48  57  50  51]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.attention.recurrent_layers` | `[False]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.rope.section_order` | `[122 121 120 116]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[ 60 124 116 117 114 110  95 101 111 115 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.bos_token_id` | `[248045]` |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.unknown_token_id` | `[248077]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## MiniCPM5-2B-F16.gguf
**Pfad:** `openbmb\MiniCPM5-2B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[381]` |
| `GGUF.kv_count` | `[36]` |
| `general.architecture` | `[108 108  97 109  97]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 77 105 110 105  67  80  77  53  32  50  46  54  66]` |
| `general.organization` | `[ 79 112 101 110  98 109  98]` |
| `general.finetune` | *Siehe Code-Block unten* |
| `general.basename` | `[ 77 105 110 105  67  80  77  53]` |
| `general.size_label` | `[50 46 54 66]` |
| `llama.block_count` | `[42]` |
| `llama.context_length` | `[131072]` |
| `llama.embedding_length` | `[2048]` |
| `llama.feed_forward_length` | `[6144]` |
| `llama.attention.head_count` | `[16]` |
| `llama.attention.head_count_kv` | `[2]` |
| `llama.rope.freq_base` | `[5.e+06]` |
| `llama.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `llama.attention.key_length` | `[128]` |
| `llama.attention.value_length` | `[128]` |
| `general.file_type` | `[1]` |
| `llama.vocab_size` | `[130560]` |
| `llama.rope.dimension_count` | `[128]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[109 105 110 105  99 112 109  53]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  95 116 111 107 101 110  95  52  55  55  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 97 115 115  32 105 115 116  97 110 116]` |
| `tokenizer.ggml.bos_token_id` | `[0]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[130074]` |
| `tokenizer.ggml.padding_token_id` | `[1]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.finetune:**
```jinja
[ 48  56  50  50  95 106 111  98  95  55  53  56  55  57  56  46 105 116
 101 114  95  48  48  48  49  52  52  46 111 112 100  46  67  50  54  95
 109 111 112 100  95 118  52  95 102 118  95 112 103  95 114 107 108  95
 108 114  49 101  54  46  49  54 116  99 104  46 109 103  50 104 102]
```

**tokenizer.chat_template:**
```jinja
[123 123  45 ...  37 125  10]
```

---

## mmproj-MiniCPM-V-4.7-35B-A3B-f16.gguf
**Pfad:** `openbmb\mmproj-MiniCPM-V-4.7-35B-A3B-f16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[460]` |
| `GGUF.kv_count` | `[26]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 51  53  97  51  95 104 102  95 118  52  55  95  48  57  50  51]` |
| `general.finetune` | `[ 51  53  97  51  95 104 102  95 118  52  55  95  48  57  50  51]` |
| `general.size_label` | `[53 53 51 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[448]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[109 105 110 105  99 112 109 118  52  95  55]` |
| `clip.vision.projector.scale_factor` | `[4]` |
| `clip.vision.max_slice_nums` | `[9]` |
| `clip.vision.wa_layer_indexes` | `[6]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

---

## VoxCPM2-BaseLM-F16.gguf
**Pfad:** `openbmb\VoxCPM2-BaseLM-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[257]` |
| `GGUF.kv_count` | `[31]` |
| `general.architecture` | `[109 105 110 105  99 112 109  52]` |
| `minicpm4.context_length` | `[32768]` |
| `minicpm4.embedding_length` | `[2048]` |
| `minicpm4.block_count` | `[28]` |
| `minicpm4.feed_forward_length` | `[6144]` |
| `minicpm4.vocab_size` | `[73448]` |
| `minicpm4.attention.head_count` | `[16]` |
| `minicpm4.attention.head_count_kv` | `[2]` |
| `minicpm4.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `minicpm4.attention.key_length` | `[128]` |
| `minicpm4.attention.value_length` | `[128]` |
| `minicpm4.rope.dimension_count` | `[128]` |
| `minicpm4.rope.freq_base` | `[10000.]` |
| `minicpm4.rope.scaling.finetuned` | `[ True]` |
| `minicpm4.embedding_scale` | `[0.]` |
| `minicpm4.residual_scale` | `[0.]` |
| `minicpm4.logit_scale` | `[0.125]` |
| `minicpm4.rope.scaling.type` | `[108 111 110 103 114 111 112 101]` |
| `minicpm4.rope.scaling.original_context_length` | `[32768]` |
| `tokenizer.ggml.model` | `[108 108  97 109  97]` |
| `tokenizer.ggml.pre` | `[100 101 102  97 117 108 116]` |
| `tokenizer.ggml.tokens` | `[ 60 124 102 105 109  95 115 117 102 102 105 120 124  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[226 150 129  98 117 100  32 115]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[73440]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**tokenizer.chat_template:**
```jinja
[123  37  32 102 111 114  32 109 101 115 115  97 103 101  32 105 110  32
 109 101 115 115  97 103 101 115  32  37 125 123 123  39  60 124 105 109
  95 115 116  97 114 116 124  62  39  32  43  32 109 101 115 115  97 103
 101  91  39 114 111 108 101  39  93  32  43  32  39  10  39  32  43  32
 109 101 115 115  97 103 101  91  39  99 111 110 116 101 110 116  39  93
  32  43  32  39  60 124 105 109  95 101 110 100 124  62  39  32  43  32
  39  10  39 125 125 123  37  32 101 110 100 102 111 114  32  37 125 123
  37  32 105 102  32  97 100 100  95 103 101 110 101 114  97 116 105 111
 110  95 112 114 111 109 112 116  32  37 125 123 123  32  39  60 124 105
 109  95 115 116  97 114 116 124  62  97 115 115 105 115 116  97 110 116
  10  39  32 125 125 123  37  32 101 110 100 105 102  32  37 125]
```

---

## mmproj-Ornith-1.5-35B-BF16.gguf
**Pfad:** `ornith-ai\mmproj-Ornith-1.5-35B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[26]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[51 53 98 32 52 48 48 48]` |
| `general.version` | `[52 48 48 48]` |
| `general.finetune` | `[51 53 98]` |
| `general.size_label` | `[52 52 55 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## mmproj-Ornith-1.5-9B-BF16.gguf
**Pfad:** `ornith-ai\mmproj-Ornith-1.5-9B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[22]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 57  98  32  67 107 112 116  49  48  48  48]` |
| `general.finetune` | `[ 57  98  45  99 107 112 116  49  48  48  48]` |
| `general.size_label` | `[52 53 54 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## ornith-1.0-35b-Q6_K.gguf
**Pfad:** `ornith-ai\ornith-1.0-35b-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  32  49  46  48  32  51  53  66]` |
| `general.basename` | `[ 79 114 110 105 116 104  45  49  46  48]` |
| `general.size_label` | `[51 53 66]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[1608]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

**quantize.imatrix.file:**
```jinja
[111 114 110 105 116 104  95  51  53  98  47 105 109  97 116 114 105 120
  46 100  97 116]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 115 101  47 103 103 117
 102  47  99  97 108 105  98 114  97 116 105 111 110  95  51  53  98  95
  56  48  48 107  46 116 120 116]
```

---

## Ornith-1.5-35B-Q4_K_M.gguf
**Pfad:** `ornith-ai\Ornith-1.5-35B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[47]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  51  53  66]` |
| `general.version` | `[52 48 48 48]` |
| `general.finetune` | `[51 53 98]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[3636]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 103 103 117 102  95 119
 111 114 107  47 111 117 116  47  51  53  98  45  52  48  48  48  47  79
 114 110 105 116 104  45  49  46  53  45  51  53  66  46 105 109  97 116
 114 105 120  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 103 103 117 102  95 119
 111 114 107  47  99  97 108 105  98  47 109 101 114 103 101 100  47 115
 104  97 114 100 115  45  51  53  98  45  52  48  48  48  47 115  95  48
  48]
```

---

## Ornith-1.5-9B-Q8_0.gguf
**Pfad:** `ornith-ai\Ornith-1.5-9B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[427]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  57  66]` |
| `general.finetune` | `[ 57  98  45  99 107 112 116  49  48  48  48]` |
| `general.size_label` | `[57 46 48 66]` |
| `qwen35.block_count` | `[32]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[4096]` |
| `qwen35.feed_forward_length` | `[12288]` |
| `qwen35.attention.head_count` | `[16]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[32]` |
| `qwen35.ssm.inner_size` | `[4096]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[72]` |
| `general.architecture` | `[108  97 103 117 110  97]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[1.]` |
| `general.sampling.min_p` | `[0.]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 76  97 103 117 110  97  45  83  45  50  46  49]` |
| `general.version` | `[50 46 49]` |
| `general.basename` | `[ 76  97 103 117 110  97  45  83  45  50  46  49]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 50  53  54 120  52  46  53  66]` |
| `general.license` | `[111 112 101 110 109 100 119  45  49  46  49]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 76  97 103 117 110  97  32  83  32  50  46  49]` |
| `general.base_model.0.version` | `[50 46 49]` |
| `general.base_model.0.organization` | `[ 80 111 111 108 115 105 100 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `laguna.block_count` | `[48]` |
| `laguna.context_length` | `[1048576]` |
| `laguna.embedding_length` | `[3072]` |
| `laguna.feed_forward_length` | `[12288]` |
| `laguna.attention.head_count` | `[72]` |
| `laguna.attention.head_count_kv` | `[8]` |
| `laguna.rope.scaling.type` | `[121  97 114 110]` |
| `laguna.rope.scaling.factor` | `[128.]` |
| `laguna.rope.scaling.original_context_length` | `[8192]` |
| `laguna.rope.scaling.yarn_attn_factor` | `[1.485203]` |
| `laguna.rope.scaling.yarn_beta_fast` | `[32.]` |
| `laguna.rope.scaling.yarn_beta_slow` | `[1.]` |
| `laguna.rope.freq_base` | `[500000.]` |
| `laguna.rope.freq_base_swa` | `[10000.]` |
| `laguna.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `laguna.expert_count` | `[256]` |
| `laguna.expert_used_count` | `[10]` |
| `laguna.attention.key_length` | `[128]` |
| `laguna.attention.value_length` | `[128]` |
| `laguna.vocab_size` | `[100352]` |
| `laguna.attention.sliding_window` | `[512]` |
| `laguna.expert_feed_forward_length` | `[1024]` |
| `laguna.expert_shared_feed_forward_length` | `[1024]` |
| `laguna.expert_weights_norm` | `[ True]` |
| `laguna.expert_weights_scale` | `[2.5]` |
| `laguna.expert_gating_func` | `[2]` |
| `laguna.leading_dense_block_count` | `[1]` |
| `laguna.rope.dimension_count` | `[64]` |
| `laguna.rope.dimension_count_swa` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108  97 103 117 110  97]` |
| `tokenizer.ggml.tokens` | `[ 73 115  78 111 116  78 117 108 108]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[ 73 115  32  78 111 116  78 117 108 108]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.seperator_token_id` | `[8]` |
| `tokenizer.ggml.padding_token_id` | `[9]` |
| `tokenizer.ggml.mask_token_id` | `[12]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[ True]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[24]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[572]` |
| `quantize.imatrix.chunks_count` | `[46]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 112 111 111 108 115 105 100 101  47  76  97 103 117
 110  97  45  83  45  50  46  49]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 76  97 103 117 110  97  45  83  45  50  46  49  45  71  71  85  70  47
 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103
 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  76  97 103 117 110  97  45  83  45  50  46  49  46 116 120 116]
```

---

## Laguna-S-2.1-UD-Q3_K_XL-00002-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[761]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

---

## Laguna-S-2.1-UD-Q3_K_XL-00003-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[53]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

---

## Ternary-Bonsai-2-27B-mmproj-BF16.gguf
**Pfad:** `prism-ml\Ternary-Bonsai-2-27B-mmproj-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[26]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## Ternary-Bonsai-2-27B-PQ2_0.gguf
**Pfad:** `prism-ml\Ternary-Bonsai-2-27B-PQ2_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| Fehler | Konnte Metadaten nicht lesen: np.uint32(142) is not a valid GGMLQuantizationType |

---

## Ternary-Bonsai-2-27B-PTQ1_0.gguf
**Pfad:** `prism-ml\Ternary-Bonsai-2-27B-PTQ1_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| Fehler | Konnte Metadaten nicht lesen: np.uint32(143) is not a valid GGMLQuantizationType |

---

## Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf
**Pfad:** `prism-ml\Ternary-Bonsai-2.5-27B-mmproj-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[26]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf
**Pfad:** `prism-ml\Ternary-Bonsai-2.5-27B-Q2_0-mtp.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| Fehler | Konnte Metadaten nicht lesen: np.uint32(42) is not a valid GGMLQuantizationType |

---

## mmproj-SenseNova-Loop-9B-F16.gguf
**Pfad:** `sensenova\mmproj-SenseNova-Loop-9B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[30]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 83 101 110 115 101  78 111 118  97  32  76 111 111 112  32  57  66]` |
| `general.finetune` | `[57 98]` |
| `general.basename` | `[ 83 101 110 115 101  78 111 118  97  45  76 111 111 112]` |
| `general.size_label` | `[52 53 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `general.file_type` | `[1]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `clip.vision.image_min_pixels` | `[65536]` |
| `clip.vision.image_max_pixels` | `[16777216]` |
| `general.quantization_version` | `[2]` |

---

## SenseNova-Loop-9B-BF16.gguf
**Pfad:** `sensenova\SenseNova-Loop-9B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[490]` |
| `GGUF.kv_count` | `[56]` |
| `general.architecture` | `[115 101 110 115 101 110 111 118  97  95 108 111 111 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 83 101 110 115 101  78 111 118  97  32  76 111 111 112  32  57  66]` |
| `general.basename` | `[ 83 101 110 115 101  78 111 118  97  45  76 111 111 112]` |
| `general.size_label` | `[57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `sensenova_loop.block_count` | `[57]` |
| `sensenova_loop.context_length` | `[524288]` |
| `sensenova_loop.embedding_length` | `[4096]` |
| `sensenova_loop.feed_forward_length` | `[12288]` |
| `sensenova_loop.attention.head_count` | `[16]` |
| `sensenova_loop.attention.head_count_kv` | `[4]` |
| `sensenova_loop.rope.scaling.type` | `[121  97 114 110]` |
| `sensenova_loop.rope.scaling.factor` | `[2.]` |
| `sensenova_loop.rope.scaling.original_context_length` | `[262144]` |
| `sensenova_loop.rope.dimension_sections` | `[0]` |
| `sensenova_loop.rope.freq_base` | `[1.e+07]` |
| `sensenova_loop.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `sensenova_loop.attention.key_length` | `[256]` |
| `sensenova_loop.attention.value_length` | `[256]` |
| `general.file_type` | `[32]` |
| `sensenova_loop.nextn_predict_layers` | `[1]` |
| `sensenova_loop.ssm.conv_kernel` | `[4]` |
| `sensenova_loop.ssm.state_size` | `[128]` |
| `sensenova_loop.ssm.group_count` | `[16]` |
| `sensenova_loop.ssm.time_step_rank` | `[32]` |
| `sensenova_loop.ssm.inner_size` | `[4096]` |
| `sensenova_loop.attention.recurrent_layers` | `[False]` |
| `sensenova_loop.full_attention_interval` | `[4]` |
| `sensenova_loop.rope.dimension_count` | `[64]` |
| `sensenova_loop.loop.start` | `[4]` |
| `sensenova_loop.loop.end` | `[28]` |
| `sensenova_loop.loop.count` | `[2]` |
| `sensenova_loop.mhc.start` | `[4]` |
| `sensenova_loop.mhc.end` | `[28]` |
| `sensenova_loop.mhc.streams` | `[2]` |
| `sensenova_loop.mhc.iterations` | `[20]` |
| `sensenova_loop.mhc.fixed_identity` | `[False]` |
| `sensenova_loop.mhc.pre_scale` | `[1.]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `tokenizer.ggml.clean_spaces` | `[False]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## SenseNova-Loop-9B-Q8_0.gguf
**Pfad:** `sensenova\SenseNova-Loop-9B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[490]` |
| `GGUF.kv_count` | `[56]` |
| `general.architecture` | `[115 101 110 115 101 110 111 118  97  95 108 111 111 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 83 101 110 115 101  78 111 118  97  32  76 111 111 112  32  57  66]` |
| `general.basename` | `[ 83 101 110 115 101  78 111 118  97  45  76 111 111 112]` |
| `general.size_label` | `[57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[122 104]` |
| `sensenova_loop.block_count` | `[57]` |
| `sensenova_loop.context_length` | `[524288]` |
| `sensenova_loop.embedding_length` | `[4096]` |
| `sensenova_loop.feed_forward_length` | `[12288]` |
| `sensenova_loop.attention.head_count` | `[16]` |
| `sensenova_loop.attention.head_count_kv` | `[4]` |
| `sensenova_loop.rope.scaling.type` | `[121  97 114 110]` |
| `sensenova_loop.rope.scaling.factor` | `[2.]` |
| `sensenova_loop.rope.scaling.original_context_length` | `[262144]` |
| `sensenova_loop.rope.dimension_sections` | `[0]` |
| `sensenova_loop.rope.freq_base` | `[1.e+07]` |
| `sensenova_loop.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `sensenova_loop.attention.key_length` | `[256]` |
| `sensenova_loop.attention.value_length` | `[256]` |
| `sensenova_loop.nextn_predict_layers` | `[1]` |
| `sensenova_loop.ssm.conv_kernel` | `[4]` |
| `sensenova_loop.ssm.state_size` | `[128]` |
| `sensenova_loop.ssm.group_count` | `[16]` |
| `sensenova_loop.ssm.time_step_rank` | `[32]` |
| `sensenova_loop.ssm.inner_size` | `[4096]` |
| `sensenova_loop.attention.recurrent_layers` | `[False]` |
| `sensenova_loop.full_attention_interval` | `[4]` |
| `sensenova_loop.rope.dimension_count` | `[64]` |
| `sensenova_loop.loop.start` | `[4]` |
| `sensenova_loop.loop.end` | `[28]` |
| `sensenova_loop.loop.count` | `[2]` |
| `sensenova_loop.mhc.start` | `[4]` |
| `sensenova_loop.mhc.end` | `[28]` |
| `sensenova_loop.mhc.streams` | `[2]` |
| `sensenova_loop.mhc.iterations` | `[20]` |
| `sensenova_loop.mhc.fixed_identity` | `[False]` |
| `sensenova_loop.mhc.pre_scale` | `[1.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.normalizer.nfc` | `[ True]` |
| `tokenizer.ggml.clean_spaces` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## dspark-Inkling-2-Mini-Q8_0.gguf
**Pfad:** `thinkingmachines\dspark-Inkling-2-Mini-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[58]` |
| `GGUF.kv_count` | `[39]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 83 114  99  95  66  51  53  55  52  48]` |
| `general.size_label` | `[51 54 56 77]` |
| `dflash.block_count` | `[3]` |
| `dflash.context_length` | `[1048576]` |
| `dflash.embedding_length` | `[2304]` |
| `dflash.feed_forward_length` | `[9216]` |
| `dflash.attention.head_count` | `[24]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.rope.dimension_count` | `[128]` |
| `dflash.rope.freq_base` | `[1.e+06]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.sliding_window` | `[1025]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `dflash.final_logit_softcapping` | `[30.]` |
| `dflash.block_size` | `[8]` |
| `dflash.target_layers` | `[48]` |
| `dflash.sample_from_anchor` | `[False]` |
| `dflash.has_confidence_head` | `[False]` |
| `dflash.has_markov_context` | `[ True]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[105 110 107 108 105 110 103  50]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  83  67  72  32  79]` |
| `tokenizer.ggml.eos_token_id` | `[200006]` |
| `tokenizer.ggml.padding_token_id` | `[199999]` |
| `tokenizer.ggml.eot_token_id` | `[200006]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.clean_spaces` | `[False]` |
| `tokenizer.ggml.mask_token_id` | `[0]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124  99 111 110 116 101 110 116  95 105 110 118 111 107 101  95 116
 111 111 108  95 116 101 120 116 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## Inkling-2-Mini-UD-Q8_K_XL.gguf
**Pfad:** `thinkingmachines\Inkling-2-Mini-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[744]` |
| `GGUF.kv_count` | `[59]` |
| `general.architecture` | `[105 110 107 108 105 110 103  50  95 108 111 111 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 73 110 107 108 105 110 103  45  50  45  77 105 110 105]` |
| `general.size_label` | `[ 49  50  56 120  49  46  56  66]` |
| `inkling2_loop.block_count` | `[24]` |
| `inkling2_loop.context_length` | `[1048576]` |
| `inkling2_loop.embedding_length` | `[2304]` |
| `inkling2_loop.feed_forward_length` | `[1152]` |
| `inkling2_loop.attention.head_count` | `[24]` |
| `inkling2_loop.attention.head_count_kv` | `[8]` |
| `inkling2_loop.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `inkling2_loop.expert_count` | `[128]` |
| `inkling2_loop.expert_used_count` | `[8]` |
| `inkling2_loop.attention.key_length` | `[128]` |
| `inkling2_loop.attention.value_length` | `[128]` |
| `general.file_type` | `[32]` |
| `inkling2_loop.loop.count` | `[3]` |
| `inkling2_loop.loop.start` | `[6]` |
| `inkling2_loop.loop.end` | `[18]` |
| `inkling2_loop.loop.gate_rank` | `[64]` |
| `inkling2_loop.vocab_size` | `[201024]` |
| `inkling2_loop.vocab_size_unpadded` | `[200058]` |
| `inkling2_loop.expert_feed_forward_length` | `[1152]` |
| `inkling2_loop.attention.key_length_swa` | `[128]` |
| `inkling2_loop.attention.value_length_swa` | `[128]` |
| `inkling2_loop.rope.dimension_count` | `[0]` |
| `inkling2_loop.attention.sliding_window` | `[1024]` |
| `inkling2_loop.attention.sliding_window_pattern` | `[False]` |
| `inkling2_loop.shortconv.l_cache` | `[4]` |
| `inkling2_loop.expert_weights_scale` | `[8.]` |
| `inkling2_loop.activation.situ_beta` | `[4.]` |
| `inkling2_loop.activation.situ_linear_beta` | `[25.]` |
| `inkling2_loop.attn_res.block_size` | `[6]` |
| `inkling2_loop.final_logit_softcapping` | `[30.]` |
| `inkling2_loop.attention.relative_length` | `[16]` |
| `inkling2_loop.attention.relative_extent` | `[1024]` |
| `inkling2_loop.attention.qk_clip` | `[22.627417]` |
| `inkling2_loop.attention.relative_clip` | `[8.]` |
| `inkling2_loop.attention.log_scale_floor` | `[8192]` |
| `inkling2_loop.attention.log_scale_alpha` | `[0.1]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[105 110 107 108 105 110 103  50]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  83  67  72  32  79]` |
| `tokenizer.ggml.eos_token_id` | `[200006]` |
| `tokenizer.ggml.eot_token_id` | `[200006]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.ggml.clean_spaces` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.padding_token_id` | `[199999]` |
| `general.sampling.temp` | `[1.]` |
| `general.sampling.top_k` | `[0]` |
| `general.sampling.top_p` | `[1.]` |
| `general.sampling.min_p` | `[0.]` |
| `general.basename` | `[ 73 110 107 108 105 110 103  45  50  45  77 105 110 105]` |
| `inkling2_loop.loop.varying` | `[False]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124  99 111 110 116 101 110 116  95 105 110 118 111 107 101  95 116
 111 111 108  95 116 101 120 116 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## mmproj-Inkling-2-Mini-F32.gguf
**Pfad:** `thinkingmachines\mmproj-Inkling-2-Mini-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[422]` |
| `GGUF.kv_count` | `[19]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 73 110 107 108 105 110 103  45  50  45  77 105 110 105]` |
| `general.size_label` | `[51 49 56 77]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2304]` |
| `clip.vision.image_size` | `[640]` |
| `clip.vision.patch_size` | `[20]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[23]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.vision.projector_type` | `[105 110 107 108 105 110 103  50 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

---

## Spark-X2.5-4B.gguf
**Pfad:** `XHTokens\Spark-X2.5-4B.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[290]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[115 112  97 114 107  50  95  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[-1]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102  95  70 111 114 109  97 116]` |
| `general.size_label` | `[52 46 49 66]` |
| `spark2_5.block_count` | `[36]` |
| `spark2_5.context_length` | `[1048576]` |
| `spark2_5.embedding_length` | `[2560]` |
| `spark2_5.feed_forward_length` | `[10240]` |
| `spark2_5.attention.head_count` | `[16]` |
| `spark2_5.attention.head_count_kv` | `[4]` |
| `spark2_5.rope.freq_base` | `[5.e+06]` |
| `spark2_5.rope.freq_base_swa` | `[10000.]` |
| `spark2_5.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `spark2_5.attention.key_length` | `[256]` |
| `spark2_5.attention.value_length` | `[256]` |
| `general.file_type` | `[32]` |
| `spark2_5.vocab_size` | `[131072]` |
| `spark2_5.attention.sliding_window` | `[512]` |
| `spark2_5.attention.sliding_window_pattern` | `[False]` |
| `spark2_5.rope.dimension_count` | `[64]` |
| `spark2_5.rope.dimension_count_swa` | `[256]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[115 112  97 114 107  50  95  53]` |
| `tokenizer.ggml.tokens` | `[ 60 124 124 117 110 117 115 101 100  56  55 124 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160 115 104  32 105 110 101 115]` |
| `tokenizer.ggml.bos_token_id` | `[0]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[5]` |
| `tokenizer.ggml.padding_token_id` | `[2]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## xing4_0-29b-IQ4_NL.gguf
**Pfad:** `XingChen-AGI\xing4_0-29b-IQ4_NL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[977]` |
| `GGUF.kv_count` | `[60]` |
| `general.architecture` | `[120 105 110 103  52  95  48]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.sampling.penalty_repeat` | `[1.05]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | *Siehe Code-Block unten* |
| `general.size_label` | `[ 54  52 120  52  46  50  66]` |
| `xing4_0.block_count` | `[41]` |
| `xing4_0.context_length` | `[262144]` |
| `xing4_0.embedding_length` | `[3584]` |
| `xing4_0.feed_forward_length` | `[9216]` |
| `xing4_0.attention.head_count` | `[32]` |
| `xing4_0.attention.head_count_kv` | `[1]` |
| `xing4_0.rope.scaling.type` | `[121  97 114 110]` |
| `xing4_0.rope.scaling.factor` | `[64.]` |
| `xing4_0.rope.scaling.original_context_length` | `[4096]` |
| `xing4_0.rope.scaling.yarn_beta_fast` | `[32.]` |
| `xing4_0.rope.scaling.yarn_beta_slow` | `[1.]` |
| `xing4_0.rope.freq_base` | `[10000.]` |
| `xing4_0.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `xing4_0.expert_count` | `[64]` |
| `xing4_0.expert_used_count` | `[4]` |
| `xing4_0.expert_group_count` | `[1]` |
| `xing4_0.expert_group_used_count` | `[1]` |
| `xing4_0.expert_gating_func` | `[2]` |
| `xing4_0.leading_dense_block_count` | `[2]` |
| `xing4_0.vocab_size` | `[131072]` |
| `xing4_0.attention.q_lora_rank` | `[768]` |
| `xing4_0.attention.kv_lora_rank` | `[512]` |
| `xing4_0.attention.key_length` | `[576]` |
| `xing4_0.attention.value_length` | `[512]` |
| `xing4_0.attention.key_length_mla` | `[192]` |
| `xing4_0.attention.value_length_mla` | `[128]` |
| `xing4_0.expert_feed_forward_length` | `[1024]` |
| `xing4_0.expert_shared_count` | `[1]` |
| `xing4_0.expert_weights_scale` | `[2.]` |
| `xing4_0.expert_weights_norm` | `[ True]` |
| `xing4_0.rope.dimension_count` | `[64]` |
| `xing4_0.rope.scaling.yarn_log_multiplier` | `[0.1]` |
| `xing4_0.nextn_predict_layers` | `[1]` |
| `xing4_0.hyper_connection.count` | `[4]` |
| `xing4_0.hyper_connection.sinkhorn_iterations` | `[20]` |
| `xing4_0.hyper_connection.epsilon` | `[1.e-06]` |
| `tokenizer.ggml.model` | `[108 108  97 109  97]` |
| `tokenizer.ggml.pre` | `[100 101 102  97 117 108 116]` |
| `tokenizer.ggml.tokens` | `[233 161 134]` |
| `tokenizer.ggml.scores` | `[-130675.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[25]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[592]` |
| `quantize.imatrix.chunks_count` | `[8]` |

**general.name:**
```jinja
[ 50  57  98  95  49  50  46  48  53  84  95 119 111 100 115  97  95 105
 100  50  48  95 115  51  95  50  53  54 107  95  56  95  50  54  48  57
  48  51  95  49  48  48  48  95 102 105 110  97 108  95 104 102]
```

**general.finetune:**
```jinja
[ 50  57  98  95  49  50  46  48  53  84  95 119 111 100 115  97  95 105
 100  50  48  95 115  51  95  50  53  54 107  95  56  95  50  54  48  57
  48  51  95  49  48  48  48  95 102 105 110  97 108  95 104 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  10  10  10]
```

**quantize.imatrix.file:**
```jinja
[ 47 103 101 109 105 110 105  47 115 112  97  99 101  47 115 104 105 120
 113  55  47 105 109  97 116 114 105 120  46 100  97 116]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 103 101 109 105 110 105  47 115 112  97  99 101  47 115 104 105 120
 113  55  47  99  97 108 105  98 114  97 116 105 111 110  46 116 120 116]
```

---

## GLM-4.7-Flash-UD_Q6_K_XL.gguf
**Pfad:** `Zhipu AI\GLM 4.7 Flash\GLM-4.7-Flash-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[844]` |
| `GGUF.kv_count` | `[60]` |
| `general.architecture` | `[100 101 101 112 115 101 101 107  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 108 109  45  52  46  55  45  70 108  97 115 104]` |
| `general.basename` | `[ 71 108 109  45  52  46  55  45  70 108  97 115 104]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 54  52 120  50  46  54  66]` |
| `general.license` | `[109 105 116]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71  76  77  32  52  46  55  32  70 108  97 115 104]` |
| `general.base_model.0.organization` | `[ 90  97 105  32  79 114 103]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `deepseek2.block_count` | `[47]` |
| `deepseek2.context_length` | `[202752]` |
| `deepseek2.embedding_length` | `[2048]` |
| `deepseek2.feed_forward_length` | `[10240]` |
| `deepseek2.attention.head_count` | `[20]` |
| `deepseek2.attention.head_count_kv` | `[1]` |
| `deepseek2.rope.freq_base` | `[1.e+06]` |
| `deepseek2.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `deepseek2.expert_used_count` | `[4]` |
| `deepseek2.expert_group_count` | `[1]` |
| `deepseek2.expert_group_used_count` | `[1]` |
| `deepseek2.expert_gating_func` | `[2]` |
| `deepseek2.leading_dense_block_count` | `[1]` |
| `deepseek2.vocab_size` | `[154880]` |
| `deepseek2.attention.q_lora_rank` | `[768]` |
| `deepseek2.attention.kv_lora_rank` | `[512]` |
| `deepseek2.attention.key_length` | `[576]` |
| `deepseek2.attention.value_length` | `[512]` |
| `deepseek2.attention.key_length_mla` | `[256]` |
| `deepseek2.attention.value_length_mla` | `[256]` |
| `deepseek2.expert_feed_forward_length` | `[1536]` |
| `deepseek2.expert_count` | `[64]` |
| `deepseek2.expert_shared_count` | `[1]` |
| `deepseek2.expert_weights_scale` | `[1.8]` |
| `deepseek2.expert_weights_norm` | `[ True]` |
| `deepseek2.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 108 109  52]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 52 56 55 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 196 189  32 194 191]` |
| `tokenizer.ggml.eos_token_id` | `[154820]` |
| `tokenizer.ggml.padding_token_id` | `[154821]` |
| `tokenizer.ggml.bos_token_id` | `[154822]` |
| `tokenizer.ggml.eot_token_id` | `[154827]` |
| `tokenizer.ggml.unknown_token_id` | `[154820]` |
| `tokenizer.ggml.eom_token_id` | `[154829]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[607]` |
| `quantize.imatrix.chunks_count` | `[85]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  97 105  45 111 114 103  47  71  76  77  45  52
  46  55  45  70 108  97 115 104]
```

**tokenizer.chat_template:**
```jinja
[ 91 103  77 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 71  76  77  45  52  46  55  45  70 108  97 115 104  45  71  71  85  70
  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103
 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  71  76  77  45  52  46  55  45  70 108  97 115 104  46 116 120
 116]
```

---

## mtp-GLM-4.7-Flash-Q4_K_M.gguf
**Pfad:** `Zhipu AI\GLM 4.7 Flash\mtp-GLM-4.7-Flash-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[27]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[100 101 101 112 115 101 101 107  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | *Siehe Code-Block unten* |
| `general.size_label` | `[ 54  52 120  49  46  51  66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `deepseek2.block_count` | `[48]` |
| `deepseek2.context_length` | `[202752]` |
| `deepseek2.embedding_length` | `[2048]` |
| `deepseek2.feed_forward_length` | `[10240]` |
| `deepseek2.attention.head_count` | `[20]` |
| `deepseek2.attention.head_count_kv` | `[1]` |
| `deepseek2.rope.freq_base` | `[1.e+06]` |
| `deepseek2.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `deepseek2.expert_count` | `[64]` |
| `deepseek2.expert_used_count` | `[4]` |
| `deepseek2.expert_group_count` | `[1]` |
| `deepseek2.expert_group_used_count` | `[1]` |
| `deepseek2.leading_dense_block_count` | `[1]` |
| `deepseek2.vocab_size` | `[154880]` |
| `deepseek2.attention.q_lora_rank` | `[768]` |
| `deepseek2.attention.kv_lora_rank` | `[512]` |
| `deepseek2.attention.key_length` | `[576]` |
| `deepseek2.attention.value_length` | `[512]` |
| `deepseek2.attention.key_length_mla` | `[256]` |
| `deepseek2.attention.value_length_mla` | `[256]` |
| `deepseek2.expert_feed_forward_length` | `[1536]` |
| `deepseek2.expert_shared_count` | `[1]` |
| `deepseek2.expert_weights_scale` | `[1.8]` |
| `deepseek2.expert_weights_norm` | `[ True]` |
| `deepseek2.rope.dimension_count` | `[64]` |
| `deepseek2.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 108 109  52]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 52 56 55 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 196 189  32 194 191]` |
| `tokenizer.ggml.eos_token_id` | `[154820]` |
| `tokenizer.ggml.padding_token_id` | `[154820]` |
| `tokenizer.ggml.bos_token_id` | `[154822]` |
| `tokenizer.ggml.eot_token_id` | `[154827]` |
| `tokenizer.ggml.unknown_token_id` | `[154820]` |
| `tokenizer.ggml.eom_token_id` | `[154829]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.name:**
```jinja
[122  97 105  45 111 114 103  47  71  76  77  45  52  46  55  45  70 108
  97 115 104]
```

**general.finetune:**
```jinja
[ 55 100 100  50  48  56  57  52  97  54  52  50  97  48  97  97  50  56
  55 101  57  56  50  55  99  98  49  97  49 102  55 102  57  49  51  56
  54  98  54  55]
```

---

## GLM-OCR-Q8_0.gguf
**Pfad:** `Zhipu AI\GLM OCR\GLM-OCR-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[179]` |
| `GGUF.kv_count` | `[32]` |
| `general.architecture` | `[103 108 109  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.size_label` | `[56 57 49 77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[107 111]` |
| `glm4.block_count` | `[17]` |
| `glm4.context_length` | `[131072]` |
| `glm4.embedding_length` | `[1536]` |
| `glm4.feed_forward_length` | `[4608]` |
| `glm4.attention.head_count` | `[16]` |
| `glm4.attention.head_count_kv` | `[8]` |
| `glm4.rope.dimension_sections` | `[0]` |
| `glm4.rope.freq_base` | `[10000.]` |
| `glm4.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `glm4.attention.key_length` | `[128]` |
| `glm4.attention.value_length` | `[128]` |
| `glm4.rope.dimension_count` | `[128]` |
| `glm4.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[ 99 104  97 116 103 108 109  45  98 112 101]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 53 57 51 57 49 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 133 196 164  97  32 116]` |
| `tokenizer.ggml.eos_token_id` | `[59246]` |
| `tokenizer.ggml.padding_token_id` | `[59246]` |
| `tokenizer.ggml.eot_token_id` | `[59253]` |
| `tokenizer.ggml.unknown_token_id` | `[59246]` |
| `tokenizer.ggml.bos_token_id` | `[59246]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template:**
```jinja
[ 91 103  77 ...  37 125  10]
```

---

## mmproj-GLM-OCR-Q8_0.gguf
**Pfad:** `Zhipu AI\GLM OCR\mmproj-GLM-OCR-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[348]` |
| `GGUF.kv_count` | `[21]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.size_label` | `[52 51 52 77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[107 111]` |
| `general.file_type` | `[7]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1536]` |
| `clip.vision.image_size` | `[336]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[103 108 109  52 118]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `general.quantization_version` | `[2]` |

---

