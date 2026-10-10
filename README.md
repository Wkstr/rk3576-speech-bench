# RK3576 ASR / TTS Benchmark

[![Live site](https://img.shields.io/badge/site-GitHub%20Pages-0d9488?logo=github)](https://wkstr.github.io/rk3576-speech-bench/)
[![platform](https://img.shields.io/badge/platform-RK3576%20NPU%20%2B%20Cortex--A72-blue)](https://www.seeedstudio.com/)
[![harness](https://img.shields.io/badge/harness-sherpa--onnx%20%2B%20librknnrt%202.2.0-orange)](https://github.com/k2-fsa/sherpa-onnx)

English | [中文](README.zh-CN.md)

Reproducible on-device benchmarks for speech recognition (ASR) and speech synthesis (TTS) on the Rockchip RK3576 board (reComputer, 8GB): one harness, two backends — self-converted fp16 RKNN on the NPU vs fp32 on CPU. All numbers carry Wilson 95% confidence intervals; raw JSON results and synthesized audio samples are checked into the repo.

**👉 Interactive comparison site (sortable tables, charts, TTS audio playback, zh/en toggle): <https://wkstr.github.io/rk3576-speech-bench/>**

**📄 Full report: [report/REPORT.md](report/REPORT.md)** (Chinese main text + English abstract — methodology, limitations, and reproduction commands included)

## Results at a glance (snapshot 2026-10-10; the site always shows the latest)

### ASR (streaming zipformer, 30 clips per language)

| Model | Backend | zh CER | en WER | RTF (zh / en) |
|---|---|---|---|---|
| small (bilingual) | **NPU fp16 self-converted** | 10.1% | 19.3% | 0.14 / 0.15 |
| small (bilingual) | CPU fp32 | 10.1% | 19.3% | 0.16 / 0.17 |
| large (bilingual) | **NPU fp16 self-converted** | 7.6% | 11.4% | 0.41 / 0.41 |
| large (bilingual) | CPU fp32 | 7.8% | 11.4% | 0.41 / 0.44 |

Key points: the self-converted NPU fp16 models produce output near-identical to CPU fp32 (zh 30/30 and 29/30; en 27/30 and 30/30 per-clip match) — zero fp16 degradation. The NPU gives no speedup on large (its bottleneck is CPU-side decoding). The vendor's official RKNN zipformer is broken upstream ([sherpa-onnx#2861](https://github.com/k2-fsa/sherpa-onnx/issues/2861)), so all NPU numbers here come from self-converted models (rknn-toolkit2 2.3.2).

### TTS (CPU fp32, self-built text set)

| Model | Lang | RTF | Round-trip intelligibility ER (lower bound) |
|---|---|---|---|
| Matcha + vocos | zh+en | **0.25** | 36.7% |
| Piper (huayan) | zh | 0.34 | **13.4%** |
| Piper (ryan) | en | 0.32 | 21.0% |
| MMS | en | 1.66 | 31.2% |
| Kokoro 82M v1.1 | zh+en | 2.87 | 29.8% |

Key points: Matcha is the fastest but its English intelligibility is poor; Piper is the balanced recommendation; Kokoro is batch-only on the quad-core A72; MMS serves as a floor reference. The round-trip metric is a lower bound (number readouts and homophones are counted as errors, which keeps cross-model comparison fair) — see the methodology section of the report. A hybrid Matcha variant (acoustic model on CPU + vocos vocoder on NPU) reaches RTF 0.18.

## Repository layout

```
bench/    measurement harness (run_asr.py / run_tts.py / sysmon.py / run_p1.sh / run_p3.sh)
results/  raw result JSON + 10 Hz resource sampling (single source of truth for the report and site)
report/   build_report.py → REPORT.md (bilingual)
site/     build_site.py → index.html (self-contained single page) + audio/ (84 TTS samples)
models.yaml / datasets.yaml  model and dataset registries
```

## Reproduce

```bash
bash setup/install.sh && bash setup/check_env.sh     # environment (librknnrt 2.2.0 + sherpa-onnx)
.venv/bin/python scripts/prepare_data.py en          # and zh; dataset notes in AGENTS.md
bash bench/run_p1.sh                                 # ASR: 4 configs × zh/en
bash bench/run_p3.sh                                 # TTS: 5 models
.venv/bin/python report/build_report.py              # report
.venv/bin/python site/build_site.py --copy-audio 99  # site
```

Data and model sources: [LibriSpeech test-clean](https://www.openslr.org/12), [FLEURS cmn_hans_cn](https://huggingface.co/datasets/google/fleurs), [sherpa-onnx model hub](https://github.com/k2-fsa/sherpa-onnx), [rknn-toolkit2 2.3.2](https://github.com/airockchip/rknn-toolkit2).
