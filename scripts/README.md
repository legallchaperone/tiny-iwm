# scirpts

`scripts` folder contains bash scripts for you to scale up your project on cloud.
Don't put your jupyter notebooks here! They belongs to `debug` folder.

General scripts that are useful for all projects can be put in the `script` folder directly.

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.

## M1 codec gate

`python -m scripts.validate_m1_codec` runs a selected validation or test split
through the strict SANA reader and a frozen codec supplied by a local
`module.path:factory`. It writes one atomic JSON report and exits nonzero if any
sample fails reading, alignment, encode/decode, shape, or causality checks. See
`python -m scripts.validate_m1_codec --help` for all required paths and options.

The released streaming codec factory is
`representations.sana_wm_streaming:create_official_sana_wm_streaming_codec`.
It requires local paths in `SANA_WM_VAE_PATH` and `SANA_WM_UPSTREAM_PATH` and
verifies the published 4.89 GB safetensors SHA-256 before loading.

### Reproduce the official 961-frame run on Modal

Run `modal run scripts/modal_m1_gate.py`. The `prepare_assets` CPU function
first downloads weights and creates the held-out validation clip in a persistent
Modal Volume. Only the subsequent `run_gate` function requests a GPU, so cache
misses and retries do not spend GPU time on downloads.

## M2 correctness gate

`python -m scripts.validate_m2_correctness` runs the deterministic CPU
debug-size M2 gate. It connects Stage A batch construction, native Flow
Matching, the joint DiT, camera PRoPE, backward gradients, and named probes.
Pass `--output` to write the machine-readable report used by CWX-18.

## M3 full-length profile

`modal run scripts/modal_m3_profile.py` measures the 300M model with the full
961-frame temporal layout on one A10G. The Modal image build installs PyTorch
before the GPU function starts; this job has no model or dataset download. The
GPU function records training and 16-step rollout time, peak CUDA memory,
parameters, tokens, projected KV size, attention settings, and hardware. It
also compares resolution and patch alternatives while holding 961 frames,
16 FPS, and 60 seconds fixed.

## Stage A baseline

`modal run scripts/modal_stage_a_train.py` reproduces the CWX-21 baseline. A
CPU function uses HTTP Range requests to extract a fixed, scene-disjoint subset
from the pinned official SANA-WM latent archive and writes it to the persistent
`tiny-iwm-stage-a` Volume. The A10G function starts only after preparation and
does not download anything. It trains 100 steps, validates EMA weights at steps
50 and 100, and keeps the lowest validation-loss checkpoint with complete
same-run resume state in the Volume.
