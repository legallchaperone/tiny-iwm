# Fixed SANA-WM subset

`data/manifests/sana_wm_eval_subset_v1.json` pins the public official benchmark
at dataset revision `0e5279250e7a531b83ce4e9ee2870fab7003be0b`. It selects
one seeded scene from each of four categories, in both 60-second splits, with
generation seed 42: eight scene/split/seed rows. Selection is independent of
model checkpoint and can be reused across representations and runs.

To reproduce it, download only `scene_set/kept_scenes.txt`, each split's
`sanawm_export_v2/run_manifest.jsonl`, and each split's
`scene_trajectories_v2.json`, plus the four selected images and eight selected
camera NPZ files, from that dataset revision into the official directory
layout. The selection includes the SHA-256 of every selected asset. Then run:

```bash
python -m evaluation.selection --source-root /path/to/SANA-WM-Bench \
  --revision 0e5279250e7a531b83ce4e9ee2870fab7003be0b \
  --output data/manifests/sana_wm_eval_subset_v1.json \
  --per-category 1 --selection-seed 2026 --generation-seed 42
```

The command verifies the 80-scene source set and refuses to overwrite a changed
selection. Each row preserves the official 961-frame/16-FPS camera trajectory.
The upstream scoring manifest separately requests 960 trimmed video frames;
both values are recorded. Revisit pair lists stay in the pinned official
trajectory metadata. Their count, maximum frame index, and SHA-256 are recorded
per row so scoring can verify it loaded the same pairs without redefining them.

`generation_identity` hashes the frozen selection, checkpoint and weight flavor,
resolved model configuration, codec weights, normalization, encoding and execution settings,
spatial resolution, preprocessing, the complete temporal rollout layout,
source conditions, resolved camera/text conditioning settings, implementation
revision, numerical settings and hardware backend, seed, and sampler. `claim_output_directory` writes a
matching `identity.json` into an identity-specific directory and rejects a
conflicting claim. Generation and scoring should verify this identity before
using any video from that directory.
Before creating an identity, it rehashes the selected image and camera files
under the supplied source root, catching changes after CPU preparation.

Official generation admits only the source-image initial prefix. The sampler
records `initial_history_policy: source_image_only`; a resumed or prefetched
rollout with additional clean chunks is outside this evaluation identity.
The shared `rollout_latents` enforces this policy when it is passed from the
generation identity's sampler.
Each row is generated alone (`batch_size: 1`) so its seeded noise stream does
not depend on batch position or size.

## Official metric adapter

`evaluation.official` wraps Sana's unchanged metric entry points at commit
`f9178744c096dcf2a2ea773da183e341bcbeb044` (Apache-2.0). No official
source is vendored or modified. Keep that checkout and the benchmark release
separate from model training and generation; install the upstream evaluation
dependencies in a scoring environment only.

The `stage` command verifies the frozen metadata and each `identity.json`,
requires a single compatible run identity across scenes, then links each
`video.mp4` into the official `<split>/<scene>_generated.mp4` layout. It writes
an immutable `staging.json` recording the source commit, license, zero local
modifications, selected scene identities, and source paths. For example:

```bash
python -m evaluation.official stage \
  --selection data/manifests/sana_wm_eval_subset_v1.json \
  --benchmark-root /path/to/SANA-WM-Bench \
  --outputs-root /path/to/identity-outputs \
  --method-dir /path/to/evaluation-method --seed 42
```

Run the same command with `score` and `--official-repo /path/to/Sana` to
evaluate existing videos without regenerating them. The adapter pins the
official checkout, rejects modified metric scripts, and calls the official
VBench/revisit/temporal and Pi3 camera entry points. Their raw per-scene files
remain under the method directory. It uses the official nine VBench dimensions,
five revisit pairs per scene, 16 FPS reference, 10-second windows, and no
first-frame skip for this unrefined model. Camera/Pi3 evaluation uses GPU, so
run `stage` and validate video completeness on CPU before invoking `score`.
