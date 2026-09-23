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
codec weights, normalization, encoding policy, spatial resolution, preprocessing, the complete temporal rollout layout,
source conditions, resolved camera/text conditioning settings, implementation
revision, numerical settings and hardware backend, seed, and sampler. `claim_output_directory` writes a
matching `identity.json` into an identity-specific directory and rejects a
conflicting claim. Generation and scoring should verify this identity before
using any video from that directory.
