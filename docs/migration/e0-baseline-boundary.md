# E0: baseline and migration boundary

This inventory records the repository state at `d5f64b2` before the E0–E6
refactor. It is a code and artifact inventory, not a claim that every command
was rerun during this milestone.

## Retained baseline paths

| Concern | Existing source of truth | Migration boundary |
|---|---|---|
| Hydra task dispatch | `main.py`, `configurations/config.yaml`, `experiments/world_model.py` | Keep the template entrypoint. New resolved configuration and builders will be introduced behind it. |
| Stage A recipe and launcher | `configurations/runs/stage_a_baseline.yaml`, `scripts/modal_stage_a_train.py`, `scripts/modal_stage_a_readiness.py` | Preserve the existing run identity, initialization, and invocation while later milestones add a shared construction path. |
| Stage B recipe and launcher | `configurations/runs/stage_b_baseline.yaml`, `scripts/modal_stage_b_train.py`, `scripts/modal_stage_b_initialize.py` | Preserve teacher-forced Stage A-to-B semantics and the existing checkpoint provenance. |
| Model and training math | `algorithms/world_model/models/`, `algorithms/world_model/flow.py`, `algorithms/world_model/training_batch.py` | Keep the current DiT, PRoPE, FM, and A/B batch implementations as the first registered implementations; do not copy their math into new adapters. |
| Time and camera contracts | `core/video_layout.py`, `core/camera.py`, `core/types.py` | Keep these as the single existing layout and camera contracts while E1 makes their values configurable end to end. |
| Representation and rollout state | `representations/`, `inference/history.py`, `inference/rollout.py` | Preserve codec identity, clean-history commit, and reference/cache behavior while E2/E3 add construction and sampler seams. |
| Evaluation and probes | `evaluation/`, `probing/` | Keep the official SANA selection/scoring behavior available as one protocol; later adapters must not redefine its identity. |
| External code provenance | `third_party/UPSTREAMS.yaml` | Extend the current registry when a new implementation is actually introduced; a design reference alone is not an imported implementation. |

## Evidence and its limits

- The checked-in Stage A/B run YAMLs and Modal launchers describe the existing
  bounded recipes. The scripts documentation records the commands and their
  expected data/checkpoint locations.
- `artifacts/m1` through `artifacts/m4` contain machine-readable records for
  earlier codec, correctness, profiling/readiness, initialization, and Stage B
  gates. They are historical evidence attached to those runs, not evidence that
  this refactor has rerun them.
- The Stage A/B training configs specify 100-step runs. They establish useful
  engineering and resume paths; they do not establish a sufficiently trained
  model or a scientific claim about full-minute generation quality.
- Checkpoint paths such as `/stage-a/checkpoints/...` refer to the persistent
  Modal Volume used by the launchers. The design repository does not contain
  those checkpoint binaries, so their current remote availability is not
  asserted here.
- `third_party/UPSTREAMS.yaml` records pinned upstream revisions and per-file
  provenance. Its `current_state` values are live claims and must stay aligned
  with edits since the imported commit. E0 reconciles locally changed template
  mappings as `adapted` / `modified-after-import`; an upstream reference alone
  does not imply that current code is unchanged upstream code.

## Compatibility rules for E1–E5

1. Keep the current Stage A/B commands and run configs interpretable until a
   shared path can construct the same implementations and semantics.
2. Do not rename checkpoint keys or require retraining just to move code behind
   a builder. Treat incompatible model, objective, or codec changes as explicit
   checkpoint compatibility failures.
3. Treat 961 frames, 16 FPS, FM, the SANA data format, and official SANA scoring
   as choices of the existing recipe/protocol. They are not global platform
   restrictions.
4. Record new implementations and verification evidence where they belong;
   keep smoke completion, protocol completeness, and scientific conclusions as
   separate claims.

## Planned implementation seams

The current scripts construct their own `JointVideoDiT` and `FlowMatchSpec`,
and `inference/rollout.py` owns the Euler update. The smallest intended seams
are therefore:

- E1: resolve temporal values once and pass the resulting layout/window through
  data preparation, camera, model, history, and writer code.
- E2: add a builder that initially returns the current model, then route train,
  generate, and probe through that builder before opening real component
  overrides.
- E3: adapt existing FM, Stage A/B policy, and Euler sampling first; add a new
  objective only with its own prediction/noise and sampler semantics.
- E4: adapt data sources, evaluators, and probes around their actual input and
  capability requirements rather than SANA/FM assumptions.

This boundary is a migration aid, not an additional recipe or a new baseline.
