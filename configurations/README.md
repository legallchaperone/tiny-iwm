# configurations

We use [Hydra](https://hydra.cc/docs/intro/) to manage configurations. Change/Add the yaml files in this folder
to change the default configurations. You can also override the default configurations by
passing command line arguments.

The default `config.yaml` composes the M0 world-model skeleton and performs no
training (`experiment.tasks=[]`). Scientific values that still require an
experiment or a memory profile are written as `null`, rather than receiving an
implicit guess. W&B is disabled by default, so composing and launching this
baseline locally requires no credentials. Example configurations remain
available through explicit Hydra overrides.

Run the M0 contract check on CPU without loading data or model weights:

```bash
python -m scripts.preflight --output-dir outputs/preflight +name=m0-preflight runtime.accelerator=cpu
```

The command validates random DiT initialization and mutually exclusive
`checkpoint.init_from` / `checkpoint.resume_from` settings, then writes
`resolved_config.yaml` and `provenance.json` with W&B disabled by default.

## Temporal contract

Temporal geometry has exactly five inputs: `train.window_rgb_frames`,
`temporal.chunk_latent_frames`, `rollout.rgb_frames`,
`temporal.initial_condition_rgb_frames`, and `temporal.fps`. Codec padding,
latent/token/chunk ranges, and physical duration are derived by `VideoLayout`;
do not repeat FPS, frame counts, or duration in dataset/stage configuration.
The default Hydra config uses a 161-frame training window and the named
`rollout/minute` 961-frame preset. Select `rollout=debug_81` for short debugging.
The published Stage A/B run YAMLs retain their original 961-frame training
meaning through an explicit compatibility adapter. Only the 961-frame, 16 FPS
preset is eligible for the formal minute evaluation gate.
`resources.available_rgb_frames` checks the training source; validate a
generation source separately with `purpose="rollout"` when generating.

All configurations are automatically saved in wandb run.

---

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template). By its MIT license, you must keep the above sentence in `README.md` and the `LICENSE` file to credit the author.

Prepared latent-cache windows must end on a codec-group boundary. The CPU preflight rejects a requested RGB length whose final cached latent also contains frames beyond that window; choose an aligned length (such as 161 frames with temporal compression 8) or use an RGB path that can encode its padded boundary itself.

## Model construction

`experiments.build.build_model` resolves a model once for training, rollout, or
probing. The default `model=joint_dit` preserves the previous module names and
checkpoint keys. `model=joint_dit_swiglu` selects a structurally different
SwiGLU feed-forward network; `model.components.norm=rms_norm` and
`model.components.position=none` are explicit internal alternatives. New
component implementations must be wired into their small construction function.
Unknown names fail during CPU preflight. A changed architecture or component
requires a new checkpoint identity; old weights are never loaded implicitly.
