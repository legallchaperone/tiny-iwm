# World-model runtime

`training.py` validates the precision/device/distribution scope, applies the
strict CUDA math policy, and constructs the configured AdamW optimizer and constant scheduler.
`ema.py` owns the floating model shadow and records its decay and update count.

`checkpoint.py` has two separate load paths:

- `initialize_new_stage` loads either raw or EMA model weights, sets the child
  checkpoint parent identity, and creates new optimizer, scheduler, EMA, step,
  and epoch state. The EMA rule is explicitly `copy_loaded_model`.
- `resume_same_run` requires an exact run ID match and restores model,
  optimizer, scheduler, EMA, step, epoch, Python RNG, CPU RNG, and CUDA RNG.

Every checkpoint includes JSON snapshots of model, codec, camera, resolved
configuration, stage, run, and parent provenance. Saving is atomic and writes a
SHA-256 sidecar; loading rejects a changed file before deserialization.
