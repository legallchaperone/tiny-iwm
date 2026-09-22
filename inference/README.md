# Autoregressive latent history

`rollout_latents` is the shared Flow Matching loop. It accepts the exact clean
initial-condition prefix, then denoises each remaining chunk from `t=1` to `t=0` with
the same Euler solver in `reference` and `cached` modes. The caller supplies
the codec-derived `VideoLayout` and full-trajectory camera projection.

`HistorySession.predict` never commits target-step KV. After the solver finishes,
`commit_clean` reruns that chunk at `t=0` and appends its KV. Earlier chunks also
use `t=0` in reference recomputation. Stage B training exposes `model_time`
with this same clean-history/variable-target-time convention while retaining
`flow_time` for the FM target and loss.

Each session is bound to an episode, checkpoint, condition, and CFG branch.
Create a separate session for every such identity. The initial condition may
end inside the first chunk but must align with temporal patches. Video writing
and frame-level output checks belong to later rollout issues.
