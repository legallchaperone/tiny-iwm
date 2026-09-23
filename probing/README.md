# Bounded feature capture

`capture_forward` calls the existing `JointVideoDiT.forward` with named
observation points and returns the same velocity (and optional KV cache) as a
normal call. It never changes model weights, inputs, random state, or the
gradient path. `FeatureRecorder` detaches each selected token before computing
summary statistics or writing an optional raw tensor.

The caller supplies `CaptureContext` for sample/run/checkpoint/config identity,
generation and training seeds, rollout and Flow Matching time, branch, and
forward purpose. Each `TokenSelection` supplies a local token index, its role
(for example `history` or `target`), and physical coordinates. Cached chunk
forwards use local token indices, so the caller must give the corresponding
global physical position explicitly. Records are separate JSON files with one
event per selected point and token; raw `.pt` tensors are opt-in.

`standard.yaml` requests one layer and one token, with a 256-record cap and no
raw retention. `off.yaml` disables capture. A full-minute all-layer/all-step
dump is never the default. The recorder raises when the record or serialized
raw-byte budget would be exceeded, including files from a prior process.
Writers sharing a capture directory reserve their budget under a filesystem
lock before publishing records.
`checkpoint_contexts` can be passed as the
`context_fn` to non-reentrant `torch.utils.checkpoint.checkpoint`; it suppresses
records during activation recomputation while leaving the recomputed forward
unchanged.

## Held-out controlled replay

`controlled_replay` accepts one held-out sample's ground-truth and generated
clean histories through the same `HistorySession` reference forward. It fixes
the clean target, noise, Flow Matching time, camera projection, text-disabled
condition, and selected probe tokens. Its two forwards differ only in history
source. The returned report records hashes of those fixed inputs, both history
hashes, provenance IDs, and prediction error/drift. Join it to rollout or
revisit results using `selection_id`, `generation_id`, and `sample_id`; the
report is labeled as a controlled replay rather than an observed rollout.
