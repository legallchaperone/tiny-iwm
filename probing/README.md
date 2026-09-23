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
dump is never the default. The recorder raises when the record or raw-byte
budget would be exceeded. `checkpoint_contexts` can be passed as the
`context_fn` to non-reentrant `torch.utils.checkpoint.checkpoint`; it suppresses
records during activation recomputation while leaving the recomputed forward
unchanged.
