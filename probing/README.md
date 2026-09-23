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

`python -m probing.metrics --capture-root CAPTURE --output report.json` computes
norm distribution, centered singular values, and energy-based effective rank
from the opt-in raw tokens. Rows are selected sample/token observations;
columns are feature channels; centering subtracts the mean of each channel
across rows. `--compare-root OTHER` joins the same sample, generation seed,
rollout time, FM time, token position/type, layer, branch, and forward purpose
before computing centered linear CKA and, for equal channel counts, mean row
cosine and L2 drift. `--history-alignment` pairs controlled GT-history and
generated-history records from the same checkpoint/config and reports drift
separately by layer and each rollout/FM time. Reports link to the exact raw
files and hash both record metadata and raw tensor bytes. At least two matched
observations per comparison group are required.
