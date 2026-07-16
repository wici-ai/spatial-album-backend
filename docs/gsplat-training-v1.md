# gsplat Training v1

`wici.gsplat-training.v1` is the bounded multi-view reconstruction contract
served by Orbit. Public deployment routes keep the existing Caddy port and
prefix:

```text
POST   /orbit/training/jobs
GET    /orbit/training/jobs/<job-id>
GET    /orbit/training/jobs/<job-id>/result
DELETE /orbit/training/jobs/<job-id>
```

Caddy removes `/orbit`; the internal handler receives `/training/...`.
`GET /orbit/health` advertises the protocol only when its worker is enabled.

## Create

Create uses fixed-length `multipart/form-data`. It contains one `protocol`
field, one JSON `manifest` field, and one or more repeated `image` fields. Each
image part must include `X-Wici-Candidate-Id`; the uploaded ID set must exactly
match `selectedCandidateIds`. The current server requires 2-64 images by
default.

```json
{
  "protocol": "wici.gsplat-training.v1",
  "sceneId": "room-1",
  "anchorCandidateId": "image-a",
  "selectedCandidateIds": ["image-a", "image-b"],
  "options": {
    "preloadTrainingData": true,
    "preloadChunkBytes": 786432,
    "batchSize": 8,
    "equivalentSteps": 30000
  }
}
```

The server rejects unknown manifest and multipart fields. It also applies
total-body, per-image, image-count, step-count, and active-job limits before a
worker is admitted.

## Lifecycle

Create returns HTTP 202 and a job record. Polling returns `queued`, `running`,
`succeeded`, `failed`, or `cancelled`, plus stage, step, total steps, elapsed
time, and preload metadata. A successful result is a 32-byte-per-Gaussian
`.splat` stream compatible with the Android GLES viewer. Clients must delete a
terminal job after downloading it; deleting a running job requests worker
cancellation and can return a retryable conflict until termination completes.

## CUDA-resident training path

The built-in worker normalizes the uploaded images, runs CPU COLMAP, and calls
the repository-pinned gsplat 1.5.2 trainer. The deploy patch adds a mandatory
preload mode: factor-reduced training tensors, intrinsics, poses, IDs, and masks
are copied once to CUDA in sub-1 MiB chunks. Each optimization step then selects
and crops its batch on the CUDA device instead of decoding, pinning,
transferring, and synchronizing image tensors repeatedly. Gaussian
rasterization, gradient accumulation, densification, and optimizer work remain
native gsplat CUDA operations.

The `equivalentSteps` value preserves a comparable image-sample budget:
batch 8 scales 30,000 single-image-equivalent steps into 3,750 optimizer
iterations. This is an intentional throughput optimization, not a claim that
the number of optimizer updates is unchanged.

## Operational boundary

This endpoint is intended for a trusted LAN. It has bounded resource admission
and strict parsing, but does not by itself add public-internet authentication,
TLS, per-user isolation, or durable restart recovery. Job workspaces contain
uploaded media until the client deletes the terminal job. Set
`WICI_GSPLAT_TRAINING_ENABLED=0` to withdraw the capability.
