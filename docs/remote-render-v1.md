# Remote render protocol v1

`remote-render-v1` is the strict client/server contract for an interactive WiCi
Gaussian-splat render session. Unknown JSON or multipart fields are rejected;
clients must negotiate the exact protocol capability and must not infer support
from unrelated Orbit endpoints. All responses are `Cache-Control: no-store`.

The routes below are internal Orbit routes. Deployment keeps the existing
`/orbit/*` reverse-proxy prefix, so an external client uses
`/orbit/render/sessions...`. Authentication is outside this protocol; exposing
these routes to an untrusted network without a separate access-control layer is
not supported.

## Create a session

`POST /render/sessions` accepts one of two sources.

For a curated gallery asset, send `Content-Type: application/json`:

```json
{
  "protocol": "remote-render-v1",
  "source": {"type": "gallery", "photoId": "opaque-gallery-id"}
}
```

`photoId` is an opaque identifier. The server adapter maps it through the
gallery resolver; it is never interpreted as a caller-selected file path.

For an imported image, send `multipart/form-data` with exactly these parts:

| Part | Value |
| --- | --- |
| `protocol` | UTF-8 text `remote-render-v1` |
| `sourceType` | UTF-8 text `image` |
| `image` | One `image/jpeg` or `image/png` file within the advertised limit |

On success the server returns HTTP 201:

```json
{
  "protocol": "remote-render-v1",
  "sessionId": "opaque_url_safe_token",
  "camera": {
    "convention": "opencv-w2c-row-major-4x4",
    "w2c": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2, 0, 0, 0, 1],
    "intrinsics": {"fx": 900, "fy": 900, "cx": 320, "cy": 240},
    "width": 640,
    "height": 480
  },
  "pivot": [0, 0, 0],
  "focusDistance": 2,
  "source": {"width": 1536, "height": 1024},
  "limits": {
    "maxFramePixels": 1638400,
    "maxUploadBytes": 20971520,
    "sessionIdleTtlSeconds": 120
  }
}
```

The returned camera, pivot, and positive focus distance are authoritative
initial navigation metadata. A client must not substitute a fixed scene pivot.

## Camera convention

`w2c` is a 16-number row-major world-to-camera homogeneous transform. Camera
coordinates follow OpenCV: +X right, +Y down, +Z forward. The last row must be
`[0, 0, 0, 1]`; the upper-left 3x3 must be orthonormal with determinant +1.
Translation is in the same world units as the splat. The pinhole projection is
`u = fx * X/Z + cx`, `v = fy * Y/Z + cy`.

`fx` and `fy` must be positive. `cx` and `cy` must lie inside the inclusive
viewport bounds. Width and height are positive integers whose product does not
exceed `maxFramePixels`. When a viewport is resized independently in X and Y,
scale `fx` and `cx` by `newWidth/oldWidth`, and `fy` and `cy` by
`newHeight/oldHeight`; do not alter `w2c`. Canonical cross-platform vectors are
in `backend/tests/fixtures/remote-render-v1-camera-vectors.json`.

## Request a frame

Send `POST /render/sessions/{sessionId}/frames` as JSON:

```json
{
  "protocol": "remote-render-v1",
  "requestId": 42,
  "mode": "interactive",
  "camera": {
    "convention": "opencv-w2c-row-major-4x4",
    "w2c": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -2, 0, 0, 0, 1],
    "intrinsics": {"fx": 900, "fy": 900, "cx": 320, "cy": 240},
    "width": 640,
    "height": 480
  }
}
```

`requestId` is a non-negative integer and must strictly increase within the
session. The server records a request before it waits for the shared render
lock, then checks it again inside that lock. Therefore a queued request replaced
by a newer request returns `request_superseded` without calling the engine.

An `interactive` response is HTTP 200 `image/jpeg`. A `capture` response is
HTTP 200 `application/zip`. Both include:

```text
X-Wici-Protocol: remote-render-v1
X-Wici-Request-Id: 42
X-Wici-Frame-Width: 640
X-Wici-Frame-Height: 480
```

The capture ZIP has exactly `manifest.json`, `seed.jpg`, `preview.jpg`, and
`alpha.png`. The manifest contains `protocol`, `sessionId`, `requestId`,
`width`, `height`, and a role-to-member map. `seed.jpg` is the original-view RGB
seed, `preview.jpg` is the rendered RGB frame, and `alpha.png` is lossless
coverage. Alpha must never be JPEG encoded. Clients must validate member names,
expanded sizes, image dimensions, manifest identity, and protocol before use.

## Lifecycle and ownership

The session manager exclusively owns each engine handle after successful engine
creation. `DELETE /render/sessions/{sessionId}` returns 204 and prevents new
work immediately. Idle sessions are removed after `sessionIdleTtlSeconds`.
Deletion or expiry marks an active session pending but defers engine release
until its active-request reference count reaches zero. Release is idempotent.

Recently expired or deleted identifiers remain tombstoned and return 410. Once
the bounded tombstone window elapses they return 404. Activity updates when a
valid, monotonic request is admitted; polling an invalid request does not keep a
session alive. Clients should best-effort DELETE on shutdown and treat 204, 404,
and 410 as cleanup-complete outcomes.

## Errors

Errors use `application/json` and a stable envelope:

```json
{
  "protocol": "remote-render-v1",
  "error": {
    "code": "session_expired",
    "message": "session expired",
    "retryable": false
  }
}
```

Optional `details` contains machine-readable validation context. Clients branch
on `code`, not English `message`.

| HTTP | Boundary | Representative codes | Retryable |
| --- | --- | --- | --- |
| 400 | Malformed/unknown fields, protocol, source, camera, mode | `invalid_request`, `unsupported_protocol`, `invalid_camera`, `invalid_source` | No |
| 404 | Unknown route or session | `route_not_found`, `session_not_found` | No |
| 410 | Recently expired/deleted session | `session_expired` | No; create a session |
| 409 | Non-increasing or superseded request | `request_out_of_order`, `request_superseded` | Superseded only |
| 413 | Upload/body/frame pixel limit | `upload_too_large`, `body_too_large`, `frame_too_large` | No |
| 429 | Bounded session capacity | `session_capacity` | Yes, with backoff |
| 503 | Engine/token/render/output unavailable | `engine_unavailable`, `token_unavailable`, `render_unavailable`, `invalid_engine_output` | As indicated |

Unexpected server faults may use 500 `internal_error` and are retryable. No
error response silently changes source, camera convention, or render mode.

An oversized request is rejected before session or renderer ownership is
created. The server sends and flushes a non-retryable JSON 413 with
`Cache-Control: no-store` and `Connection: close`, then discards request bytes
for at most 500 ms. A complete, graceful HTTP response is guaranteed when the
declared body is no larger than the route limit plus 64 KiB and the whole body
arrives inside that window. This budget is derived from the active route limit;
the server never allocates the declared body size. For a larger, stalled, or
infinite sender, the same JSON response is attempted first, but the server stops
after the byte/time budget and closes the connection; TCP reset is therefore a
documented possibility outside the bounded guarantee. Clients must not retry
`body_too_large`.

## Compatibility policy

This version has no implicit aliases or field coercion. Additive server metadata
requires a protocol-version decision because v1 clients are strict. Any change
to camera meaning, capture members, lifecycle semantics, or required fields
requires a new protocol capability. A server may advertise multiple exact
capabilities during migration; clients choose one exact version and send it in
every create and frame request.
