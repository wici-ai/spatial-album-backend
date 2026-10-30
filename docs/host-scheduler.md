# Host scheduler deployment (v0.1.0)

The Docker installer and its hardware preflight remain unchanged. The backend
can also run under the WiCi environment bootstrap host runtime, which supplies
CUDA/Python dependencies, model paths and systemd units. The tested 5060 Ti
profile uses persistent workers and scheduler eviction rather than requiring
every model to fit in GPU memory simultaneously.

`backend/execution_control.py` is the optional host integration boundary.
Without `WICI_REQUEST_CONTROL_MODULE`, endpoint behavior is unchanged. To use
the Box scheduler, bootstrap sets `WICI_REQUEST_CONTROL_MODULE=request_control`
and provides that module on `PYTHONPATH`. The module must expose
`controlled_http_task(service, paths, *, finish)` returning a handler decorator.
An explicitly selected missing or invalid module fails startup, never silently
disabling admission. The host adapter owns request admission, cancellation and
completion; each endpoint supplies CUDA synchronization as its completion hook.

The service/path contract is:

| Service | Methods and paths |
| --- | --- |
| `wici-spatial-orbit.service` | GET `/splat`, `/splat_stream`, `/orbit_preview`; POST `/ingest`, `/ingest_preview` |
| `wici-spatial-difix.service` | POST `/refine`, `/api/difix-refine` |
| `wici-spatial-flux.service` | POST `/fill` |

Other routes, including health checks and static content, do not claim a GPU
request slot. The `X-WiCi-Trace-Id` header is propagated into completion metadata
and timing/error logs where supported; callers should supply a non-sensitive
request identifier.

Bootstrap checks out the release tag and its exact commit, then initializes
the submodule revisions recorded by that commit. It does not replace endpoint
files with local overlays. Models and user photos remain outside Git releases;
upgrades must retain existing backend photo directories. No models, tokens or
GPU runtimes are included in this source release.

Run the CPU adapter tests with `python3 -m unittest discover -s tests`.
Host GPU qualification additionally exercises SHARP/Orbit, Difix, FLUX and
switching back to Q&A against the actual deployment.
