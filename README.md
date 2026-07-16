# Spatial Album Backend

Self-hostable GPU backend for the
[Spatial Album Android app](https://github.com/wici-ai/spatial-album). It turns
a device photo into a 3D Gaussian-splatting scene with SHARP, renders Orbit
previews, refines reframed views with Difix, and fills peripheral content with
FLUX. The production runtime is Docker-based and exposes one LAN backend for the
app to discover.

This backend is for non-commercial / research use. See [Models And Licenses](#models-and-licenses).

## Quick Install

Run this on the local GPU box:

```bash
curl -fsSL https://raw.githubusercontent.com/wici-ai/spatial-album-backend/main/deploy/install.sh | bash
```

The installer self-clones or updates
`${WICI_INSTALL_DIR:-$HOME/spatial-album-backend}` with submodules, runs GPU and
disk preflight, checks Docker and the NVIDIA container runtime, prompts for a
Hugging Face token if needed, downloads model weights, starts Docker Compose,
and installs the host-side mDNS advertiser. The Android app on the same LAN can
then auto-discover the backend.

Manual alternative:

```bash
git clone --recurse-submodules https://github.com/wici-ai/spatial-album-backend.git
cd spatial-album-backend/deploy
./install.sh
```

## Requirements

These are enforced by `deploy/gpu-preflight.sh` and the deploy scripts:

- NVIDIA GPU with at least `24576 MiB` VRAM.
- Blackwell compute capability `12.0` / `sm_120`; the current SHARP endpoint
  expects RTX 5090-class hardware.
- NVIDIA driver major version `570` or newer.
- `nvidia-smi` reporting CUDA `12.8` or newer.
- At least `80 GiB` free disk: about `32 GiB` Docker image footprint, about
  `35 GiB` model weights, plus build/download margin. `100+ GiB` is safer.
- Docker with Compose v2 and working `--gpus all` support. On native
  Debian/Ubuntu Linux, `install.sh` can prompt to install Docker Engine and the
  NVIDIA Container Toolkit. On Windows, use Docker Desktop with WSL2 integration
  and GPU passthrough enabled.

## What Runs

`deploy/docker-compose.yml` starts four long-running services with
`restart: unless-stopped`:

- `orbit` on container port `8772`: `backend/orbit_preview_endpoint.py`, with
  `backend/sharp_endpoint.py` loaded in-process for fused SHARP ingest.
- `difix` on container port `8765`: `backend/mvp_difix_endpoint.py`.
- `flux` on container port `8771`: `backend/flux_endpoint.py`.
- `caddy` on host port `${WICI_BACKEND_PORT:-54228}`: reverse proxy and static
  file server.

Caddy exposes the app-facing surface on port `54228` by default:

```text
http://<box-lan-ip>:54228/orbit/health
http://<box-lan-ip>:54228/difix/health
http://<box-lan-ip>:54228/flux/health
```

The proxy maps `/orbit/*` to Orbit, `/difix/*` to Difix, and `/flux/*` to
FLUX. Caddy also keeps compatibility routes such as `/gallery`, `/source`,
`/thumbnail`, `/splat`, `/splat_stream`, and `/orbit_preview` pointed at Orbit.

Orbit also implements versioned remote interactive rendering. The app-facing
routes are `/orbit/render/sessions` and
`/orbit/render/sessions/<session-id>/frames`; Caddy strips `/orbit` before
forwarding them to Orbit. This uses the existing Caddy port and does not expose
Orbit's container port directly. Local Android rendering continues to use the
downloaded splat path, while remote rendering keeps the splat on the backend.
See [`docs/remote-render-v1.md`](docs/remote-render-v1.md) for the wire contract.

Orbit can also advertise `wici.gsplat-training.v1`. The Android app uploads an
explicitly confirmed multi-view selection to `/orbit/training/jobs`, polls the
bounded job, downloads the resulting `.splat`, and deletes the terminal job.
The pinned gsplat trainer preloads the training tensors to CUDA once and crops
batches on-device, avoiding per-step image H2D RPC traffic. See
[`docs/gsplat-training-v1.md`](docs/gsplat-training-v1.md) for the wire,
privacy, resource-limit, and step-scaling contract.

## Auto-Discovery

The deploy installs a host-side Avahi/systemd advertiser for:

```text
_wici-backend._tcp on port 54228
```

The advertisement is health-gated. It is published only while
`http://127.0.0.1:54228/orbit/health` returns OK, and is withdrawn when Orbit is
down so Android NSD does not discover a dead backend. mDNS requires
`avahi-daemon` on Linux and a LAN that allows multicast DNS; guest Wi-Fi,
client isolation, and some WSL2 networking modes can block discovery.

To reinstall only the host-side advertiser:

```bash
cd ~/spatial-album-backend/deploy
./install.sh --mdns-only
```

## Configuration

The main knobs are shown in `deploy/.env.example`:

```text
WICI_MODEL_DIR=/home/you/.cache/wici-form1/models
WICI_RUNTIME_DIR=/home/you/.cache/wici-form1/runtime
WICI_BACKEND_PORT=54228
WICI_REQUIRED_DISK_GB=80
COMPOSE_PROJECT_NAME=wici-form1
WICI_ENABLE_MDNS=1
WICI_MDNS_SERVICE_TYPE=_wici-backend._tcp
WICI_MDNS_VERSION=1
WICI_MDNS_BASE_PATH=/
WICI_REMOTE_RENDER_ENABLED=1
WICI_REMOTE_MAX_SESSIONS=4
WICI_REMOTE_SESSION_TTL_SECONDS=120
WICI_REMOTE_MAX_FRAME_PIXELS=1638400
WICI_REMOTE_MAX_UPLOAD_MB=20
WICI_REMOTE_INTERACTIVE_JPEG_QUALITY=82
WICI_REMOTE_CAPTURE_JPEG_QUALITY=92
```

`install.sh` writes `deploy/.env` for Compose from the selected model/runtime
directories and port. You can export the same variables before running the
installer to override defaults.

Remote sessions are deliberately bounded. `WICI_REMOTE_MAX_SESSIONS` limits
concurrent GPU-resident scenes, `WICI_REMOTE_SESSION_TTL_SECONDS` reclaims idle
sessions, and `WICI_REMOTE_MAX_FRAME_PIXELS` caps each requested frame. Clients
should still send `DELETE /orbit/render/sessions/<session-id>` when finished;
TTL is crash recovery, not the normal close path. Uploads and JPEG qualities
are bounded by the remaining settings above. Increasing session count, pixel
count, upload size, or quality can increase VRAM, host memory, bandwidth, and
latency; measure changes on the target GPU rather than treating these defaults
as a capacity claim. Set `WICI_REMOTE_RENDER_ENABLED=0` to omit the capability
and reject remote session routes while retaining local-render support.

The remote protocol and lifecycle tests run without a GPU using a fake engine
bound only to loopback. That CPU validation has passed during development, but
real GPU image correctness, latency, VRAM limits, and concurrency have not yet
been integration-tested. This deployment is a trusted-LAN backend, not a
completed public-internet authentication or tenant-isolation design. Do not
publish its port to an untrusted network without adding an appropriate auth and
transport-security boundary.

FLUX.1-Fill-dev is gated on Hugging Face. The installer accepts either:

- `HF_TOKEN` in the environment, or
- a token stored at `${WICI_MODEL_DIR}/hf-token`.

If neither exists, `install.sh` prompts for a token and writes it to
`${WICI_MODEL_DIR}/hf-token` with a restrictive umask. Do not commit or print
real tokens.

Model weights are downloaded into `${WICI_MODEL_DIR}`. The expected layout is
documented in `deploy/README.md`.

## Models And Licenses

This repository wires together model code and gated weights with restrictive
terms:

- `third_party/ml-sharp` is the Apple SHARP model code submodule. Its
  `LICENSE_MODEL` limits model use to non-commercial research purposes.
- `third_party/Difix3D` is the NVIDIA Difix3D submodule. Its license limits the
  work and derivatives to non-commercial research/evaluation use.
- `black-forest-labs/FLUX.1-Fill-dev` is downloaded from Hugging Face and
  requires an account/token that has accepted the FLUX non-commercial terms.
- `nvidia/difix_ref`, `comfyanonymous/flux_text_encoders`, and the Apple SHARP
  checkpoint are downloaded by `deploy/scripts/download_weights.py`.

Treat this backend as non-commercial / research unless you have independently
confirmed and obtained every license needed for your use case.

## Repo Layout

```text
backend/
  orbit_preview_endpoint.py   # Orbit render, ingest, gallery/source helpers
  gsplat_training_service.py  # bounded multi-view job and HTTP protocol
  gsplat_training_worker.py   # COLMAP, CUDA-preloaded gsplat, viewer export
  sharp_endpoint.py           # SHARP module used in-process by Orbit
  mvp_difix_endpoint.py       # Difix refinement endpoint
  flux_endpoint.py            # FLUX.1-Fill endpoint
  assets/                     # small runtime fixtures

deploy/
  Dockerfile.backend          # CUDA 12.8 image with orbit/difix/flux venvs
  docker-compose.yml          # orbit, difix, flux, caddy, downloader
  install.sh                  # one-line bootstrap + installer
  gpu-preflight.sh            # GPU/CUDA/driver/disk checks
  caddy/Caddyfile             # reverse proxy on :54228
  avahi/                      # health-gated mDNS service
  scripts/                    # service launcher and weight downloader

third_party/
  ml-sharp/                   # git submodule
  Difix3D/                    # git submodule
```
