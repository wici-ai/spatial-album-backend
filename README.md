# Spatial Album Backend

Self-hostable GPU backend for the
[Spatial Album Android app](https://github.com/wici-ai/spatial-album). It turns
a device photo into a 3D Gaussian-splatting scene with SHARP, renders Orbit
previews, refines reframed views with Difix, and fills peripheral content with
FLUX. The production runtime is Docker-based and exposes one LAN backend for the
app to discover.

This backend is for non-commercial / research use. See [Models And Licenses](#models-and-licenses).

## Released host integration

Version **v0.1.0** includes optional request admission and trace propagation
for the WiCi Box host runtime. See [host scheduler deployment](docs/host-scheduler.md).
Standalone Docker deployments do not require a scheduler adapter.

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
```

`install.sh` writes `deploy/.env` for Compose from the selected model/runtime
directories and port. You can export the same variables before running the
installer to override defaults.

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
