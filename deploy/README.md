# Spatial Album form-1 local backend deploy

This directory contains the Docker-based one-click local backend install for
form-1 users. It has been build and run validated on the WiCi Box RTX 5090 with
orbit, difix, flux, Caddy, and host-side mDNS discovery running together.

Target:

- Linux native with an NVIDIA Blackwell GPU, driver 570+, CUDA reported as 12.8,
  compute capability 12.0, and at least 24 GiB VRAM.
- Windows through Docker Desktop with WSL2 GPU passthrough. Run `install.sh`
  inside WSL after Docker Desktop WSL integration is enabled.

Install:

```bash
curl -fsSL https://raw.githubusercontent.com/wici-ai/spatial-album-backend/main/deploy/install.sh | bash
```

The one-line installer clones or updates
`${WICI_INSTALL_DIR:-$HOME/spatial-album-backend}`, initializes submodules,
reruns the local `deploy/install.sh`, then performs the normal GPU preflight,
Docker setup, model download, compose startup, and health-gated mDNS install.

Manual clone alternative:

```bash
git clone --recurse-submodules https://github.com/wici-ai/spatial-album-backend.git
cd spatial-album-backend/deploy
./install.sh
```

Draft files:

- `Dockerfile.backend` builds one CUDA 12.8 image with three venvs:
  `/opt/venvs/orbit`, `/opt/venvs/difix`, and `/opt/venvs/flux`.
- `docker-compose.yml` launches `orbit`, `difix`, `flux`, and `caddy`.
- `caddy/Caddyfile` exposes the same `/orbit`, `/difix`, and `/flux` surface on
  local port `54228`.
- `gpu-preflight.sh` checks `nvidia-smi`, driver/CUDA, VRAM, compute capability,
  and at least 80 GiB free disk.
- `spatial-preflight.sh` performs a read-only installed-state check for the
  backend image, Compose/Caddy inputs, NVIDIA runtime, model artifacts, and the
  backend port. It is safe for the profile control manager to call before it
  stops another GPU workload.
- `install.sh` runs preflight, checks or installs Docker/toolkit, builds the
  image, downloads weights, starts compose, installs host-side health-gated
  Avahi discovery, and prints the LAN URL.
- `avahi/health-advertise.sh` and `avahi/wici-mdns-health.service` install a
  host-side mDNS gate. The gate advertises `_wici-backend._tcp` on port `54228`
  only while `http://127.0.0.1:54228/orbit/health` returns `ok`; it removes the
  Avahi service file when orbit is down so Android NSD does not discover a dead
  Caddy route.
- `scripts/download_weights.py` downloads SHARP, Difix, FLUX, and FLUX text
  encoder weights into the model volume.
- `scripts/start-service.sh` applies the env flags observed on vast and starts
  the selected endpoint.

Remote interactive rendering uses the existing Caddy listener: public
`/orbit/render/...` requests are prefix-stripped to Orbit's internal
`/render/...` routes. No additional host port is opened. Android local mode
continues to download and render the splat; remote mode creates a backend
session and receives interactive JPEG frames or a capture package without
downloading the complete splat.

Remote-session defaults (also listed in `.env.example`) are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `WICI_REMOTE_RENDER_ENABLED` | `1` | Advertise and serve `remote-render-v1`; set `0` to disable. |
| `WICI_REMOTE_MAX_SESSIONS` | `4` | Maximum simultaneous GPU-resident sessions. |
| `WICI_REMOTE_SESSION_TTL_SECONDS` | `120` | Idle time before server reclamation. |
| `WICI_REMOTE_MAX_FRAME_PIXELS` | `1638400` | Maximum requested width times height. |
| `WICI_REMOTE_MAX_UPLOAD_MB` | `20` | Maximum uploaded source image body. |
| `WICI_REMOTE_INTERACTIVE_JPEG_QUALITY` | `82` | Interactive preview JPEG quality. |
| `WICI_REMOTE_CAPTURE_JPEG_QUALITY` | `92` | Capture preview and seed JPEG quality; alpha remains PNG. |

Clients should explicitly `DELETE /orbit/render/sessions/<session-id>` when a
view closes. The TTL is a bounded fallback for disconnected clients. Raising
session or frame limits can exhaust VRAM; raising upload or quality limits can
increase memory, bandwidth, and latency. The `/orbit/health` response reports
the exact protocol capability, configured public limits, and anonymous session
counts, but never session IDs or photo paths.

The first-stage tests use a fake engine on loopback and require no CUDA device.
They validate protocol, routing, lifecycle, and configuration only. GPU image
correctness, interaction latency, real VRAM ceilings, and concurrency remain a
separate integration gate and have not been run for this change. The Caddy
listener and mDNS discovery target a trusted LAN; they are not a complete
public authentication, authorization, TLS, or multi-tenant isolation solution.

The Docker image copies model code from git submodules:

- `../third_party/ml-sharp`
- `../third_party/Difix3D`

FlashWorld is not included because the production form-1 backend does not load
it. The old `sharp-libs` copied site-packages tree is not included; the needed
packages are installed from pinned pip requirements in `Dockerfile.backend`.

Weight volume layout:

```text
$WICI_MODEL_DIR/
  torch/hub/checkpoints/sharp_2572gikvuh.pt
  huggingface/hub/models--nvidia--difix_ref/
  ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors
  ComfyUI/models/vae/ae.safetensors
  ComfyUI/models/text_encoders/clip_l.safetensors
  ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors
  flux/flux_peripheral/hf_flux1_fill_dev_config/
  flux/flux_peripheral/prompt_embeds.pt
  difix/source_photo_original.jpg
```

Known draft risks:

- `black-forest-labs/FLUX.1-Fill-dev` is gated and requires an HF token for an
  account that accepted the model terms. The Box validation reused existing
  weights; the end-to-end `download_weights.py` gated FLUX download path with a
  fresh token still needs validation.
- The live SHARP endpoint currently hard-checks compute capability `(12, 0)`.
  A 24 GiB non-Blackwell GPU is not enough without code changes.
- The Docker image currently has about a 32 GiB local Docker footprint. It could
  be slimmed with a multi-stage build that keeps build tools and transient
  compile artifacts out of the runtime image.
- The hard disk preflight minimum is 80 GiB: about 32 GiB for the Docker image,
  about 35 GiB for model weights, plus margin for build cache and partial
  downloads.
- mDNS discovery requires `avahi-daemon` on Linux and a LAN that permits
  multicast DNS. Guest Wi-Fi, client/AP isolation, and some WSL2 networking
  modes can prevent Android NSD from seeing the service.
- The mDNS advertiser is deliberately health-gated on orbit rather than always
  on. If orbit is deliberately stopped with `docker compose stop orbit`, the
  `_wici-backend._tcp` record is withdrawn; when orbit is started and healthy
  again, the record reappears. To reinstall only this host-side gate without
  rebuilding images or starting GPU services, run `deploy/install.sh --mdns-only`.
- Windows Docker Desktop with WSL2 needs separate mDNS advertisement handling
  because Avahi inside WSL2 is not always visible on the physical LAN.
- Running `avahi-browse` on the backend host can also show Docker bridge/veth
  interfaces. LAN clients should resolve the instance on the physical LAN
  interface, for example `192.168.x.x:54228`.
