# Spatial Album Backend

Production local backend for form-1 spatial album users.

The runtime surface is intentionally small:

- `backend/orbit_preview_endpoint.py` with in-process fused SHARP
- `backend/sharp_endpoint.py`
- `backend/mvp_difix_endpoint.py`
- `backend/flux_endpoint.py`
- `deploy/` Docker install, Caddy proxy, GPU preflight, and Avahi/mDNS discovery

Third-party model code is tracked as submodules:

- `third_party/ml-sharp` pinned to `1eaa046834b81852261262b41b0919f5c1efdd2e`
- `third_party/Difix3D` pinned to `c76edc595586e16732c91ddee82f3a6d83a8a9cc`

FlashWorld and the old `sharp-libs` copied site-packages tree are not included.
The Docker image installs the needed Python packages from pip instead.

Clone with submodules:

```bash
git clone --recurse-submodules git@github.com:wici-ai/spatial-album-backend.git
```

Run the local install from `deploy/`:

```bash
cd spatial-album-backend/deploy
./install.sh
```
