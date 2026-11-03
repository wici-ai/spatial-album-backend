# Compact host integration

Bootstrap owns hardware selection. This backend reads optional native environment
settings at process startup; it does not identify GPU products or change scheduler
policy. Standalone Docker preflight and its existing hardware requirements remain
unchanged. No new checkpoint, HTTP API or CPU offload path is introduced.

| Setting | Default / original large host | Compact host setting | Effect |
| --- | --- | --- | --- |
| `SHARP_WEIGHT_DTYPE` | `fp32` | `fp16` | Actual CUDA weight dtype, separate from autocast precision; also accepts `bf16` |
| `MVP_DIFIX_MAX_SIDE` | Backend `1280`, Bootstrap `960` | `640` | Existing Difix working-image bound |
| `WICI_FLUX_QUANT_MODE` | `bnb8` | `bnb4` | Adds NF4 double quantization, BF16 computation and tiled/sliced VAE; existing `fp8` remains supported |
| `WICI_INPAINT_MAX_SIDE` | `0` (uncapped) | `640` | Maximum aligned FLUX working-image side; 0 or a positive multiple of 64 |
| `WICI_SPATIAL_TRIM_UNUSED_CACHE` | `0` | `1` | Orbit and Difix synchronize and release unused allocator cache after requests; weights stay loaded |

These are process environment inputs, not automatically loaded `.env` files.
Invalid added settings fail startup rather than silently selecting another mode.
The compact working image/mask uses 64-pixel tile alignment. Masks use nearest
resampling; generated content is restored to the padded source dimensions before
the existing feather compositor preserves unmasked source pixels. This reduces
detail and can affect fill quality. Clients retain the original output dimensions.

Host integration still executes dependent stages sequentially and uses the
existing request-control completion hooks. Releasing unused cache does not evict
weights or substitute for scheduler admission. To recover original behavior,
remove the added settings or use the large profile, then restart the three host
services through their normal lifecycle. Model files are unchanged.

See [the dated change record](changes/2026-11-03-compact-spatial.md) for measurements
and incomplete quality acceptance. Bootstrap must pin the reviewed backend commit;
existing v0.1.0 release assets do not contain this unreleased change.
