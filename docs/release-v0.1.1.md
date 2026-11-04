# Spatial Album Backend v0.1.1

Adds optional compact host runtime controls for SHARP/Orbit, Difix and FLUX.
Bootstrap owns hardware-capacity selection; the backend reads explicit process
settings and retains its original large-host defaults when they are absent.

- SHARP supports actual FP16/BF16 weight placement separately from autocast.
- Difix accepts the host's working-image bound; the compact profile uses 640.
- FLUX supports NF4 double quantization and a 640-pixel aligned working image,
  with the generated fill restored to the original output dimensions.
- Orbit/Difix can release unused allocator cache after request completion while
  keeping model weights resident. Existing sequential admission and HTTP APIs
  remain unchanged; this does not add simultaneous request execution.

Validation: eight CPU contract tests passed, and all six backend modules parsed.
Existing real 16 GiB GPU evidence returned all three operations successfully;
all 287 approximately 100 ms samples contained three resident processes. Their
summed sampled peak was 14,284 MiB (14,977,040,384 bytes). Whole-board usage was
higher because of other processes; this sample is not a universal 15 GB bound.

Known limitation: the inspected NF4 fill produced unwanted cartoon content and
FAILED quality acceptance. Full native UI NF4 acceptance, a 5090 hardware
regression and clean-host installation were not run for publication. This release
makes the optional controls available; it does not establish their output quality.
Use the original large profile to recover original precision and working sizes.

As with v0.1.0, this is a source release with pinned submodule gitlinks and no
uploaded model or GPU runtime binaries. Initialize submodules when checking out
the tag. Bootstrap will pin this release's exact commit. Checkpoint families,
client protocol, scheduler implementation and standalone Docker GPU preflight
requirements remain unchanged. The native Bootstrap compact profile does not
mean the standalone Docker installer now accepts 16 GiB GPUs.

No machine IP, account path or GPU product-name matching is introduced. No Box
services are updated merely by publishing this release.
