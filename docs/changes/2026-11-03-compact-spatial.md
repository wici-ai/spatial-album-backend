# Optional compact Spatial runtime for host provisioning

- Owner: Pending: assign a GitHub username
- Date: 2026-11-03
- Issue: https://github.com/wici-ai/wici-environment-bootstrap/issues/39
- PR: https://github.com/wici-ai/spatial-album-backend/pull/2
- Status: draft

## Purpose

Keep SHARP/Orbit, Difix and FLUX resident on a physical 16 GiB GPU without
replacing their existing checkpoint families or changing the application protocol.

## Scope

Add optional actual SHARP weight precision, FLUX NF4 and bounded fill working
images; use completion hooks to trim unused Orbit/Difix allocator cache. Bootstrap
owns automatic capacity selection. Preserve backend defaults for existing large
hosts. No changes to the scheduler, WiCi One or model downloads.

## Impact

The compact profile trades fill fidelity and working resolution for memory.
Dependencies are the existing bitsandbytes, diffusers, Torch and Pillow packages.
No new public API or network identity configuration. See ../configuration.md.

## Verification

- CPU: `python3 -m unittest discover -s tests` passed 8 tests (fill sizing,
  validation, masks, output dimensions and existing request-control behavior).
- Real hardware, existing 2026-11-03 run: RTX 5060 Ti physical 16 GiB;
  SHARP FP16, Difix BF16/640, FLUX NF4/640, 20 steps, guidance 30, no CPU offload.
  Three HTTP requests returned successfully. In the final trimmed run all 287
  NVML samples at approximately 100 ms intervals contained all three resident
  processes. Their summed sampled peak was 14,284 MiB (14,977,040,384 bytes):
  Orbit 1,702 MiB, Difix 6,254 MiB, FLUX 6,328 MiB. Whole-board peak at that
  sample was 15,117.938 MiB, including other processes. This fixture has only
  about 23 MB margin against 15 decimal GB, not a universal peak guarantee.
- Request durations in that run: Orbit 2.82 s, FLUX 21.80 s, Difix 29.22 s.
  These overlap in wall time but include admission waits; they are not isolated
  kernel timings or a full WiCi One UI E2E measurement.
- Quality acceptance FAILED for the inspected NF4 background-fill fixture:
  unwanted cartoon content appeared. Full native UI NF4 acceptance and a real
  5090 regression are NOT RUN. Large behavior is checked by defaults/source and
  provisioning tests only. Do not describe this draft as fully qualified.

## Recovery

Select original settings and restart via the existing host lifecycle. Checkpoints
and output API remain compatible; no model migration is required.

## Follow-up

On 2026-11-04 the owner explicitly requested merging PR #2 into main using local
`--ff-only` integration and publishing v0.1.1 as a source release, matching
v0.1.0's format. Publication does not change the failed quality result or imply
full native UI/5090/clean-host acceptance. Owner/test-date assignment and fill
quality follow-up remain tracked by Issue 39. CPU checks were rerun locally:
8 tests passed and all six backend modules parsed. Substantive implementation,
tests and release preparation used Codex assistance. No live service deployment
or restart is part of publication.
