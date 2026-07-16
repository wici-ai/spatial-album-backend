from __future__ import annotations

import pathlib
import re
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING_ENV_DEFAULTS = {
    "WICI_GSPLAT_TRAINING_ENABLED": "1",
    "WICI_GSPLAT_MAX_JOBS": "1",
    "WICI_GSPLAT_MAX_IMAGES": "64",
    "WICI_GSPLAT_MAX_UPLOAD_MB": "512",
    "WICI_GSPLAT_MAX_IMAGE_MB": "24",
}

REMOTE_ENV_DEFAULTS = {
    "WICI_REMOTE_RENDER_ENABLED": "1",
    "WICI_REMOTE_MAX_SESSIONS": "4",
    "WICI_REMOTE_SESSION_TTL_SECONDS": "120",
    "WICI_REMOTE_MAX_FRAME_PIXELS": "1638400",
    "WICI_REMOTE_MAX_UPLOAD_MB": "20",
    "WICI_REMOTE_INTERACTIVE_JPEG_QUALITY": "82",
    "WICI_REMOTE_CAPTURE_JPEG_QUALITY": "92",
}


class RemoteRenderDeployConfigTest(unittest.TestCase):
    def test_compose_and_launcher_expose_bounded_defaults(self) -> None:
        compose = yaml.safe_load((ROOT / "deploy/docker-compose.yml").read_text())
        orbit = compose["services"]["orbit"]
        environment = orbit["environment"]
        for name, default in {**REMOTE_ENV_DEFAULTS, **TRAINING_ENV_DEFAULTS}.items():
            self.assertIn(name, environment)
            self.assertEqual(f"${{{name}:-{default}}}", environment[name])
        self.assertIn("gsplat_training_worker.py", environment["WICI_GSPLAT_TRAIN_COMMAND"])
        self.assertEqual("${WICI_GSPLAT_JOB_ROOT:-/runtime/gsplat-training}", environment["WICI_GSPLAT_JOB_ROOT"])

        # Orbit stays private behind Caddy; only its container port is exposed.
        self.assertNotIn("ports", orbit)
        self.assertEqual(["8772"], orbit["expose"])

        launcher = (ROOT / "deploy/scripts/start-service.sh").read_text()
        for name, default in REMOTE_ENV_DEFAULTS.items():
            self.assertIn(f'export {name}="${{{name}:-{default}}}"', launcher)

        env_example = (ROOT / "deploy/.env.example").read_text().splitlines()
        for name, default in {**REMOTE_ENV_DEFAULTS, **TRAINING_ENV_DEFAULTS}.items():
            self.assertIn(f"{name}={default}", env_example)

        dockerfile = (ROOT / "deploy/Dockerfile.backend").read_text()
        self.assertRegex(dockerfile, r"(?m)^\s+colmap \\$")
        self.assertIn("gsplat-preload-training.patch", dockerfile)
        self.assertIn("patch -p1 -d /opt/sota/gsplat", dockerfile)
        self.assertIn("/opt/venvs/orbit/bin/python -m pip install --no-build-isolation", dockerfile)
        self.assertIn("/opt/sota/gsplat", dockerfile)

    def test_caddy_strips_orbit_prefix_for_remote_routes(self) -> None:
        caddy = (ROOT / "deploy/caddy/Caddyfile").read_text()
        block = re.search(
            r"handle_path\s+/orbit/\*\s*\{(?P<body>.*?)\}", caddy, re.DOTALL
        )
        self.assertIsNotNone(block)
        self.assertIn("reverse_proxy orbit:8772", block.group("body"))
        self.assertNotRegex(caddy, r"(?m)^\s*reverse_proxy\s+[^\n]*:54228")


if __name__ == "__main__":
    unittest.main()
