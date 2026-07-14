from __future__ import annotations

import pathlib
import re
import unittest

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
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
        for name, default in REMOTE_ENV_DEFAULTS.items():
            self.assertIn(name, environment)
            self.assertEqual(f"${{{name}:-{default}}}", environment[name])

        # Orbit stays private behind Caddy; only its container port is exposed.
        self.assertNotIn("ports", orbit)
        self.assertEqual(["8772"], orbit["expose"])

        launcher = (ROOT / "deploy/scripts/start-service.sh").read_text()
        for name, default in REMOTE_ENV_DEFAULTS.items():
            self.assertIn(f'export {name}="${{{name}:-{default}}}"', launcher)

        env_example = (ROOT / "deploy/.env.example").read_text().splitlines()
        for name, default in REMOTE_ENV_DEFAULTS.items():
            self.assertIn(f"{name}={default}", env_example)

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
