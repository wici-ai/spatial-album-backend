from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.remote_render_service import Limits, ProtocolError, parse_camera  # noqa: E402


class CameraFixtureTest(unittest.TestCase):
    def test_canonical_vectors_match_parser(self) -> None:
        path = pathlib.Path(__file__).parent / "fixtures" / "remote-render-v1-camera-vectors.json"
        fixture = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual("remote-render-v1", fixture["protocol"])
        for vector in fixture["vectors"]:
            payload = {key: vector[key] for key in ("w2c", "intrinsics", "width", "height")}
            payload["convention"] = fixture["convention"]
            if vector["valid"]:
                camera = parse_camera(payload, Limits())
                self.assertEqual((vector["width"], vector["height"]), (camera.width, camera.height), vector["name"])
            else:
                with self.assertRaises(ProtocolError, msg=vector["name"]) as caught:
                    parse_camera(payload, Limits())
                self.assertEqual(vector["errorCode"], caught.exception.code, vector["name"])


if __name__ == "__main__":
    unittest.main()
