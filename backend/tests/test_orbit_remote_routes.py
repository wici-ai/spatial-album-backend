from __future__ import annotations

import http.client
import json
import pathlib
import socket
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import orbit_preview_endpoint as orbit  # noqa: E402
from backend.remote_render_service import (  # noqa: E402
    CAMERA_CONVENTION,
    PROTOCOL,
    Camera,
    Limits,
    RenderedFrame,
    SessionManager,
    SessionResource,
)
from backend.gsplat_training_service import PROTOCOL as TRAINING_PROTOCOL  # noqa: E402


JPEG = b"\xff\xd8orbit-fake\xff\xd9"


class FakeOrbitEngine:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.rendered: list[tuple[str, Camera, str]] = []
        self.released: list[str] = []

    def create_gallery(self, photo_id: str) -> SessionResource:
        handle = f"gallery:{photo_id}"
        self.created.append(handle)
        return SessionResource(
            handle=handle,
            initial_camera=Camera(
                CAMERA_CONVENTION,
                (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0),
                30.0,
                31.0,
                15.5,
                11.5,
                32,
                24,
            ),
            pivot=(0.0, 0.0, 1.0),
            focus_distance=1.0,
            source_width=640,
            source_height=480,
        )

    def create_upload(self, image: bytes, filename: str, content_type: str) -> SessionResource:
        return self.create_gallery(f"upload:{filename}:{content_type}:{len(image)}")

    def render(self, handle: str, camera: Camera, mode: str) -> RenderedFrame:
        self.rendered.append((handle, camera, mode))
        return RenderedFrame(JPEG, camera.width, camera.height)

    def release(self, handle: str) -> None:
        self.released.append(handle)


def frame_payload(request_id: int) -> dict:
    return {
        "protocol": PROTOCOL,
        "requestId": request_id,
        "mode": "interactive",
        "camera": {
            "convention": CAMERA_CONVENTION,
            "w2c": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0],
            "intrinsics": {"fx": 30.0, "fy": 31.0, "cx": 15.5, "cy": 11.5},
            "width": 32,
            "height": 24,
        },
    }


class OrbitRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_manager = orbit._remote_render_manager
        self.original_smi = orbit.smi_processes
        self.original_user_gallery = orbit.load_user_gallery
        self.original_build_gallery = orbit.build_gallery_records
        self.engine = FakeOrbitEngine()
        self.manager = SessionManager(
            self.engine,
            limits=Limits(max_sessions=2, max_frame_pixels=1024),
            token=lambda: "orbit_test_session",
        )
        orbit._remote_render_manager = self.manager
        orbit.smi_processes = lambda: []
        orbit.load_user_gallery = lambda: []
        orbit.build_gallery_records = lambda **_kwargs: []
        self.server = orbit.Server(("127.0.0.1", 0), orbit.Handler)
        self.rejections = []
        self.rejection_recorded = threading.Event()

        def observe(record) -> None:
            self.rejections.append(record)
            self.rejection_recorded.set()

        self.server.remote_body_rejection_observer = observe
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        orbit._remote_render_manager = self.original_manager
        orbit.smi_processes = self.original_smi
        orbit.load_user_gallery = self.original_user_gallery
        orbit.build_gallery_records = self.original_build_gallery

    def request(self, method: str, path: str, body: bytes | None = None, content_type: str | None = None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {"Content-Type": content_type, "Content-Length": str(len(body or b""))} if content_type else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        result = (response.status, dict(response.getheaders()), data)
        connection.close()
        return result

    def oversized_socket(self, declared: int) -> socket.socket:
        connection = socket.create_connection(self.server.server_address, timeout=2)
        connection.settimeout(2)
        connection.sendall((
            "POST /render/sessions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {declared}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii"))
        return connection

    def read_complete_response(self, connection: socket.socket) -> tuple[dict[str, str], bytes]:
        wire = bytearray()
        while b"\r\n\r\n" not in wire:
            chunk = connection.recv(4096)
            self.assertTrue(chunk)
            wire.extend(chunk)
        headers, body = bytes(wire).split(b"\r\n\r\n", 1)
        header_lines = headers.decode("iso-8859-1").split("\r\n")
        self.assertEqual("413", header_lines[0].split(" ", 2)[1])
        response_headers = {
            name.lower(): value.strip()
            for name, value in (line.split(":", 1) for line in header_lines[1:])
        }
        content_length = int(response_headers["content-length"])
        while len(body) < content_length:
            chunk = connection.recv(content_length - len(body))
            self.assertTrue(chunk)
            body += chunk
        self.assertEqual(content_length, len(body))
        self.assertEqual(("body_too_large", "close", "no-store"), (
            json.loads(body)["error"]["code"],
            response_headers["connection"],
            response_headers["cache-control"],
        ))
        return response_headers, body

    def test_fake_engine_flows_through_orbit_handler_without_model_load(self) -> None:
        create = json.dumps(
            {"protocol": PROTOCOL, "source": {"type": "gallery", "photoId": "curated-1"}}
        ).encode()
        status, _, body = self.request("POST", "/render/sessions", create, "application/json")
        self.assertEqual(201, status)
        session = json.loads(body)
        session_id = session["sessionId"]

        frame = json.dumps(frame_payload(1)).encode()
        status, headers, body = self.request(
            "POST", f"/render/sessions/{session_id}/frames", frame, "application/json"
        )
        self.assertEqual((200, "image/jpeg", JPEG), (status, headers["Content-Type"], body))
        self.assertEqual(1, len(self.engine.rendered))

        status, _, _ = self.request("DELETE", f"/render/sessions/{session_id}")
        self.assertEqual(204, status)
        self.assertEqual(["gallery:curated-1"], self.engine.released)
        self.assertNotIn("sharp_endpoint", sys.modules)

    def test_health_is_anonymous_and_legacy_gallery_route_remains(self) -> None:
        self.manager.create_gallery("private-photo")
        status, _, body = self.request("GET", "/health")
        self.assertEqual(200, status)
        health = json.loads(body)
        self.assertIn(PROTOCOL, health["capabilities"])
        self.assertIn(TRAINING_PROTOCOL, health["capabilities"])
        self.assertEqual((True, TRAINING_PROTOCOL, True), (
            health["gsplatTraining"]["enabled"],
            health["gsplatTraining"]["protocol"],
            health["gsplatTraining"]["preloadTrainingData"],
        ))
        self.assertEqual(1, health["remoteRender"]["sessions"]["activeSessions"])
        remote_json = json.dumps(health["remoteRender"])
        self.assertNotIn("orbit_test_session", remote_json)
        self.assertNotIn("private-photo", remote_json)

        status, _, body = self.request("GET", "/gallery")
        self.assertEqual((200, []), (status, json.loads(body)))

    def test_client_schema_error_is_not_mapped_to_500(self) -> None:
        create = json.dumps(
            {"protocol": PROTOCOL, "source": {"type": "gallery", "photoId": "curated-1"}}
        ).encode()
        _, _, body = self.request("POST", "/render/sessions", create, "application/json")
        session_id = json.loads(body)["sessionId"]
        invalid = frame_payload(1)
        invalid["camera"]["w2c"][0] = 2.0
        status, _, body = self.request(
            "POST",
            f"/render/sessions/{session_id}/frames",
            json.dumps(invalid).encode(),
            "application/json",
        )
        self.assertEqual(400, status)
        self.assertEqual("invalid_camera", json.loads(body)["error"]["code"])

    def test_oversized_body_rejection_matches_http11_orbit_handler(self) -> None:
        declared = 64 * 1024 + 4096
        body = b"x" * declared
        connection = self.oversized_socket(declared)
        with connection:
            connection.sendall(body)
            self.read_complete_response(connection)
        self.assertTrue(self.rejection_recorded.wait(1))
        self.assertEqual((declared, True, "complete"), (
            self.rejections[-1].discarded_bytes,
            self.rejections[-1].body_complete,
            self.rejections[-1].outcome,
        ))
        self.assertEqual([], self.engine.created)

    def test_oversized_body_paced_send_matches_http11_orbit_handler(self) -> None:
        declared = 64 * 1024 + 4096
        body = b"x" * declared
        first_chunk = 4096
        continue_sending = threading.Event()
        sender_errors = []
        connection = self.oversized_socket(declared)
        connection.sendall(body[:first_chunk])

        def send_remainder() -> None:
            continue_sending.wait(1)
            try:
                for offset in range(first_chunk, declared, 4096):
                    connection.sendall(body[offset : offset + 4096])
                    time.sleep(0.002)
            except Exception as exc:
                sender_errors.append(exc)

        sender = threading.Thread(target=send_remainder)
        sender.start()
        try:
            self.read_complete_response(connection)
            continue_sending.set()
            sender.join(1)
            self.assertEqual([], sender_errors)
            self.assertTrue(self.rejection_recorded.wait(1))
        finally:
            continue_sending.set()
            sender.join(1)
            connection.close()
        self.assertEqual((declared, "complete"), (
            self.rejections[-1].discarded_bytes, self.rejections[-1].outcome
        ))
        self.assertEqual([], self.engine.created)

    def test_oversized_body_headers_only_matches_http11_orbit_handler(self) -> None:
        declared = 64 * 1024 + 4096
        connection = self.oversized_socket(declared)
        started = time.monotonic()
        with connection:
            self.read_complete_response(connection)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(self.rejection_recorded.wait(1))
        self.assertEqual((0, False, "peer_closed"), (
            self.rejections[-1].discarded_bytes,
            self.rejections[-1].body_complete,
            self.rejections[-1].outcome,
        ))
        self.assertEqual([], self.engine.created)


class OrbitEngineMappingTest(unittest.TestCase):
    def test_render_passes_protocol_w2c_and_k_directly_to_rasterization(self) -> None:
        spec = orbit.AssetSpec(
            path=pathlib.Path("owned-test-asset"),
            kind="test",
            source_width=2,
            source_height=2,
            k_source=np.eye(3, dtype=np.float32),
            photo_id=None,
            title=None,
        )
        tensor = orbit.torch.zeros((1, 3), dtype=orbit.torch.float32)
        asset = orbit.LoadedAsset(spec, tensor, tensor, tensor, orbit.torch.zeros(1), tensor, 1, 0.0, 0.0)
        handle = orbit.RemoteAssetHandle(asset)
        camera = Camera(
            CAMERA_CONVENTION,
            (0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0),
            11.0,
            12.0,
            0.75,
            1.25,
            2,
            2,
        )
        captured: dict = {}

        def fake_rasterization(**kwargs):
            captured.update(kwargs)
            colors = orbit.torch.full((1, 2, 2, 3), 0.2)
            alpha = orbit.torch.full((1, 2, 2, 1), 0.5)
            return colors, alpha, {}

        original_tensor = orbit.torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return original_tensor(*args, **kwargs)

        with mock.patch.object(orbit, "rasterization", fake_rasterization), mock.patch.object(
            orbit.torch, "tensor", side_effect=cpu_tensor
        ):
            frame = orbit.OrbitRemoteRenderEngine().render(handle, camera, "capture")

        np.testing.assert_allclose(captured["viewmats"].numpy().reshape(-1), camera.w2c)
        np.testing.assert_allclose(
            captured["Ks"].numpy()[0],
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
        )
        self.assertEqual((2, 2), (captured["width"], captured["height"]))
        self.assertTrue(frame.preview_jpeg.startswith(b"\xff\xd8"))
        self.assertTrue((frame.seed_jpeg or b"").startswith(b"\xff\xd8"))
        self.assertTrue((frame.alpha_png or b"").startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
