from __future__ import annotations

import io
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.remote_render_service import (  # noqa: E402
    CAMERA_CONVENTION,
    CAPTURE_MEMBERS,
    PROTOCOL,
    Camera,
    Limits,
    ProtocolError,
    RenderedFrame,
    SessionManager,
    SessionResource,
    BodyRejectionRecord,
    loopback_server,
    parse_camera,
    parse_frame_request,
    parse_gallery_create,
)


JPEG = b"\xff\xd8fake-jpeg\xff\xd9"
PNG = b"\x89PNG\r\n\x1a\nfake-lossless-alpha"


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def camera_payload(*, width: int = 32, height: int = 24, w2c: list[float] | None = None) -> dict:
    return {
        "convention": CAMERA_CONVENTION,
        "w2c": w2c
        or [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0],
        "intrinsics": {"fx": 30.0, "fy": 31.0, "cx": 15.5, "cy": 11.5},
        "width": width,
        "height": height,
    }


def frame_payload(request_id: int, *, mode: str = "interactive", camera: dict | None = None) -> dict:
    return {"protocol": PROTOCOL, "requestId": request_id, "mode": mode, "camera": camera or camera_payload()}


class FakeEngine:
    def __init__(self) -> None:
        self.created: list[tuple] = []
        self.rendered: list[tuple] = []
        self.released: list[str] = []
        self.render_started: threading.Event | None = None
        self.render_continue: threading.Event | None = None
        self.fail_create = False

    def _resource(self, handle: str) -> SessionResource:
        return SessionResource(
            handle=handle,
            initial_camera=parse_camera(camera_payload(), Limits()),
            pivot=(0.25, -0.5, 1.5),
            focus_distance=2.5,
            source_width=640,
            source_height=480,
        )

    def create_gallery(self, photo_id: str) -> SessionResource:
        if self.fail_create:
            raise RuntimeError("offline")
        handle = f"gallery:{photo_id}:{len(self.created)}"
        self.created.append(("gallery", photo_id, handle))
        return self._resource(handle)

    def create_upload(self, image: bytes, filename: str, content_type: str) -> SessionResource:
        handle = f"upload:{len(self.created)}"
        self.created.append(("upload", image, filename, content_type, handle))
        return self._resource(handle)

    def render(self, handle: str, camera: Camera, mode: str) -> RenderedFrame:
        self.rendered.append((handle, camera, mode))
        if self.render_started is not None:
            self.render_started.set()
        if self.render_continue is not None:
            self.render_continue.wait(2)
        return RenderedFrame(JPEG, camera.width, camera.height, JPEG if mode == "capture" else None, PNG if mode == "capture" else None)

    def release(self, handle: str) -> None:
        self.released.append(handle)


class ManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.engine = FakeEngine()
        tokens = iter(f"session_{index:02d}" for index in range(20))
        self.manager = SessionManager(
            self.engine,
            limits=Limits(max_sessions=2, idle_ttl_seconds=10, tombstone_ttl_seconds=20, max_frame_pixels=1024),
            clock=self.clock,
            token=lambda: next(tokens),
        )

    def create(self, photo_id: str = "photo-1") -> dict:
        return self.manager.create_gallery(photo_id)

    def test_gallery_and_upload_create_metadata(self) -> None:
        gallery = self.create()
        upload = self.manager.create_upload(b"pixels", "../unsafe/name.png", "image/png")
        self.assertEqual(PROTOCOL, gallery["protocol"])
        self.assertEqual(CAMERA_CONVENTION, gallery["camera"]["convention"])
        self.assertEqual([0.25, -0.5, 1.5], gallery["pivot"])
        self.assertEqual("name.png", self.engine.created[1][2])
        self.assertNotEqual(gallery["sessionId"], upload["sessionId"])

    def test_capacity_is_429_and_engine_failure_is_503(self) -> None:
        self.create("one")
        self.create("two")
        with self.assertRaises(ProtocolError) as caught:
            self.create("three")
        self.assertEqual(429, caught.exception.status)
        self.assertTrue(caught.exception.retryable)
        other = SessionManager(self.engine, clock=self.clock, token=lambda: "valid_token")
        self.engine.fail_create = True
        with self.assertRaises(ProtocolError) as unavailable:
            other.create_gallery("one")
        self.assertEqual(503, unavailable.exception.status)

    def test_token_failure_releases_new_engine_resource(self) -> None:
        manager = SessionManager(self.engine, clock=self.clock, token=lambda: "bad/token")
        with self.assertRaises(ProtocolError) as caught:
            manager.create_gallery("one")
        self.assertEqual((503, "token_unavailable"), (caught.exception.status, caught.exception.code))
        self.assertEqual([self.engine.created[-1][-1]], self.engine.released)

    def test_expiry_tombstone_recovery_and_explicit_delete(self) -> None:
        first = self.create()
        first_handle = self.engine.created[-1][-1]
        self.clock.advance(10)
        replacement = self.create("replacement")
        self.assertIn(first_handle, self.engine.released)
        with self.assertRaises(ProtocolError) as expired:
            self.manager.render(first["sessionId"], frame_payload(1))
        self.assertEqual(410, expired.exception.status)
        self.manager.delete(replacement["sessionId"])
        with self.assertRaises(ProtocolError) as gone:
            self.manager.delete(replacement["sessionId"])
        self.assertEqual(410, gone.exception.status)
        self.clock.advance(21)
        self.manager.sweep()
        with self.assertRaises(ProtocolError) as forgotten:
            self.manager.delete(first["sessionId"])
        self.assertEqual(404, forgotten.exception.status)

    def test_request_ids_are_monotonic(self) -> None:
        session_id = self.create()["sessionId"]
        mime, body, headers = self.manager.render(session_id, frame_payload(4))
        self.assertEqual("image/jpeg", mime)
        self.assertEqual(JPEG, body)
        self.assertEqual("4", headers["X-Wici-Request-Id"])
        with self.assertRaises(ProtocolError) as caught:
            self.manager.render(session_id, frame_payload(4))
        self.assertEqual((409, "request_out_of_order"), (caught.exception.status, caught.exception.code))

    def test_superseded_waiter_does_not_call_engine(self) -> None:
        render_lock = threading.Lock()
        manager = SessionManager(self.engine, clock=self.clock, token=lambda: "latest_token", render_lock=render_lock)
        session_id = manager.create_gallery("one")["sessionId"]
        render_lock.acquire()
        errors: list[ProtocolError] = []

        def invoke(request_id: int) -> None:
            try:
                manager.render(session_id, frame_payload(request_id))
            except ProtocolError as error:
                errors.append(error)

        old = threading.Thread(target=invoke, args=(1,))
        new = threading.Thread(target=invoke, args=(2,))
        old.start()
        new.start()
        deadline = time.monotonic() + 1
        while manager._sessions[session_id].active_requests != 2:  # bounded white-box synchronization
            if time.monotonic() >= deadline:
                self.fail("requests did not reach the render lock")
        render_lock.release()
        old.join(2)
        new.join(2)
        self.assertEqual(1, len(self.engine.rendered))
        self.assertEqual("request_superseded", errors[0].code)

    def test_delete_does_not_release_inflight_resource(self) -> None:
        session_id = self.create()["sessionId"]
        handle = self.engine.created[-1][-1]
        self.engine.render_started = threading.Event()
        self.engine.render_continue = threading.Event()
        outcomes: list[object] = []
        thread = threading.Thread(target=lambda: outcomes.append(self.manager.render(session_id, frame_payload(1))))
        thread.start()
        self.assertTrue(self.engine.render_started.wait(1))
        self.manager.delete(session_id)
        self.assertNotIn(handle, self.engine.released)
        self.engine.render_continue.set()
        thread.join(2)
        self.assertIn(handle, self.engine.released)
        self.assertEqual(1, len(outcomes))

    def test_capture_zip_is_fixed_and_alpha_is_png(self) -> None:
        session_id = self.create()["sessionId"]
        mime, body, headers = self.manager.render(session_id, frame_payload(7, mode="capture"))
        self.assertEqual("application/zip", mime)
        self.assertEqual("7", headers["X-Wici-Request-Id"])
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            self.assertEqual(set(CAPTURE_MEMBERS), set(archive.namelist()))
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual((PROTOCOL, session_id, 7), (manifest["protocol"], manifest["sessionId"], manifest["requestId"]))
            self.assertTrue(archive.read("alpha.png").startswith(b"\x89PNG"))


class ParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = Limits(max_frame_pixels=1024)

    def test_strict_create_and_frame_fields(self) -> None:
        self.assertEqual("p", parse_gallery_create({"protocol": PROTOCOL, "source": {"type": "gallery", "photoId": "p"}}))
        with self.assertRaises(ProtocolError):
            parse_gallery_create({"protocol": PROTOCOL, "source": {"type": "gallery", "photoId": "p"}, "extra": 1})
        with self.assertRaises(ProtocolError):
            parse_frame_request({**frame_payload(1), "extra": True}, self.limits)

    def test_rejects_non_rigid_reflection_and_bad_intrinsics(self) -> None:
        scaled = camera_payload(w2c=[2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        reflected = camera_payload(w2c=[-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        outside = camera_payload()
        outside["intrinsics"]["cx"] = 99.0
        for value in (scaled, reflected, outside):
            with self.subTest(value=value), self.assertRaises(ProtocolError) as caught:
                parse_camera(value, self.limits)
            self.assertEqual(400, caught.exception.status)

    def test_pixel_limit_is_413(self) -> None:
        with self.assertRaises(ProtocolError) as caught:
            parse_camera(camera_payload(width=33, height=32), self.limits)
        self.assertEqual(413, caught.exception.status)

    def test_error_schema_is_stable(self) -> None:
        error = ProtocolError(503, "render_unavailable", "no renderer", retryable=True, details={"phase": "render"})
        self.assertEqual(
            {"protocol": PROTOCOL, "error": {"code": "render_unavailable", "message": "no renderer", "retryable": True, "details": {"phase": "render"}}},
            error.payload(),
        )

    def test_module_import_does_not_load_gpu_stack(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import backend.remote_render_service; assert 'torch' not in sys.modules and 'gsplat' not in sys.modules"],
            cwd=ROOT,
            env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stderr)


class HttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        self.manager = SessionManager(self.engine, token=lambda: "http_token_01", limits=Limits(max_upload_bytes=1024))
        self.server = loopback_server(self.manager)
        self.rejections: list[BodyRejectionRecord] = []
        self.rejection_recorded = threading.Event()

        def observe(record: BodyRejectionRecord) -> None:
            self.rejections.append(record)
            self.rejection_recorded.set()

        self.server.remote_body_rejection_observer = observe
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.assertEqual("127.0.0.1", host)
        self.base = f"http://127.0.0.1:{port}"
        # Never allow host proxy configuration to move a fake-engine test off
        # loopback or synthesize proxy body-size responses.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, path: str, *, method: str = "POST", body: bytes = b"", content_type: str = "application/json"):
        request = urllib.request.Request(self.base + path, data=body, method=method, headers={"Content-Type": content_type})
        try:
            with self.opener.open(request, timeout=2) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.headers, error.read()

    def create(self) -> str:
        body = json.dumps({"protocol": PROTOCOL, "source": {"type": "gallery", "photoId": "photo-1"}}).encode()
        status, headers, data = self.request("/render/sessions", body=body)
        self.assertEqual((201, "application/json"), (status, headers.get_content_type()))
        return json.loads(data)["sessionId"]

    def oversized_socket(self, declared_bytes: int) -> socket.socket:
        connection = socket.create_connection(self.server.server_address, timeout=2)
        connection.settimeout(2)
        headers = (
            "POST /render/sessions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {declared_bytes}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        connection.sendall(headers)
        return connection

    def read_complete_response(self, connection: socket.socket) -> tuple[int, dict[str, str], bytes]:
        wire = bytearray()
        while b"\r\n\r\n" not in wire:
            chunk = connection.recv(4096)
            self.assertTrue(chunk, "connection closed before response headers completed")
            wire.extend(chunk)
        raw_headers, body = bytes(wire).split(b"\r\n\r\n", 1)
        lines = raw_headers.decode("iso-8859-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        headers = {name.lower(): value.strip() for name, value in (line.split(":", 1) for line in lines[1:])}
        content_length = int(headers["content-length"])
        while len(body) < content_length:
            chunk = connection.recv(content_length - len(body))
            self.assertTrue(chunk, "connection closed before response body completed")
            body += chunk
        self.assertEqual(content_length, len(body))
        return status, headers, body

    def assert_oversized_response(self, response: tuple[int, dict[str, str], bytes]) -> None:
        status, headers, body = response
        payload = json.loads(body)
        self.assertEqual((413, "application/json", "no-store", "close"), (
            status,
            headers["content-type"],
            headers["cache-control"],
            headers["connection"],
        ))
        self.assertEqual(("body_too_large", False), (
            payload["error"]["code"], payload["error"]["retryable"]
        ))
        self.assertEqual([], self.engine.created)

    def test_oversized_body_full_send_returns_complete_413(self) -> None:
        body = b"x" * (64 * 1024 + 4096)
        connection = self.oversized_socket(len(body))
        with connection:
            connection.sendall(body)
            self.assert_oversized_response(self.read_complete_response(connection))
        self.assertTrue(self.rejection_recorded.wait(1))
        record = self.rejections[-1]
        self.assertEqual((len(body), len(body), "complete"), (
            record.discarded_bytes, record.discard_budget_bytes, record.outcome
        ))
        self.assertTrue(record.body_complete)

    def test_oversized_body_paced_send_returns_complete_413(self) -> None:
        body = b"x" * (64 * 1024 + 4096)
        first_chunk = 4096
        continue_sending = threading.Event()
        sender_errors: list[Exception] = []
        connection = self.oversized_socket(len(body))
        connection.sendall(body[:first_chunk])

        def send_remainder() -> None:
            continue_sending.wait(1)
            try:
                for offset in range(first_chunk, len(body), 4096):
                    connection.sendall(body[offset : offset + 4096])
                    time.sleep(0.002)
            except Exception as exc:  # surfaced by the test thread below
                sender_errors.append(exc)

        sender = threading.Thread(target=send_remainder)
        sender.start()
        try:
            # The complete 413 must be readable before the paced sender is
            # released to finish the declared request body.
            self.assert_oversized_response(self.read_complete_response(connection))
            continue_sending.set()
            sender.join(1)
            self.assertFalse(sender.is_alive())
            self.assertEqual([], sender_errors)
            self.assertTrue(self.rejection_recorded.wait(1))
        finally:
            continue_sending.set()
            sender.join(1)
            connection.close()
        record = self.rejections[-1]
        self.assertEqual((len(body), "complete"), (record.discarded_bytes, record.outcome))
        self.assertLessEqual(record.elapsed_seconds, 0.75)

    def test_oversized_body_headers_only_does_not_wait_unbounded(self) -> None:
        declared = 64 * 1024 + 4096
        connection = self.oversized_socket(declared)
        started = time.monotonic()
        with connection:
            self.assert_oversized_response(self.read_complete_response(connection))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(self.rejection_recorded.wait(1))
        record = self.rejections[-1]
        self.assertEqual((0, False, "peer_closed"), (
            record.discarded_bytes, record.body_complete, record.outcome
        ))
        self.assertLessEqual(record.elapsed_seconds, 0.75)

    def test_oversized_body_discard_is_byte_bounded(self) -> None:
        declared = 10 * 1024 * 1024
        discard_budget = 128 * 1024
        connection = self.oversized_socket(declared)
        connection.sendall(b"x" * (discard_budget + 4096))
        with connection:
            self.assert_oversized_response(self.read_complete_response(connection))
            self.assertTrue(self.rejection_recorded.wait(1))
        record = self.rejections[-1]
        self.assertEqual((discard_budget, discard_budget, "budget_exhausted"), (
            record.discarded_bytes, record.discard_budget_bytes, record.outcome
        ))
        self.assertFalse(record.body_complete)

    def test_create_frame_headers_delete_and_structured_error(self) -> None:
        session_id = self.create()
        status, headers, data = self.request(
            f"/render/sessions/{session_id}/frames", body=json.dumps(frame_payload(3)).encode()
        )
        self.assertEqual((200, "image/jpeg", "3", JPEG), (status, headers.get_content_type(), headers["X-Wici-Request-Id"], data))
        status, _, data = self.request(f"/render/sessions/{session_id}", method="DELETE")
        self.assertEqual((204, b""), (status, data))
        status, headers, data = self.request(
            f"/render/sessions/{session_id}/frames", body=json.dumps(frame_payload(4)).encode()
        )
        payload = json.loads(data)
        self.assertEqual((410, "application/json", PROTOCOL, "session_expired", False), (
            status, headers.get_content_type(), payload["protocol"], payload["error"]["code"], payload["error"]["retryable"]
        ))

    def test_multipart_upload_and_unknown_field_rejection(self) -> None:
        boundary = "wici-boundary"
        def multipart(extra: bool = False) -> bytes:
            parts = []
            for name, value in (("protocol", PROTOCOL), ("sourceType", "image")):
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + JPEG + b"\r\n")
            if extra:
                parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"path\"\r\n\r\n/etc/passwd\r\n".encode())
            parts.append(f"--{boundary}--\r\n".encode())
            return b"".join(parts)
        status, _, _ = self.request("/render/sessions", body=multipart(), content_type=f"multipart/form-data; boundary={boundary}")
        self.assertEqual(201, status)
        self.assertEqual(("upload", JPEG, "x.jpg", "image/jpeg"), self.engine.created[-1][:-1])
        status, _, data = self.request("/render/sessions", body=multipart(True), content_type=f"multipart/form-data; boundary={boundary}")
        self.assertEqual((400, "invalid_multipart"), (status, json.loads(data)["error"]["code"]))

    def test_bad_json_and_body_limit(self) -> None:
        status, _, data = self.request("/render/sessions", body=b"{")
        self.assertEqual((400, "invalid_json"), (status, json.loads(data)["error"]["code"]))
        status, _, data = self.request("/render/sessions", body=b"x" * 70000)
        self.assertEqual((413, "body_too_large"), (status, json.loads(data)["error"]["code"]))


if __name__ == "__main__":
    unittest.main()
