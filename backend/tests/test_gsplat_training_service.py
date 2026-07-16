from __future__ import annotations

import http.client
import json
import pathlib
import tempfile
import threading
import time
import unittest

from backend.gsplat_training_service import (
    PROTOCOL,
    JobManager,
    Limits,
    TrainingError,
    make_handler,
    parse_multipart,
)


class FakeRunner:
    def run(self, workspace, manifest, progress, cancelled):
        progress(8_000, 30_000, "training")
        if cancelled.is_set():
            raise TrainingError(409, "job_cancelled", "cancelled")
        progress(30_000, 30_000, "exporting")
        output = workspace / "output"
        output.mkdir()
        result = output / "result.splat"
        result.write_bytes(b"splat-result")
        return result


def manifest(selected=("image_a", "image_b"), *, preload=True):
    return {
        "protocol": PROTOCOL,
        "sceneId": "living-room",
        "anchorCandidateId": selected[0],
        "selectedCandidateIds": list(selected),
        "options": {
            "preloadTrainingData": preload,
            "preloadChunkBytes": 768 * 1024,
            "batchSize": 8,
            "equivalentSteps": 30_000,
        },
    }


def multipart(value=None, images=("image_a", "image_b")):
    boundary = "wici-gsplat-boundary"
    chunks = []

    def field(name, data, headers=""):
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"{headers}\r\n\r\n'.encode(),
            data if isinstance(data, bytes) else str(data).encode(),
            b"\r\n",
        ])

    field("protocol", PROTOCOL)
    field("manifest", json.dumps(value or manifest()))
    for candidate in images:
        field(
            "image",
            b"\xff\xd8fake-jpeg\xff\xd9",
            f'; filename="{candidate}.jpg"\r\nContent-Type: image/jpeg\r\nX-Wici-Candidate-Id: {candidate}',
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


class ProtocolTest(unittest.TestCase):
    def test_manifest_and_upload_set_are_strict_and_preload_is_required(self):
        content_type, body = multipart()
        upload = parse_multipart(content_type, body, Limits())
        self.assertEqual(["image_a", "image_b"], [image.candidate_id for image in upload.images])
        self.assertEqual((True, 768 * 1024, 8), (
            upload.manifest["options"]["preloadTrainingData"],
            upload.manifest["options"]["preloadChunkBytes"],
            upload.manifest["options"]["batchSize"],
        ))
        with self.assertRaisesRegex(TrainingError, "exactly match"):
            content_type, body = multipart(images=("image_a", "image_c"))
            parse_multipart(content_type, body, Limits())
        with self.assertRaisesRegex(TrainingError, "must be true"):
            content_type, body = multipart(manifest(preload=False))
            parse_multipart(content_type, body, Limits())


class RouteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = JobManager(
            pathlib.Path(self.temp.name), FakeRunner(), limits=Limits(max_jobs=1), token=lambda: "training_job_01"
        )
        self.server = http.server = __import__("http.server", fromlist=["ThreadingHTTPServer"]).ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.manager)
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.temp.cleanup()

    def request(self, method, path, body=None, content_type=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read()
        result = response.status, dict(response.getheaders()), data
        connection.close()
        return result

    def test_create_poll_download_and_delete(self):
        content_type, body = multipart()
        status, _, data = self.request("POST", "/training/jobs", body, content_type)
        self.assertEqual(202, status)
        created = json.loads(data)
        self.assertEqual((PROTOCOL, True, 2), (
            created["protocol"], created["preloadTrainingData"], created["imageCount"]
        ))
        job_id = created["jobId"]
        deadline = time.time() + 2
        while time.time() < deadline:
            status, _, data = self.request("GET", f"/training/jobs/{job_id}")
            payload = json.loads(data)
            if payload["state"] == "succeeded":
                break
            time.sleep(0.01)
        self.assertEqual((200, "succeeded", 1.0, 30_000), (
            status, payload["state"], payload["progress"], payload["totalSteps"]
        ))
        status, headers, data = self.request("GET", f"/training/jobs/{job_id}/result")
        self.assertEqual((200, "splat-result"), (status, data.decode()))
        self.assertIn("result.splat", headers["Content-Disposition"])
        status, _, data = self.request("DELETE", f"/training/jobs/{job_id}")
        self.assertEqual((204, b""), (status, data))
        status, _, data = self.request("GET", f"/training/jobs/{job_id}")
        self.assertEqual((404, "job_not_found"), (status, json.loads(data)["error"]["code"]))

    def test_disabled_manager_rejects_without_persisting_images(self):
        disabled = JobManager(pathlib.Path(self.temp.name) / "disabled", None, token=lambda: "disabled_job_01")
        content_type, body = multipart()
        upload = parse_multipart(content_type, body, disabled.limits)
        with self.assertRaisesRegex(TrainingError, "not configured"):
            disabled.create(upload)
        self.assertEqual([], list((pathlib.Path(self.temp.name) / "disabled").iterdir()))


if __name__ == "__main__":
    unittest.main()
