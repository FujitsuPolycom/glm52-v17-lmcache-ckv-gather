import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "base/patches"
PYPI_JSON = "https://pypi.org/pypi/lmcache/0.4.6/json"
ADAPTER_SUFFIX = "lmcache/integration/vllm/vllm_multi_process_adapter.py"


@unittest.skipUnless(
    os.environ.get("RUN_LMCACHE_SOURCE_TEST") == "1",
    "set RUN_LMCACHE_SOURCE_TEST=1 to fetch and patch pristine LMCache 0.4.6",
)
class LmcacheSourcePatchTests(unittest.TestCase):
    def test_pristine_046_source_accepts_patch_chain(self) -> None:
        with urllib.request.urlopen(PYPI_JSON, timeout=30) as response:
            metadata = json.load(response)
        source = next(item for item in metadata["urls"] if item["packagetype"] == "sdist")
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            archive = response.read()
        self.assertEqual(hashlib.sha256(archive).hexdigest(), source["digests"]["sha256"])

        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            member = next(
                item
                for item in bundle.getmembers()
                if item.name.endswith(ADAPTER_SUFFIX) and item.isfile()
            )
            extracted = bundle.extractfile(member)
            if extracted is None:
                self.fail(f"could not extract {member.name}")
            adapter_source = extracted.read()

        with tempfile.TemporaryDirectory() as temporary:
            adapter = Path(temporary) / "vllm_multi_process_adapter.py"
            adapter.write_bytes(adapter_source)
            environment = {**os.environ, "LMCACHE_ADAPTER_PATH": str(adapter)}
            for script in (
                "patch_ipc_event_lifetime.py",
                "patch_chunked_store_futures.py",
            ):
                subprocess.run(
                    [sys.executable, str(PATCHES / script)],
                    check=True,
                    env=environment,
                    capture_output=True,
                    text=True,
                )

            patched = adapter.read_text()
            compile(patched, str(adapter), "exec")
            self.assertIn("AI01-LMCACHE-IPC-EVENT-LIFETIME", patched)
            self.assertIn("AI01-LMCACHE-CHUNKED-STORE-FUTURES", patched)
            self.assertIn(
                "self.store_futures.setdefault(request_id, []).append(future)",
                patched,
            )
            self.assertIn(
                "self.store_events.setdefault(request_id, []).append(event)",
                patched,
            )

            # Both patchers must be idempotent on the exact source they emit.
            first_digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
            for script in (
                "patch_ipc_event_lifetime.py",
                "patch_chunked_store_futures.py",
            ):
                subprocess.run(
                    [sys.executable, str(PATCHES / script)],
                    check=True,
                    env=environment,
                    capture_output=True,
                    text=True,
                )
            self.assertEqual(
                hashlib.sha256(adapter.read_bytes()).hexdigest(), first_digest
            )


if __name__ == "__main__":
    unittest.main()

