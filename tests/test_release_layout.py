import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTests(unittest.TestCase):
    def test_manifest_is_explicitly_unpublished(self) -> None:
        manifest = json.loads((ROOT / "release-manifest.json").read_text())
        self.assertFalse(manifest["published"])
        self.assertEqual(
            manifest["status"], "local-staging-gpu-validation-pending"
        )
        self.assertFalse(manifest["validation"]["production_ready"])
        self.assertEqual(manifest["runtime"]["max_model_len"], 262_144)

    def test_pinned_build_and_required_patch(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("@sha256:9b6f1ab6", dockerfile)
        self.assertIn("lmcache==0.4.6", dockerfile)
        self.assertIn("patch_chunked_store_futures.py", dockerfile)
        self.assertIn("validate_lmcache.py", dockerfile)
        self.assertIn("validate_ckv_gather.py", dockerfile)

    def test_compose_uses_configurable_host_paths(self) -> None:
        compose = (ROOT / "docker-compose.yml").read_text()
        for variable in (
            "MODEL_PATH",
            "HF_CACHE_PATH",
            "RUNTIME_CACHE_PATH",
            "RUNTIME_TMP_PATH",
        ):
            self.assertIn(f"${{{variable}:?", compose)
        self.assertNotIn("/srv/ai", compose)
        self.assertNotIn("192.168.", compose)

    def test_no_credentials_or_windows_user_paths(self) -> None:
        forbidden = (
            "webster" + "dog",
            "gho" + "_",
            "C:" + "\\" + "Users" + "\\",
        )
        extensions = {".md", ".py", ".sh", ".yml", ".yaml", ".json", ".example"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            text = path.read_text(errors="replace")
            for needle in forbidden:
                self.assertNotIn(needle, text, f"{needle!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
