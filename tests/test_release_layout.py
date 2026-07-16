import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTests(unittest.TestCase):
    def test_manifest_is_gpu_validated_but_experimental(self) -> None:
        manifest = json.loads((ROOT / "release-manifest.json").read_text())
        self.assertTrue(manifest["published"])
        self.assertEqual(manifest["repository"]["visibility"], "public")
        self.assertTrue(manifest["repository"]["code_pushed"])
        self.assertEqual(
            manifest["status"], "gpu-validated-experimental"
        )
        self.assertFalse(manifest["validation"]["production_ready"])
        self.assertEqual(
            manifest["validation"]["258048_token_chunked_store"],
            "passed-three-cold-runs",
        )
        self.assertEqual(manifest["runtime"]["max_model_len"], 262_144)

    def test_pinned_build_and_required_patch(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("@sha256:9b6f1ab6", dockerfile)
        self.assertIn("lmcache==0.4.6", dockerfile)
        self.assertIn("patch_chunked_store_futures.py", dockerfile)
        self.assertIn("INDEXER_BASE_SHA=27848bf7", dockerfile)
        self.assertIn("sparse_attn_indexer.py", dockerfile)
        self.assertIn("validate_lmcache.py", dockerfile)
        self.assertIn("validate_ckv_gather.py", dockerfile)

        b12x = (
            ROOT
            / "gather/overlay/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"
        ).read_text()
        indexer = (
            ROOT / "gather/overlay/vllm/model_executor/layers/sparse_attn_indexer.py"
        ).read_text()
        self.assertIn("_dcp_all_gather_current_stream", b12x)
        self.assertIn("pynccl_comm.all_gather", b12x)
        self.assertIn("active_page_width", indexer)
        self.assertNotIn("[DEBUG-CKV-NEXT]", indexer)

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

    def test_review_patch_and_boundary_validator_are_present(self) -> None:
        patch = (ROOT / "patches/v17-ckv-gather.patch").read_text()
        for path in (
            "vllm/v1/attention/backends/mla/b12x_mla_sparse.py",
            "vllm/model_executor/layers/attention/mla_attention.py",
            "vllm/model_executor/layers/sparse_attn_indexer.py",
        ):
            self.assertIn(f"--- a/{path}", patch)
            self.assertIn(f"+++ b/{path}", patch)
        self.assertIn("_dcp_all_gather_current_stream", patch)
        self.assertIn("active_page_width", patch)

        validator = (ROOT / "scripts/validate-boundary-followups.sh").read_text()
        self.assertIn('prompt_tokens="${PROMPT_TOKENS:-258048}"', validator)
        self.assertIn('followups="${FOLLOWUPS:-4}"', validator)

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
