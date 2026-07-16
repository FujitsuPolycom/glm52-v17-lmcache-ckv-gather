import runpy
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = runpy.run_path(str(ROOT / "scripts/generate_checksums.py"))
canonical_bytes = CHECKSUMS["canonical_bytes"]


class ChecksumTests(unittest.TestCase):
    def test_text_checksums_are_independent_of_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lf = root / "lf.txt"
            crlf = root / "crlf.txt"
            lf.write_bytes(b"alpha\nbeta\n")
            crlf.write_bytes(b"alpha\r\nbeta\r\n")

            self.assertEqual(canonical_bytes(lf), canonical_bytes(crlf))

    def test_binary_files_are_not_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sample.png"
            payload = b"fake\r\npng\r\nbytes"
            image.write_bytes(payload)

            self.assertEqual(canonical_bytes(image), payload)


if __name__ == "__main__":
    unittest.main()
