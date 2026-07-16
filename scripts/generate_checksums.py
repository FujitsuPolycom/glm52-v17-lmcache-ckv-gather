#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"
EXCLUDED_PARTS = {".git", "__pycache__", "results"}
EXCLUDED_NAMES = {".env", "SHA256SUMS"}
BINARY_SUFFIXES = {".png"}


def source_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
        and path.suffix not in {".pyc", ".pyo"}
    )


def canonical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() not in BINARY_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def render() -> str:
    rows = []
    for path in source_files():
        digest = hashlib.sha256(canonical_bytes(path)).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != expected:
            print("SHA256SUMS is missing or stale")
            return 1
        print(f"Verified {len(source_files())} release files")
        return 0
    OUTPUT.write_text(expected, newline="\n")
    print(f"Wrote {OUTPUT} for {len(source_files())} release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
