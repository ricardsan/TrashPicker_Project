"""Shared fixtures for the camera tests.

Everything is synthetic and CPU-only except the tests marked `slow`, which load a real checkpoint
and are the only ones that need the trained runs to be present. Deselect them with
`-m 'not slow'`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# A real unfetched pointer, copied byte-for-byte from one of the stubs in runs/segment/.
LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:8a0e3e6a4e62a9e5c1b7f1a0d9c5b4e2f3a6d8c0b1e4f7a9c2d5e8b1a4f7c0d3\n"
    "size 23845620\n"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: loads a real checkpoint; deselect with -m 'not slow'")


@pytest.fixture
def run_dir(tmp_path):
    """Factory for a synthetic run directory: args.yaml plus a plausible weights/best.pt."""

    def make(name="run", args=None, weights=b"\x80\x05" + b"\x00" * 4096):
        directory = tmp_path / name
        (directory / "weights").mkdir(parents=True)
        if args is not None:
            (directory / "args.yaml").write_text(yaml.safe_dump(args, sort_keys=False))
        if weights is not None:
            (directory / "weights" / "best.pt").write_bytes(weights)
        return directory

    return make


@pytest.fixture
def pointer_weights(tmp_path):
    """A best.pt that is an unfetched git LFS stub rather than weights."""
    path = tmp_path / "stub" / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    path.write_text(LFS_POINTER)
    return path
