"""Regression tests for the OpenCode integration source."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_opencode_policy_includes_temp_allow_paths() -> None:
    """OpenCode should match other integrations by allowing OS temp dirs."""
    source = Path("integrations/opencode/intaris.ts").read_text()

    assert 'globPatternsFor("/tmp")' in source
    assert 'globPatternsFor("/var/tmp")' in source
    assert "process.env.TMPDIR" in source
    assert "transient scratch" in source


def test_opencode_policy_resolves_macos_private_symlinks() -> None:
    """/tmp, /var/tmp and $TMPDIR are symlinks into /private/... on macOS —
    a literal "/tmp/*" pattern never matches an already-resolved real path.
    globPatternsFor must resolve each built-in temp dir with realpathSync
    and include the result too, or reads/writes under the resolved path
    (e.g. Claude Code's own tool scratchpad) get flagged as out-of-policy.
    """
    source = Path("integrations/opencode/intaris.ts").read_text()

    assert "realpathSync" in source
    assert "globPatternsFor(tmpDir)" in source


def test_opencode_denial_approval_hook() -> None:
    """Exercise the exported plugin hook with mocked HTTP and a fake tool executor."""
    node = shutil.which("node")
    if node is None:
        pytest.skip(
            "OpenCode hook regression requires Node.js >= 22.18 (built-in TS stripping)"
        )
    version = subprocess.run(
        [node, "-p", "process.versions.node"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    major, minor, *_ = (int(part) for part in version.split("."))
    if major < 22 or (major == 22 and minor < 18) or (major == 23 and minor < 6):
        pytest.skip(
            "OpenCode hook regression requires unflagged TS stripping (Node.js >= 22.18 LTS or >= 23.6)"
        )
    subprocess.run(
        [node, "--test", "integrations/opencode/denial-approval.test.mjs"],
        check=True,
        timeout=30,
    )
