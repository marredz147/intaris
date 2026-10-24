import os
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).parents[1] / ".github" / "workflows" / "docker-publish.yml"
).read_text()


def test_ghcr_publication_workflow_contract():
    workflow = WORKFLOW
    assert "branches: [main]" in workflow
    assert "types: [published]" in workflow
    assert "!github.event.release.prerelease" in workflow
    assert "contents: read" in workflow
    assert "packages: write" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "timeout-minutes: 60" in workflow
    assert 'echo "tag=latest"' in workflow
    assert r"^v[0-9]+\.[0-9]+\.[0-9]+$" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert "linux/amd64,linux/arm64" in workflow
    assert (
        "org.opencontainers.image.source=https://github.com/${{ github.repository }}"
        in workflow
    )
    assert "docker/login-action@v3" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "github.event.release.tag_name" in workflow.split("run: |", 1)[0]


def run_tag_selection(tmp_path, event_name, release_tag="", ancestor=True):
    script = textwrap.dedent(
        WORKFLOW.split("        run: |\n", 1)[1].split("\n\n      - uses:", 1)[0]
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nif [ "$1" = "merge-base" ]; then\n  [ "$ANCESTOR" = "true" ]\nfi\n'
    )
    fake_git.chmod(0o755)
    output = tmp_path / "output"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "EVENT_NAME": event_name,
        "RELEASE_TAG": release_tag,
        "COMMIT_SHA": "commit-sha",
        "ANCESTOR": str(ancestor).lower(),
    }
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output.read_text() if output.exists() else ""


def test_main_push_selects_latest(tmp_path):
    result, output = run_tag_selection(tmp_path, "push")
    assert result.returncode == 0, result.stderr
    assert output == "tag=latest\n"


def test_stable_release_selects_version_tag(tmp_path):
    result, output = run_tag_selection(tmp_path, "release", "v1.2.3")
    assert result.returncode == 0, result.stderr
    assert output == "tag=v1.2.3\n"


@pytest.mark.parametrize("tag", ["hermes-v1.2.3", "openclaw-v1.2.3", "not-a-version"])
def test_integration_and_malformed_release_tags_are_rejected(tmp_path, tag):
    result, output = run_tag_selection(tmp_path, "release", tag)
    assert result.returncode != 0
    assert output == ""


def test_off_main_release_is_rejected(tmp_path):
    result, output = run_tag_selection(tmp_path, "release", "v1.2.3", ancestor=False)
    assert result.returncode != 0
    assert output == ""


def test_prereleases_are_excluded_before_tag_selection():
    assert "!github.event.release.prerelease" in WORKFLOW
