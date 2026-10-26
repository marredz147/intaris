"""Static read grants retain legacy latency without treating exemptions as effects."""

import pytest

from intaris import read_only
from intaris.classifier import (
    _READ_ONLY_TOOLS,
    is_read_only,
    observe_hybrid,
    verified_read_only,
    verified_read_only_bash,
)
from tests.test_hybrid_races import race_store as race_store

EXAMPLES = [
    ("get_current_datetime", {"timezone": "UTC"}),
    ("skill_load", {"skill_id": "cognis-coding"}),
    ("skill", {"name": "architecture-fit"}),
    (
        "compress",
        {
            "topic": "Closed phase",
            "content": [{"startId": "@27@", "endId": "@56@", "summary": "Done."}],
        },
    ),
    ("task_status", {"sessionID": "ses_child"}),
    ("task_result", {"sessionID": "ses_child"}),
    ("question", {"questions": [{"question": "Which parser?", "options": []}]}),
    ("wait_for_user", {"reason": "Waiting for review."}),
    ("serena_read_memory", {"memory_file_name": "overview"}),
    ("read", {"file_path": "/repo/README.md"}),
    ("grep", {"path": "/repo", "pattern": "test"}),
    ("list_directory", {"path": "/repo"}),
    ("bash", {"command": "git rev-parse HEAD origin/main"}),
    ("bash", {"command": "git status --short; git rev-parse HEAD origin/main"}),
]


@pytest.mark.parametrize("operand", ["HEAD:.env", "HEAD:private.env", ".env"])
def test_git_sensitive_blob_and_path_denial(operand):
    command = f"git show --no-ext-diff --no-textconv {operand}"
    facts = observe_hybrid(
        "bash",
        {"command": command},
        policy={"deny_paths": ["*.env", "**/*.env"]},
        working_directory="/repo",
    )
    assert not facts["static_read"]
    assert facts["configured_deny"]
    assert facts["secret_argument"]


def test_existing_configured_git_drivers_are_trusted(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key in ("diff.external", "diff.private.textconv"):
        subprocess.run(
            ["git", "-C", str(tmp_path), "config", key, "unexpected-helper"], check=True
        )
    assert verified_read_only_bash({"command": "git diff"})
    assert verified_read_only_bash({"command": "git show HEAD:file"})
    assert verified_read_only_bash({"command": "git diff --no-ext-diff --no-textconv"})


@pytest.mark.parametrize("profile", ["balanced", "bounded_autonomy"])
@pytest.mark.parametrize("tool,args", EXAMPLES)
def test_production_reads_never_infer(race_store, profile, tool, args):
    sessions, audit, llm, evaluator, sid = race_store
    sessions.update_session(sid, user_id="race-owner", policy={"profile": profile})
    llm.generate.side_effect = AssertionError("Static read must not infer")
    result = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool=tool,
        args=args,
        context={"chat_mode": "plan"},
    )
    assert result["decision"] == "approve"
    assert (
        result["evaluation_metadata"]["policy_result"]["permission_basis"]
        == "read_only_static"
    )
    assert not result["evaluation_metadata"]["judge_required"]
    assert not result["evaluation_metadata"]["stages"]
    llm.generate.assert_not_called()
    assert (
        audit.get_by_call_id(result["call_id"], user_id="race-owner")["decision"]
        == "approve"
    )


@pytest.mark.parametrize("tool", sorted(_READ_ONLY_TOOLS))
def test_legacy_local_tool_table(race_store, tool):
    sessions, _, llm, evaluator, sid = race_store
    assert is_read_only(tool, {})
    result = evaluator.evaluate(
        user_id="race-owner", session_id=sid, agent_id=None, tool=tool, args={}
    )
    if tool != "todowrite":
        assert result["decision"] == "approve"
        assert (
            result["evaluation_metadata"]["policy_result"]["permission_basis"]
            == "read_only_static"
        )
        llm.generate.assert_not_called()
    else:
        # Legacy labels a metadata mutation as READ; that is not effect proof.
        assert (
            result["evaluation_metadata"]["policy_result"]["permission_basis"]
            != "read_only_static"
        )


@pytest.mark.parametrize(
    "guard", ["strict", "deny", "advisory", "sensitive", "human", "floor"]
)
def test_fast_path_guards(race_store, guard):
    sessions, audit, llm, evaluator, sid = race_store
    policy = {
        "strict": {"profile": "strict"},
        "deny": {"deny_tools": ["read"]},
        "advisory": {"deny_policies": ["Do not read private project material."]},
    }.get(guard, {})
    if policy:
        sessions.update_session(sid, user_id="race-owner", policy=policy)
    if guard == "advisory":
        llm.generate.side_effect = AssertionError("Advisory text never routes")
    args = {"file_path": "/repo/.env" if guard == "sensitive" else "/repo/README.md"}
    kwargs = dict(
        user_id="race-owner", session_id=sid, agent_id=None, tool="read", args=args
    )
    if guard == "human":
        pending = evaluator.evaluate(**kwargs, minimum_outcome="escalate")
        audit.resolve_escalation(pending["call_id"], "deny", user_id="race-owner")
    result = evaluator.evaluate(
        **kwargs, **({"minimum_outcome": "deny"} if guard == "floor" else {})
    )
    if guard in {"deny", "human", "floor"}:
        assert result["decision"] == "deny"
    elif guard == "advisory":
        assert result["decision"] == "approve"
        assert (
            result["evaluation_metadata"]["policy_result"]["permission_basis"]
            == "read_only_static"
        )
    else:
        assert (
            result["evaluation_metadata"]["policy_result"]["permission_basis"]
            != "read_only_static"
        )


@pytest.mark.parametrize("tool", ["Skill", "SKILL"])
def test_only_opencode_skill_loader_is_a_static_read(tool):
    assert is_read_only("skill", {})
    assert verified_read_only("skill", {})
    assert read_only.observe("skill", {}) == "read_only"
    # Claude Code "Skill" can run shell preprocessing: still assessed.
    assert not is_read_only(tool, {})
    assert not verified_read_only(tool, {})
    assert not observe_hybrid(tool, {}, policy={}, working_directory="/repo")[
        "static_read"
    ]


@pytest.mark.parametrize("guard", ["strict", "deny"])
def test_skill_loader_keeps_policy_guards(race_store, guard):
    sessions, _, _, evaluator, sid = race_store
    policy = {"strict": {"profile": "strict"}, "deny": {"deny_tools": ["skill"]}}
    sessions.update_session(sid, user_id="race-owner", policy=policy[guard])
    result = evaluator.evaluate(
        user_id="race-owner",
        session_id=sid,
        agent_id=None,
        tool="skill",
        args={"name": "architecture-fit"},
    )
    if guard == "deny":
        assert result["decision"] == "deny"
    assert (
        result["evaluation_metadata"]["policy_result"]["permission_basis"]
        != "read_only_static"
    )


@pytest.mark.parametrize(
    "args",
    [
        {"path": "/repo/build", "output": "/repo/build.tgz"},
        {"topic": "x", "content": [{"startId": "@1@", "path": "/etc/passwd"}]},
        {"topic": "x", "content": "rm -rf /repo"},
        {"topic": "x", "content": [], "command": "tar czf /tmp/x.tgz /repo"},
    ],
)
def test_only_the_context_compress_shape_is_a_static_read(args):
    assert not verified_read_only("compress", args)
    assert not verified_read_only(
        "Compress",
        {"topic": "x", "content": [{"startId": "@1@", "endId": "@2@", "summary": "s"}]},
    )


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("task_status", {"sessionID": "ses_child", "message": "continue"}),
        ("task_result", {"sessionID": ["ses_child"]}),
        ("question", {"questions": "Deploy now?", "command": "deploy"}),
        ("wait_for_user", {}),
        ("Task_Status", {"sessionID": "ses_child"}),
        ("task_message", {"sessionID": "ses_child"}),
    ],
)
def test_only_exact_agent_coordination_shapes_are_static_reads(tool, args):
    assert not verified_read_only(tool, args)
