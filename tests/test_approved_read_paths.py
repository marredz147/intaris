"""Approved out-of-project read prefixes fast-path later reads in both
policy versions; credential reads and uncovered paths stay assessed."""

from tests.test_approved_workspace_edit import edit, env, user_message

env = env  # fixture re-export


def read(evaluator, sid, path):
    return edit(evaluator, sid, None, tool="read", args={"file_path": path})


def test_approved_outside_read_opens_prefix(env):
    _, audit, llm, evaluator, sid, version = env
    first = read(evaluator, sid, "/srv/other/src/a.py")
    assert first["decision"] == "approve"
    assert llm.generate.call_count == 1

    second = read(evaluator, sid, "/srv/other/src/b.py")
    assert llm.generate.call_count == 1
    assert second["decision"] == "approve"
    assert second["path"] == "fast"
    assert "path approved via prior evaluation" in second["reasoning"]
    if version == "hybrid_v1":
        record = audit.get_by_call_id(second["call_id"], user_id=sid)
        metadata = record["evaluation_metadata"]
        assert metadata["policy_result"]["permission_basis"] == "approved_read_path"
        assert metadata["approved_read_prefixes"] == [
            "/srv"
        ]  # sibling prefix of WD /repo
        assert not metadata["stages"]

    # Uncovered paths and credential reads still go to the model.
    read(evaluator, sid, "/opt/elsewhere/c.py")
    assert llm.generate.call_count == 2
    read(evaluator, sid, "/srv/other/src/.env")
    assert llm.generate.call_count == 3
    # The prefix outlives a new user message, as in legacy.
    user_message(audit, sid, "keep going")
    read(evaluator, sid, "/srv/other/src/d.py")
    assert llm.generate.call_count == 3


def test_escalated_outside_read_opens_no_prefix(env):
    _, _, llm, evaluator, sid, _ = env
    from tests.test_approved_workspace_edit import _answer

    llm.generate.side_effect = _answer("escalate")
    assert read(evaluator, sid, "/srv/other/a.py")["decision"] != "approve"
    calls = llm.generate.call_count
    read(evaluator, sid, "/srv/other/b.py")
    assert llm.generate.call_count > calls


def test_deny_paths_win_over_approved_prefix(env):
    sessions, _, llm, evaluator, sid, _ = env
    read(evaluator, sid, "/srv/other/a.py")
    policy = {"deny_paths": ["/srv/other/secret*"]}
    if env[5] == "hybrid_v1":
        policy["profile"] = "balanced"
    sessions.update_session(sid, user_id=sid, policy=policy)
    assert read(evaluator, sid, "/srv/other/secret.txt")["decision"] == "deny"
