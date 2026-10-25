"""Judge response schema must satisfy strict structured outputs.

OpenAI rejected the strict Judge schema with HTTP 400 (param=response_format)
because optional properties were not listed in ``required`` and the schema
carried ``default``/``maxLength``. Every Judge review then failed closed with
503 review_unavailable.
"""

import pytest

from intaris import judge_simulator as simulator


def _objects(node, path="#"):
    if isinstance(node, dict):
        if "properties" in node:
            yield path, node
        for key, value in node.items():
            yield from _objects(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _objects(value, f"{path}[{index}]")


def _keywords(node):
    if isinstance(node, dict):
        yield from node
        for value in node.values():
            yield from _keywords(value)
    elif isinstance(node, list):
        for value in node:
            yield from _keywords(value)


@pytest.mark.parametrize("source_count", [0, 3])
def test_judge_schema_is_strict_compatible(source_count):
    output = simulator.schema(source_count)
    assert output["strict"] is True
    objects = list(_objects(output["schema"]))
    assert {path for path, _ in objects} >= {"#", "#/$defs/CorrectedAssessment"}
    for path, node in objects:
        assert node["required"] == list(node["properties"]), path
        assert node["additionalProperties"] is False, path
    assert not set(_keywords(output["schema"])) & {
        "default",
        "minLength",
        "maxLength",
    }


def test_strict_answer_with_nulls_still_validates():
    answer = simulator.Answer.model_validate(
        {
            "user_would": "approve",
            "risk": "low",
            "confidence": "high",
            "basis": "routine_within_task",
            "cites": [1],
            "question": "",
            "action_summary": "Read the task status.",
            "explanation": "Routine status check within the task.",
            "restriction_quote": "",
            "corrected_assessment": None,
        }
    )
    assert answer.corrected_assessment is None
    with pytest.raises(ValueError):
        simulator.Answer.model_validate(
            {**answer.model_dump(), "explanation": "x" * 701}
        )
