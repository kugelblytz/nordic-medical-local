from experiments.evidence_candidate_features import (
    neutral_score,
    provenance_score,
    choose_by_score,
    question_schema,
    candidate_schema,
)


def _features(**overrides):
    base = {
        "speaker_role": "clinician",
        "discourse_role": "clinician_assessment",
        "support_form": "direct_statement",
        "temporal_role": "intermediate",
        "action_status": "not_applicable",
        "explicitness": 3,
        "question_alignment": 3,
        "self_contained": True,
        "finalized": True,
    }
    base.update(overrides)
    return base


def test_schemas_are_strict_objects():
    q = question_schema()
    c = candidate_schema()
    assert q["additionalProperties"] is False
    assert c["additionalProperties"] is False
    assert "question_target" in q["required"]
    assert "question_alignment" in c["required"]


def test_neutral_score_ignores_discourse_role():
    first = _features(discourse_role="initial_report")
    second = _features(discourse_role="final_plan")
    assert neutral_score(first) == neutral_score(second)


def test_diagnosis_provenance_prefers_clinician_assessment():
    q = {
        "question_target": "diagnosis_or_assessment",
        "expects_finalized_state": True,
    }
    clinician = _features(discourse_role="clinician_assessment")
    patient = _features(
        speaker_role="patient",
        discourse_role="initial_report",
    )
    assert provenance_score(q, clinician) > provenance_score(q, patient)


def test_completed_action_penalizes_plan():
    q = {
        "question_target": "completed_treatment_or_action",
        "expects_finalized_state": True,
    }
    completed = _features(
        discourse_role="completed_action",
        action_status="completed",
    )
    planned = _features(
        discourse_role="final_plan",
        action_status="planned",
        finalized=False,
    )
    assert provenance_score(q, completed) > provenance_score(q, planned)


def test_patient_preference_prefers_patient_preference_role():
    q = {
        "question_target": "patient_preference_or_request",
        "expects_finalized_state": False,
    }
    preference = _features(
        speaker_role="patient",
        discourse_role="patient_preference",
        finalized=False,
    )
    assessment = _features(discourse_role="clinician_assessment")
    assert provenance_score(q, preference) > provenance_score(q, assessment)


def test_choose_by_score_has_deterministic_original_order_tie_break():
    candidates = [
        {"source": "first", "tiou": 0.1},
        {"source": "second", "tiou": 0.9},
    ]
    features = [
        _features(),
        _features(),
    ]
    q = {
        "question_target": "other",
        "expects_finalized_state": False,
    }
    selected, _ = choose_by_score(
        candidates,
        features,
        ranker="neutral",
        question_features=q,
    )
    assert selected["source"] == "first"


def test_direct_script_help_runs_from_repo_root():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "experiments/evidence_candidate_features.py",
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--candidate-results" in result.stdout
