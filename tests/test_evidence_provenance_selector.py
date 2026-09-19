from experiments.evidence_provenance_selector import (
    build_prompt,
    candidate_from_baseline,
    dedupe_candidates,
    local_context,
    stable_permutation,
    summarize,
    system_prompt,
)


def _payload():
    return {
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "a b"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "c d"},
            {"id": 2, "start": 4.0, "end": 6.0, "text": "e f"},
        ],
        "words": [
            {"id": 0, "text": "a", "start": 0.0, "end": 1.0, "segment_id": 0},
            {"id": 1, "text": "b", "start": 1.0, "end": 2.0, "segment_id": 0},
            {"id": 2, "text": "c", "start": 2.0, "end": 3.0, "segment_id": 1},
            {"id": 3, "text": "d", "start": 3.0, "end": 4.0, "segment_id": 1},
            {"id": 4, "text": "e", "start": 4.0, "end": 5.0, "segment_id": 2},
            {"id": 5, "text": "f", "start": 5.0, "end": 6.0, "segment_id": 2},
        ],
    }


def _row():
    return {
        "transcript_id": "sample_1",
        "question_id": "sample_1_yes_q01",
        "question_index": 0,
        "question": "Was c d said?",
        "baseline_tiou": 0.0,
        "baseline_start_word_id": 0,
        "baseline_end_word_id": 1,
        "baseline_text": "a b",
        "gold_start": 2.0,
        "gold_end": 4.0,
    }


def test_selector_objectives_are_distinct():
    semantic = system_prompt("semantic")
    provenance = system_prompt("provenance")
    assert semantic != provenance
    assert "standalone semantic entailment" in semantic
    assert "designated source evidence" in provenance


def test_stable_permutation_is_deterministic():
    row = _row()
    candidates = [
        {"start_word_id": i, "end_word_id": i}
        for i in range(4)
    ]
    first = stable_permutation(candidates, row, "seed", 0)
    second = stable_permutation(candidates, row, "seed", 0)
    assert [item["start_word_id"] for item in first] == [
        item["start_word_id"] for item in second
    ]


def test_local_context_includes_neighbor_segments():
    candidate = {
        "start_word_id": 2,
        "end_word_id": 3,
        "text": "c d",
    }
    context = local_context(_payload(), candidate, radius=1)
    assert "0:a" in context
    assert "2:c" in context
    assert "4:e" in context


def test_pass1_candidate_can_be_reconstructed_and_deduped():
    candidate = candidate_from_baseline(_row(), _payload())
    assert candidate is not None
    assert candidate["source"] == "pass1"
    assert candidate["text"] == "a b"
    deduped = dedupe_candidates([candidate, dict(candidate)])
    assert len(deduped) == 1


def test_build_prompt_anonymizes_candidates():
    row = _row()
    candidates = [
        {
            "source": "discovery_2",
            "original_rank": 2,
            "start_word_id": 2,
            "end_word_id": 3,
            "start": 2.0,
            "end": 4.0,
            "text": "c d",
            "tiou": 1.0,
        },
        {
            "source": "discovery_1",
            "original_rank": 1,
            "start_word_id": 4,
            "end_word_id": 5,
            "start": 4.0,
            "end": 6.0,
            "text": "e f",
            "tiou": 0.0,
        },
    ]
    prompt, label_map = build_prompt(row, _payload(), candidates, 0)
    assert list(label_map) == ["A", "B"]
    assert "discovery_1" not in prompt
    assert "discovery_2" not in prompt
    assert "CANDIDATE A" in prompt
    assert "CANDIDATE B" in prompt


def test_pairwise_summary_measures_provenance_advantage():
    outputs = [
        {
            "selector": "semantic",
            "bucket": "zero",
            "transcript_id": "s",
            "question_id": "q",
            "permutation_index": 0,
            "selected_start_word_id": 1,
            "selected_end_word_id": 2,
            "selected_tiou": 0.2,
            "baseline_tiou": 0.0,
            "candidate_oracle_tiou": 0.8,
            "regret": 0.6,
            "oracle_chosen": False,
            "high_quality": False,
            "any_overlap": True,
            "improved_vs_baseline": True,
            "degraded_vs_baseline": False,
        },
        {
            "selector": "provenance",
            "bucket": "zero",
            "transcript_id": "s",
            "question_id": "q",
            "permutation_index": 0,
            "selected_start_word_id": 3,
            "selected_end_word_id": 4,
            "selected_tiou": 0.8,
            "baseline_tiou": 0.0,
            "candidate_oracle_tiou": 0.8,
            "regret": 0.0,
            "oracle_chosen": True,
            "high_quality": True,
            "any_overlap": True,
            "improved_vs_baseline": True,
            "degraded_vs_baseline": False,
        },
    ]
    summary = summarize(
        outputs,
        ["semantic", "provenance"],
        permutations=1,
        model_calls=2,
        cache_hits=0,
    )
    pair = summary["pairwise"]["permutation_0"]
    assert abs(pair["mean_provenance_minus_semantic_tiou"] - 0.6) < 1e-9
    assert pair["provenance_better"] == 1
