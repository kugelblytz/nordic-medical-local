#!/usr/bin/env python3
"""Extract candidate-local features and rank evidence without listwise choice.

This experiment removes candidate-order bias by:
1. classifying the question once, without candidates;
2. evaluating each evidence candidate independently, one model call at a time;
3. caching all structured features;
4. ranking candidates deterministically from those cached features.

It never runs Whisper and never calls /predict. Gold/tIoU is used only after
ranking for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from experiments.evidence_provenance_selector import (
    classify_baseline_bucket,
    load_candidate_rows,
    load_diagnostics,
    local_context,
    prepare_candidates,
)


PROMPT_VERSION = "candidate-features-v1"

QUESTION_TARGETS = (
    "diagnosis_or_assessment",
    "patient_symptom_or_experience",
    "patient_preference_or_request",
    "test_or_result",
    "treatment_plan_or_continuation",
    "completed_treatment_or_action",
    "medication_effect",
    "history_or_timing",
    "absence_or_normal_state",
    "other",
)

SPEAKER_ROLES = ("patient", "clinician", "both", "unknown")

DISCOURSE_ROLES = (
    "initial_report",
    "patient_experience",
    "patient_preference",
    "clarification",
    "clinician_assessment",
    "explicit_confirmation",
    "test_or_result",
    "final_plan",
    "completed_action",
    "followup_outcome",
    "summary",
    "other",
)

SUPPORT_FORMS = (
    "direct_statement",
    "question_answer_exchange",
    "inferred_from_action",
    "inferred_from_result",
    "repetition_or_summary",
    "other",
)

TEMPORAL_ROLES = ("initial", "intermediate", "final", "unknown")

ACTION_STATUSES = (
    "not_applicable",
    "discussed",
    "planned",
    "agreed",
    "completed",
    "result_or_followup",
)

QUESTION_SYSTEM = """Classify the information need of one factual yes/no question
from a doctor-patient dataset. Do not answer the question. Do not infer from
any transcript because none is supplied.

Return the single best question_target. Also say whether the question is
primarily about a finalized/committed state rather than an initial mention,
possibility, or discussion.

Output JSON only matching the schema."""

CANDIDATE_SYSTEM = """Analyze ONE possible evidence occurrence from a
doctor-patient dialogue. You are not choosing among candidates; no other
candidate exists in this call.

Given the question, one selected word range, and its local dialogue context,
describe this occurrence using the requested structured features.

Rules:
- Judge only this occurrence and its local context.
- Do not guess whether another occurrence elsewhere is better.
- question_alignment measures how well this occurrence supports the exact
  proposition, from 0 (not aligned) to 3 (direct and complete).
- explicitness measures how explicitly the relevant fact is stated, from
  0 (only indirect implication) to 3 (fully explicit wording).
- self_contained is true when the selected range can be understood as evidence
  without relying materially on words outside the selected range.
- finalized is true when this occurrence represents a settled, confirmed,
  implemented, completed, or final state rather than an initial report,
  suspicion, proposal, or intermediate discussion.
- discourse_role describes what this occurrence is doing in the dialogue.
- temporal_role is relative to the local development of this fact, not absolute
  wall-clock position.
- action_status concerns treatment/action status when applicable.

Do not rank. Do not mention gold annotations. Output JSON only matching the
schema."""


ROLE_BONUSES: dict[str, dict[str, float]] = {
    "diagnosis_or_assessment": {
        "clinician_assessment": 2.5,
        "explicit_confirmation": 1.5,
        "test_or_result": 0.5,
        "patient_experience": -0.5,
        "initial_report": -0.5,
    },
    "patient_symptom_or_experience": {
        "patient_experience": 2.5,
        "initial_report": 1.0,
        "explicit_confirmation": 0.75,
        "clinician_assessment": 0.25,
    },
    "patient_preference_or_request": {
        "patient_preference": 3.0,
        "explicit_confirmation": 1.0,
        "final_plan": 0.5,
        "clinician_assessment": -0.5,
    },
    "test_or_result": {
        "test_or_result": 3.0,
        "explicit_confirmation": 1.0,
        "summary": 0.25,
    },
    "treatment_plan_or_continuation": {
        "final_plan": 3.0,
        "explicit_confirmation": 1.25,
        "clinician_assessment": 0.5,
        "initial_report": -0.5,
    },
    "completed_treatment_or_action": {
        "completed_action": 3.5,
        "followup_outcome": 1.5,
        "explicit_confirmation": 1.0,
        "final_plan": -1.0,
        "patient_preference": -1.0,
    },
    "medication_effect": {
        "explicit_confirmation": 2.5,
        "followup_outcome": 2.0,
        "patient_experience": 1.5,
        "summary": 0.25,
    },
    "history_or_timing": {
        "initial_report": 1.5,
        "clarification": 1.5,
        "explicit_confirmation": 1.0,
        "summary": 0.25,
    },
    "absence_or_normal_state": {
        "patient_experience": 1.5,
        "clinician_assessment": 1.5,
        "test_or_result": 1.5,
        "followup_outcome": 1.5,
        "explicit_confirmation": 1.0,
    },
    "other": {
        "explicit_confirmation": 1.0,
    },
}

ACTION_BONUSES: dict[str, dict[str, float]] = {
    "completed_treatment_or_action": {
        "completed": 2.0,
        "result_or_followup": 0.75,
        "agreed": -0.5,
        "planned": -1.5,
        "discussed": -2.0,
    },
    "treatment_plan_or_continuation": {
        "agreed": 1.5,
        "planned": 1.25,
        "completed": 0.5,
        "discussed": -0.75,
    },
}


def question_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question_target": {
                "type": "string",
                "enum": list(QUESTION_TARGETS),
            },
            "expects_finalized_state": {"type": "boolean"},
        },
        "required": ["question_target", "expects_finalized_state"],
        "additionalProperties": False,
    }


def candidate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "speaker_role": {"type": "string", "enum": list(SPEAKER_ROLES)},
            "discourse_role": {
                "type": "string",
                "enum": list(DISCOURSE_ROLES),
            },
            "support_form": {
                "type": "string",
                "enum": list(SUPPORT_FORMS),
            },
            "temporal_role": {
                "type": "string",
                "enum": list(TEMPORAL_ROLES),
            },
            "action_status": {
                "type": "string",
                "enum": list(ACTION_STATUSES),
            },
            "explicitness": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
            },
            "question_alignment": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
            },
            "self_contained": {"type": "boolean"},
            "finalized": {"type": "boolean"},
        },
        "required": [
            "speaker_role",
            "discourse_role",
            "support_form",
            "temporal_role",
            "action_status",
            "explicitness",
            "question_alignment",
            "self_contained",
            "finalized",
        ],
        "additionalProperties": False,
    }


def _post_json(
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> dict[str, Any]:
    request_payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "format": schema,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout, connect=min(5.0, timeout))
            ) as client:
                response = client.post(
                    f"{ollama_url.rstrip('/')}/api/chat",
                    json=request_payload,
                )
                response.raise_for_status()
                return json.loads(response.json()["message"]["content"])
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.2)
    raise RuntimeError(f"Structured Ollama call failed twice: {last_error}") from last_error


def question_cache_key(row: dict[str, Any], model: str) -> str:
    blob = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "kind": "question",
            "model": model,
            "question": row["question"],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def candidate_cache_key(
    row: dict[str, Any],
    candidate: dict[str, Any],
    model: str,
    context_radius: int,
) -> str:
    blob = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "kind": "candidate",
            "model": model,
            "context_radius": context_radius,
            "transcript_id": row["transcript_id"],
            "question": row["question"],
            "start_word_id": candidate["start_word_id"],
            "end_word_id": candidate["end_word_id"],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def extract_question_features(
    *,
    row: dict[str, Any],
    cache_dir: Path,
    refresh: bool,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> tuple[dict[str, Any], bool, Path]:
    key = question_cache_key(row, model)
    safe_qid = row["question_id"] or f"q{row['question_index']}"
    path = cache_dir / "questions" / f"{row['transcript_id']}__{safe_qid}__{key}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["features"], True, path

    features = _post_json(
        system=QUESTION_SYSTEM,
        user=f"QUESTION:\n{row['question']}",
        schema=question_schema(),
        ollama_url=ollama_url,
        model=model,
        timeout=timeout,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    if features.get("question_target") not in QUESTION_TARGETS:
        raise ValueError(f"Invalid question_target: {features}")
    if not isinstance(features.get("expects_finalized_state"), bool):
        raise ValueError(f"Invalid question features: {features}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "question_id": row["question_id"],
                "question": row["question"],
                "features": features,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return features, False, path


def extract_candidate_features(
    *,
    row: dict[str, Any],
    candidate: dict[str, Any],
    payload: dict[str, Any],
    cache_dir: Path,
    refresh: bool,
    context_radius: int,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> tuple[dict[str, Any], bool, Path]:
    key = candidate_cache_key(row, candidate, model, context_radius)
    safe_qid = row["question_id"] or f"q{row['question_index']}"
    path = (
        cache_dir
        / "candidates"
        / f"{row['transcript_id']}__{safe_qid}__"
          f"W{candidate['start_word_id']}-{candidate['end_word_id']}__{key}.json"
    )
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["features"], True, path

    context = local_context(payload, candidate, context_radius)
    user = (
        f"QUESTION:\n{row['question']}\n\n"
        f"LOCAL CONTEXT:\n{context}\n\n"
        f"SELECTED RANGE:\nW{candidate['start_word_id']}-"
        f"W{candidate['end_word_id']}\n{candidate['text']}"
    )
    features = _post_json(
        system=CANDIDATE_SYSTEM,
        user=user,
        schema=candidate_schema(),
        ollama_url=ollama_url,
        model=model,
        timeout=timeout,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

    enum_checks = {
        "speaker_role": SPEAKER_ROLES,
        "discourse_role": DISCOURSE_ROLES,
        "support_form": SUPPORT_FORMS,
        "temporal_role": TEMPORAL_ROLES,
        "action_status": ACTION_STATUSES,
    }
    for key_name, allowed in enum_checks.items():
        if features.get(key_name) not in allowed:
            raise ValueError(f"Invalid {key_name}: {features}")
    for key_name in ("explicitness", "question_alignment"):
        value = features.get(key_name)
        if not isinstance(value, int) or not 0 <= value <= 3:
            raise ValueError(f"Invalid {key_name}: {features}")
    for key_name in ("self_contained", "finalized"):
        if not isinstance(features.get(key_name), bool):
            raise ValueError(f"Invalid {key_name}: {features}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "question_id": row["question_id"],
                "question": row["question"],
                "candidate": {
                    key_name: candidate[key_name]
                    for key_name in (
                        "source",
                        "original_rank",
                        "start_word_id",
                        "end_word_id",
                        "start",
                        "end",
                        "text",
                    )
                },
                "features": features,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return features, False, path


def neutral_score(features: dict[str, Any]) -> float:
    return (
        2.0 * int(features["question_alignment"])
        + 1.25 * int(features["explicitness"])
        + 0.75 * int(bool(features["self_contained"]))
    )


def provenance_score(
    question_features: dict[str, Any],
    candidate_features: dict[str, Any],
) -> float:
    score = neutral_score(candidate_features)
    target = str(question_features["question_target"])
    role = str(candidate_features["discourse_role"])
    action_status = str(candidate_features["action_status"])

    score += ROLE_BONUSES.get(target, {}).get(role, 0.0)
    score += ACTION_BONUSES.get(target, {}).get(action_status, 0.0)

    if bool(question_features["expects_finalized_state"]):
        score += 1.0 if candidate_features["finalized"] else -0.5
    elif candidate_features["finalized"]:
        score += 0.25

    temporal_role = str(candidate_features["temporal_role"])
    if temporal_role == "final" and bool(question_features["expects_finalized_state"]):
        score += 0.5

    support_form = str(candidate_features["support_form"])
    if target == "medication_effect" and support_form == "question_answer_exchange":
        score += 0.75
    if (
        target == "completed_treatment_or_action"
        and support_form == "inferred_from_action"
    ):
        score += 0.75

    return score


def choose_by_score(
    candidates: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    ranker: str,
    question_features: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    if len(candidates) != len(feature_rows):
        raise ValueError("candidates/features length mismatch")
    scored = []
    for index, (candidate, features) in enumerate(zip(candidates, feature_rows)):
        if ranker == "neutral":
            score = neutral_score(features)
        elif ranker == "provenance":
            score = provenance_score(question_features, features)
        else:
            raise ValueError(f"Unknown ranker: {ranker}")
        scored.append((score, -index, candidate))
    score, _, candidate = max(scored, key=lambda item: (item[0], item[1]))
    return candidate, float(score)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    selected = [float(row["selected_tiou"]) for row in rows]
    baseline = [float(row["baseline_tiou"]) for row in rows]
    oracle = [float(row["candidate_oracle_tiou"]) for row in rows]
    regret = [float(row["regret"]) for row in rows]
    return {
        "n": n,
        "mean_selected_tiou": statistics.mean(selected),
        "mean_baseline_tiou": statistics.mean(baseline),
        "mean_candidate_oracle_tiou": statistics.mean(oracle),
        "mean_regret": statistics.mean(regret),
        "mean_delta_vs_baseline": statistics.mean(
            a - b for a, b in zip(selected, baseline)
        ),
        "oracle_choice_rate": sum(
            abs(a - o) <= 1e-12 for a, o in zip(selected, oracle)
        ) / n,
        "high_quality_rate": sum(value >= 0.75 for value in selected) / n,
        "overlap_rate": sum(value > 0 for value in selected) / n,
        "improved_vs_baseline": sum(
            a > b + 1e-12 for a, b in zip(selected, baseline)
        ),
        "degraded_vs_baseline": sum(
            a + 1e-12 < b for a, b in zip(selected, baseline)
        ),
    }


def summarize(
    ranking_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    model_calls: int,
    cache_hits: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "model_calls": model_calls,
        "cache_hits": cache_hits,
        "questions": len(question_rows),
        "candidate_feature_rows": len(feature_rows),
        "rankers": {},
        "pairwise": {},
        "feature_distributions": {
            "question_target": dict(Counter(
                row["question_target"] for row in question_rows
            )),
            "discourse_role": dict(Counter(
                row["discourse_role"] for row in feature_rows
            )),
            "speaker_role": dict(Counter(
                row["speaker_role"] for row in feature_rows
            )),
            "support_form": dict(Counter(
                row["support_form"] for row in feature_rows
            )),
        },
    }

    for ranker in ("discovery_top1", "neutral", "provenance"):
        group = [row for row in ranking_rows if row["ranker"] == ranker]
        result["rankers"][ranker] = {
            "overall": summarize_rows(group),
            "by_bucket": {
                bucket: summarize_rows([
                    row for row in group if row["bucket"] == bucket
                ])
                for bucket in ("zero", "difficult", "medium", "control")
            },
        }

    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ranking_rows:
        by_key[(row["transcript_id"], row["question_id"])][row["ranker"]] = row

    for left, right in (
        ("neutral", "discovery_top1"),
        ("provenance", "discovery_top1"),
        ("provenance", "neutral"),
    ):
        deltas = []
        for group in by_key.values():
            if left in group and right in group:
                deltas.append(
                    float(group[left]["selected_tiou"])
                    - float(group[right]["selected_tiou"])
                )
        result["pairwise"][f"{left}_minus_{right}"] = {
            "n": len(deltas),
            "mean_tiou_delta": statistics.mean(deltas) if deltas else 0.0,
            "left_better": sum(delta > 1e-12 for delta in deltas),
            "right_better": sum(delta < -1e-12 for delta in deltas),
            "equal": sum(abs(delta) <= 1e-12 for delta in deltas),
        }
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    print("Independent candidate-feature experiment")
    print(f"  questions               {summary['questions']}")
    print(f"  candidate feature rows  {summary['candidate_feature_rows']}")
    print(f"  model calls             {summary['model_calls']}")
    print(f"  cache hits              {summary['cache_hits']}")
    print()
    for ranker, block in summary["rankers"].items():
        overall = block["overall"]
        print(ranker)
        print(f"  mean selected tIoU      {overall.get('mean_selected_tiou', 0.0):.3f}")
        print(f"  candidate oracle        {overall.get('mean_candidate_oracle_tiou', 0.0):.3f}")
        print(f"  mean regret             {overall.get('mean_regret', 0.0):.3f}")
        print(f"  oracle-choice rate      {overall.get('oracle_choice_rate', 0.0):.3f}")
        print(f"  tIoU>=.75 rate          {overall.get('high_quality_rate', 0.0):.3f}")
        print(f"  delta vs Pass 1         {overall.get('mean_delta_vs_baseline', 0.0):+.3f}")
        print()
    for name, block in summary["pairwise"].items():
        print(
            f"{name}: {block['mean_tiou_delta']:+.3f} | "
            f"left better {block['left_better']} | "
            f"right better {block['right_better']} | "
            f"equal {block['equal']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".diagnostics/evidence-candidate-features"),
    )
    parser.add_argument("--include-pass1", action="store_true")
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "qwen3.5:27b"),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--keep-alive",
        default=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    )
    parser.add_argument("--num-predict", type=int, default=256)
    args = parser.parse_args()

    if args.context_radius < 0:
        raise ValueError("--context-radius must be >= 0")

    rows = load_candidate_rows(args.candidate_results)
    if args.limit is not None:
        rows = rows[:max(0, args.limit)]
    if not rows:
        raise ValueError("No candidate rows to evaluate")

    payloads = load_diagnostics(args.diagnostics_dir)
    missing = sorted({
        row["transcript_id"]
        for row in rows
        if row["transcript_id"] not in payloads
    })
    if missing:
        raise KeyError(f"Missing diagnostics for transcripts: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "cache"

    feature_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    model_calls = 0
    cache_hits = 0

    for row_index, row in enumerate(rows, start=1):
        payload = payloads[row["transcript_id"]]
        candidates = prepare_candidates(row, payload, args.include_pass1)
        if not candidates:
            continue

        q_features, cached, q_cache = extract_question_features(
            row=row,
            cache_dir=cache_dir,
            refresh=args.refresh,
            ollama_url=args.ollama_url,
            model=args.model,
            timeout=args.timeout,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
        cache_hits += int(cached)
        model_calls += int(not cached)
        question_rows.append({
            "transcript_id": row["transcript_id"],
            "question_id": row["question_id"],
            "question": row["question"],
            "bucket": classify_baseline_bucket(float(row["baseline_tiou"])),
            "question_target": q_features["question_target"],
            "expects_finalized_state": q_features["expects_finalized_state"],
            "cache_path": str(q_cache),
        })

        candidate_features = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            features, cached, feature_cache = extract_candidate_features(
                row=row,
                candidate=candidate,
                payload=payload,
                cache_dir=cache_dir,
                refresh=args.refresh,
                context_radius=args.context_radius,
                ollama_url=args.ollama_url,
                model=args.model,
                timeout=args.timeout,
                keep_alive=args.keep_alive,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
            )
            cache_hits += int(cached)
            model_calls += int(not cached)
            candidate_features.append(features)
            feature_rows.append({
                "transcript_id": row["transcript_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "bucket": classify_baseline_bucket(float(row["baseline_tiou"])),
                "candidate_index": candidate_index,
                "candidate_source": candidate["source"],
                "candidate_original_rank": candidate["original_rank"],
                "candidate_start_word_id": candidate["start_word_id"],
                "candidate_end_word_id": candidate["end_word_id"],
                "candidate_start": candidate["start"],
                "candidate_end": candidate["end"],
                "candidate_text": candidate["text"],
                "candidate_tiou": candidate["tiou"],
                "question_target": q_features["question_target"],
                "expects_finalized_state": q_features["expects_finalized_state"],
                **features,
                "neutral_score": neutral_score(features),
                "provenance_score": provenance_score(q_features, features),
                "cache_path": str(feature_cache),
            })

        oracle = max(candidates, key=lambda item: float(item["tiou"]))
        selected_by_ranker = {
            "discovery_top1": (candidates[0], 0.0),
            "neutral": choose_by_score(
                candidates,
                candidate_features,
                ranker="neutral",
                question_features=q_features,
            ),
            "provenance": choose_by_score(
                candidates,
                candidate_features,
                ranker="provenance",
                question_features=q_features,
            ),
        }
        for ranker, (selected, ranking_score) in selected_by_ranker.items():
            selected_tiou = float(selected["tiou"])
            baseline_tiou = float(row["baseline_tiou"])
            ranking_rows.append({
                "transcript_id": row["transcript_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "bucket": classify_baseline_bucket(baseline_tiou),
                "question_target": q_features["question_target"],
                "expects_finalized_state": q_features["expects_finalized_state"],
                "ranker": ranker,
                "selected_source": selected["source"],
                "selected_original_rank": selected["original_rank"],
                "selected_start_word_id": selected["start_word_id"],
                "selected_end_word_id": selected["end_word_id"],
                "selected_text": selected["text"],
                "ranking_score": ranking_score,
                "selected_tiou": selected_tiou,
                "baseline_tiou": baseline_tiou,
                "candidate_oracle_tiou": float(oracle["tiou"]),
                "regret": float(oracle["tiou"]) - selected_tiou,
            })

        print(
            f"[{row_index}/{len(rows)}] "
            f"{row['question_id'] or row['transcript_id']} "
            f"({len(candidates)} independent candidate calls)"
        )

    question_path = args.output_dir / "question_features.csv"
    feature_path = args.output_dir / "candidate_features.csv"
    ranking_path = args.output_dir / "ranking_results.csv"
    write_csv(question_rows, question_path)
    write_csv(feature_rows, feature_path)
    write_csv(ranking_rows, ranking_path)

    if args.extract_only:
        print()
        print(f"Wrote {question_path}")
        print(f"Wrote {feature_path}")
        print("Feature extraction complete; ranking file also contains current deterministic rules.")
        print(f"Wrote {ranking_path}")
        return

    summary = summarize(
        ranking_rows,
        feature_rows,
        question_rows,
        model_calls,
        cache_hits,
    )
    summary.update({
        "candidate_results": str(args.candidate_results),
        "diagnostics_dir": str(args.diagnostics_dir),
        "include_pass1": args.include_pass1,
        "context_radius": args.context_radius,
        "model": args.model,
    })
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print_summary(summary)
    print()
    print(f"Wrote {summary_path}")
    print(f"Wrote {question_path}")
    print(f"Wrote {feature_path}")
    print(f"Wrote {ranking_path}")


if __name__ == "__main__":
    main()
