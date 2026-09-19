#!/usr/bin/env python3
"""Compare semantic-best vs provenance-aware evidence candidate selectors.

Consumes cached candidate-discovery CSV plus cached diagnostics. It never runs
Whisper and never calls /predict.

Experimental invariant:
- both selector variants receive the same candidates in the same anonymized order;
- neither sees gold timestamps, tIoU, discovery rank, or Pass-1 score;
- only the selector objective differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx


PROMPT_VERSION = "provenance-selector-v1"
SELECTOR_VARIANTS = ("semantic", "provenance")
SOURCE_ROLES = (
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
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

COMMON_SYSTEM = """You choose exactly one evidence occurrence for a supported proposition in a doctor-patient dialogue.

You will receive a question and several anonymized candidate occurrences. Every candidate was independently retrieved as potentially supporting the proposition.

Important constraints:
- Consider only the supplied candidates and their local dialogue context.
- Do not infer from candidate display order. The order is intentionally shuffled.
- Do not use timestamps as a preference signal except to understand local chronology.
- Do not rewrite or refine boundaries.
- Return exactly one candidate label.
- Also classify the selected candidate's discourse role using the supplied role enum.
- Output JSON only matching the schema.
"""

SEMANTIC_OBJECTIVE = """Selection objective: choose the candidate that most directly, explicitly, self-containedly, and completely proves the proposition as written.

Prefer the strongest standalone semantic entailment. Do not try to guess annotation provenance or which wording may have generated the question."""

PROVENANCE_OBJECTIVE = """Selection objective: choose the candidate most likely to be the designated source evidence in an annotated doctor-patient dataset, even when another candidate is a stronger standalone paraphrase.

Several candidates may all truthfully support the proposition. Consider the discourse role of each occurrence: initial report, clarification, clinician assessment, explicit confirmation, informed preference, finalized plan, completed action, test/result, follow-up consequence, or later summary.

Choose the occurrence whose wording and conversational role most naturally appears to be the source/provenance of the proposition. Do NOT automatically prefer later passages, clinician speech, or the most explicit semantic paraphrase."""


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    return float(text) if text else None


def tiou(
    pred_start: float | None,
    pred_end: float | None,
    gold_start: float | None,
    gold_end: float | None,
) -> float:
    if None in {pred_start, pred_end, gold_start, gold_end}:
        return 0.0
    assert pred_start is not None and pred_end is not None
    assert gold_start is not None and gold_end is not None
    if pred_end < pred_start or gold_end < gold_start:
        return 0.0
    overlap = max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = max(pred_end, gold_end) - min(pred_start, gold_start)
    return overlap / union if union > 0 else 0.0


def load_diagnostics(path: Path) -> dict[str, dict[str, Any]]:
    files = sorted(path.glob("conversation_*.json"))
    if not files:
        raise FileNotFoundError(f"No conversation_*.json files found in {path}")
    payloads: dict[str, dict[str, Any]] = {}
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        stem = Path(str(payload["audio_filename"])).stem
        transcript_id = (
            stem[len("conversation_"):]
            if stem.startswith("conversation_")
            else stem
        )
        payloads[transcript_id] = payload
    return payloads


def _candidate_indices(fieldnames: list[str]) -> list[int]:
    indices = []
    for field in fieldnames:
        if field.startswith("candidate_") and field.endswith("_start_word_id"):
            try:
                indices.append(int(field.split("_")[1]))
            except ValueError:
                pass
    return sorted(set(indices))


def load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        candidate_indices = _candidate_indices(reader.fieldnames or [])
        if not candidate_indices:
            raise ValueError(f"No candidate columns found in {path}")
        rows = []
        for raw in reader:
            candidates = []
            for index in candidate_indices:
                start_id = raw.get(f"candidate_{index}_start_word_id", "").strip()
                end_id = raw.get(f"candidate_{index}_end_word_id", "").strip()
                if not start_id or not end_id:
                    continue
                candidates.append({
                    "source": f"discovery_{index}",
                    "original_rank": index,
                    "start_word_id": int(start_id),
                    "end_word_id": int(end_id),
                    "start": float(raw[f"candidate_{index}_start"]),
                    "end": float(raw[f"candidate_{index}_end"]),
                    "text": raw.get(f"candidate_{index}_text", ""),
                    "tiou": float(raw.get(f"candidate_{index}_tiou") or 0.0),
                })
            rows.append({
                "transcript_id": raw["transcript_id"],
                "question_id": raw.get("question_id", ""),
                "question_index": int(raw.get("question_index") or 0),
                "question": raw["question"],
                "baseline_tiou": float(raw.get("baseline_tiou") or 0.0),
                "baseline_start": float_or_none(raw.get("baseline_start")),
                "baseline_end": float_or_none(raw.get("baseline_end")),
                "baseline_start_word_id": (
                    int(raw["baseline_start_word_id"])
                    if raw.get("baseline_start_word_id", "").strip()
                    else None
                ),
                "baseline_end_word_id": (
                    int(raw["baseline_end_word_id"])
                    if raw.get("baseline_end_word_id", "").strip()
                    else None
                ),
                "baseline_text": raw.get("baseline_text", ""),
                "gold_start": float(raw["gold_start"]),
                "gold_end": float(raw["gold_end"]),
                "candidates": candidates,
            })
    return rows


def _word_lookup(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(word["id"]): word for word in payload["words"]}


def candidate_from_baseline(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    start_id = row["baseline_start_word_id"]
    end_id = row["baseline_end_word_id"]
    if start_id is None or end_id is None or end_id < start_id:
        return None
    words = _word_lookup(payload)
    ids = list(range(start_id, end_id + 1))
    if any(word_id not in words for word_id in ids):
        return None
    selected = [words[word_id] for word_id in ids]
    start = float(selected[0]["start"])
    end = float(selected[-1]["end"])
    return {
        "source": "pass1",
        "original_rank": None,
        "start_word_id": start_id,
        "end_word_id": end_id,
        "start": start,
        "end": end,
        "text": " ".join(str(word["text"]).strip() for word in selected).strip(),
        "tiou": tiou(start, end, row["gold_start"], row["gold_end"]),
    }


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    result = []
    for candidate in candidates:
        key = (candidate["start_word_id"], candidate["end_word_id"])
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def prepare_candidates(
    row: dict[str, Any],
    payload: dict[str, Any],
    include_pass1: bool,
) -> list[dict[str, Any]]:
    candidates = list(row["candidates"])
    if include_pass1:
        baseline = candidate_from_baseline(row, payload)
        if baseline is not None:
            candidates.append(baseline)
    return dedupe_candidates(candidates)


def stable_permutation(
    candidates: list[dict[str, Any]],
    row: dict[str, Any],
    seed: str,
    permutation_index: int,
) -> list[dict[str, Any]]:
    material = (
        f"{seed}|{permutation_index}|{row['transcript_id']}|"
        f"{row['question_id']}|{row['question']}"
    )
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


def segment_word_text(payload: dict[str, Any], segment_id: int) -> str:
    words = [
        word for word in payload["words"]
        if int(word["segment_id"]) == segment_id
    ]
    words.sort(key=lambda word: int(word["id"]))
    if words:
        return " ".join(f"{int(word['id'])}:{word['text']}" for word in words)
    for segment in payload["segments"]:
        if int(segment["id"]) == segment_id:
            return str(segment.get("text", ""))
    return ""


def local_context(
    payload: dict[str, Any],
    candidate: dict[str, Any],
    radius: int,
) -> str:
    segments = sorted(payload["segments"], key=lambda item: int(item["id"]))
    segment_ids = [int(segment["id"]) for segment in segments]
    words = _word_lookup(payload)
    candidate_words = [
        words[word_id]
        for word_id in range(
            candidate["start_word_id"], candidate["end_word_id"] + 1
        )
        if word_id in words
    ]
    if not candidate_words:
        return candidate["text"]
    hit_ids = sorted(set(int(word["segment_id"]) for word in candidate_words))
    positions = [
        segment_ids.index(segment_id)
        for segment_id in hit_ids
        if segment_id in segment_ids
    ]
    if not positions:
        return candidate["text"]
    lo = max(0, min(positions) - radius)
    hi = min(len(segments), max(positions) + radius + 1)
    lines = []
    for segment in segments[lo:hi]:
        sid = int(segment["id"])
        lines.append(
            f"[S{sid} {float(segment['start']):.2f}-"
            f"{float(segment['end']):.2f}]"
        )
        lines.append(segment_word_text(payload, sid))
    return "\n".join(lines)


def selector_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_candidate": {"type": "string", "enum": labels},
            "source_role": {"type": "string", "enum": list(SOURCE_ROLES)},
        },
        "required": ["selected_candidate", "source_role"],
        "additionalProperties": False,
    }


def build_prompt(
    row: dict[str, Any],
    payload: dict[str, Any],
    ordered_candidates: list[dict[str, Any]],
    context_radius: int,
) -> tuple[str, dict[str, dict[str, Any]]]:
    label_map: dict[str, dict[str, Any]] = {}
    blocks = []
    for index, candidate in enumerate(ordered_candidates):
        label = LETTERS[index]
        label_map[label] = candidate
        blocks.extend([
            f"CANDIDATE {label}",
            local_context(payload, candidate, context_radius),
            (
                f"SELECTED RANGE: W{candidate['start_word_id']}-"
                f"W{candidate['end_word_id']}\n{candidate['text']}"
            ),
            "",
        ])
    return (
        f"QUESTION:\n{row['question']}\n\n"
        + "\n".join(blocks)
        + "\nChoose exactly one candidate label."
    ), label_map


def system_prompt(selector: str) -> str:
    if selector == "semantic":
        return COMMON_SYSTEM + "\n" + SEMANTIC_OBJECTIVE
    if selector == "provenance":
        return COMMON_SYSTEM + "\n" + PROVENANCE_OBJECTIVE
    raise ValueError(f"Unknown selector: {selector}")


def selector_cache_key(
    row: dict[str, Any],
    selector: str,
    model: str,
    labels_and_candidates: list[tuple[str, dict[str, Any]]],
    context_radius: int,
    permutation_index: int,
) -> str:
    material = {
        "prompt_version": PROMPT_VERSION,
        "selector": selector,
        "model": model,
        "context_radius": context_radius,
        "permutation_index": permutation_index,
        "transcript_id": row["transcript_id"],
        "question_id": row["question_id"],
        "question": row["question"],
        "candidates": [
            [label, candidate["start_word_id"], candidate["end_word_id"]]
            for label, candidate in labels_and_candidates
        ],
    }
    blob = json.dumps(
        material, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def call_selector(
    *,
    prompt: str,
    labels: list[str],
    selector: str,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> dict[str, str]:
    request_payload = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "format": selector_schema(labels),
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
        "messages": [
            {"role": "system", "content": system_prompt(selector)},
            {"role": "user", "content": prompt},
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
                raw = json.loads(response.json()["message"]["content"])
            selected = str(raw["selected_candidate"])
            source_role = str(raw["source_role"])
            if selected not in labels:
                raise ValueError(f"Invalid candidate label: {selected}")
            if source_role not in SOURCE_ROLES:
                raise ValueError(f"Invalid source role: {source_role}")
            return {
                "selected_candidate": selected,
                "source_role": source_role,
            }
        except (
            httpx.TimeoutException,
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.2)
    raise RuntimeError(
        f"{selector} selector failed twice: {last_error}"
    ) from last_error


def select_with_cache(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    ordered_candidates: list[dict[str, Any]],
    selector: str,
    cache_dir: Path,
    refresh: bool,
    context_radius: int,
    permutation_index: int,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> tuple[dict[str, Any], bool, Path]:
    prompt, label_map = build_prompt(
        row, payload, ordered_candidates, context_radius
    )
    labels = list(label_map)
    key = selector_cache_key(
        row, selector, model, list(label_map.items()),
        context_radius, permutation_index,
    )
    safe_qid = row["question_id"] or f"q{row['question_index']}"
    cache_path = (
        cache_dir / selector /
        f"{row['transcript_id']}__{safe_qid}__p{permutation_index}__{key}.json"
    )
    if cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        result = cached["result"]
        return {
            **result,
            "selected": label_map[result["selected_candidate"]],
        }, True, cache_path

    result = call_selector(
        prompt=prompt,
        labels=labels,
        selector=selector,
        ollama_url=ollama_url,
        model=model,
        timeout=timeout,
        keep_alive=keep_alive,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "selector": selector,
                "model": model,
                "question_id": row["question_id"],
                "question": row["question"],
                "permutation_index": permutation_index,
                "label_to_candidate": {
                    label: {
                        key: candidate[key]
                        for key in (
                            "source", "original_rank",
                            "start_word_id", "end_word_id",
                            "start", "end", "text",
                        )
                    }
                    for label, candidate in label_map.items()
                },
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        **result,
        "selected": label_map[result["selected_candidate"]],
    }, False, cache_path


def classify_baseline_bucket(score: float) -> str:
    if score <= 1e-12:
        return "zero"
    if score < 0.50:
        return "difficult"
    if score < 0.75:
        return "medium"
    return "control"


def run_selector(
    *,
    rows: list[dict[str, Any]],
    payloads: dict[str, dict[str, Any]],
    selectors: list[str],
    include_pass1: bool,
    permutations: int,
    seed: str,
    cache_dir: Path,
    refresh: bool,
    context_radius: int,
    ollama_url: str,
    model: str,
    timeout: float,
    keep_alive: str,
    num_ctx: int,
    num_predict: int,
) -> tuple[list[dict[str, Any]], int, int]:
    outputs = []
    model_calls = 0
    cache_hits = 0
    for row_index, row in enumerate(rows, start=1):
        payload = payloads[row["transcript_id"]]
        candidates = prepare_candidates(row, payload, include_pass1)
        if not candidates:
            continue
        oracle = max(candidates, key=lambda item: float(item["tiou"]))
        for permutation_index in range(permutations):
            ordered = stable_permutation(
                candidates, row, seed, permutation_index
            )
            for selector in selectors:
                result, from_cache, cache_path = select_with_cache(
                    row=row,
                    payload=payload,
                    ordered_candidates=ordered,
                    selector=selector,
                    cache_dir=cache_dir,
                    refresh=refresh,
                    context_radius=context_radius,
                    permutation_index=permutation_index,
                    ollama_url=ollama_url,
                    model=model,
                    timeout=timeout,
                    keep_alive=keep_alive,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
                cache_hits += int(from_cache)
                model_calls += int(not from_cache)
                selected = result["selected"]
                selected_tiou = float(selected["tiou"])
                baseline_tiou = float(row["baseline_tiou"])
                outputs.append({
                    "transcript_id": row["transcript_id"],
                    "question_id": row["question_id"],
                    "question_index": row["question_index"],
                    "question": row["question"],
                    "bucket": classify_baseline_bucket(baseline_tiou),
                    "selector": selector,
                    "permutation_index": permutation_index,
                    "selected_label": result["selected_candidate"],
                    "source_role": result["source_role"],
                    "selected_source": selected["source"],
                    "selected_original_rank": selected["original_rank"],
                    "selected_start_word_id": selected["start_word_id"],
                    "selected_end_word_id": selected["end_word_id"],
                    "selected_start": selected["start"],
                    "selected_end": selected["end"],
                    "selected_text": selected["text"],
                    "selected_tiou": selected_tiou,
                    "baseline_tiou": baseline_tiou,
                    "candidate_oracle_tiou": float(oracle["tiou"]),
                    "candidate_oracle_source": oracle["source"],
                    "candidate_count": len(candidates),
                    "regret": float(oracle["tiou"]) - selected_tiou,
                    "improved_vs_baseline": (
                        selected_tiou > baseline_tiou + 1e-12
                    ),
                    "degraded_vs_baseline": (
                        selected_tiou + 1e-12 < baseline_tiou
                    ),
                    "high_quality": selected_tiou >= 0.75,
                    "any_overlap": selected_tiou > 0.0,
                    "oracle_chosen": (
                        abs(selected_tiou - float(oracle["tiou"])) <= 1e-12
                    ),
                    "cache_path": str(cache_path),
                })
        print(
            f"[{row_index}/{len(rows)}] "
            f"{row['question_id'] or row['transcript_id']} "
            f"({len(candidates)} candidates)"
        )
    return outputs, model_calls, cache_hits


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "mean_selected_tiou": 0.0,
            "mean_baseline_tiou": 0.0,
            "mean_candidate_oracle_tiou": 0.0,
            "mean_regret": 0.0,
            "oracle_choice_rate": 0.0,
            "high_quality_rate": 0.0,
            "overlap_rate": 0.0,
            "improved_vs_baseline": 0,
            "degraded_vs_baseline": 0,
            "unchanged_vs_baseline": 0,
            "mean_delta_vs_baseline": 0.0,
        }
    n = len(rows)
    deltas = [
        float(row["selected_tiou"]) - float(row["baseline_tiou"])
        for row in rows
    ]
    improved = sum(bool(row["improved_vs_baseline"]) for row in rows)
    degraded = sum(bool(row["degraded_vs_baseline"]) for row in rows)
    return {
        "n": n,
        "mean_selected_tiou": _mean([
            float(row["selected_tiou"]) for row in rows
        ]),
        "mean_baseline_tiou": _mean([
            float(row["baseline_tiou"]) for row in rows
        ]),
        "mean_candidate_oracle_tiou": _mean([
            float(row["candidate_oracle_tiou"]) for row in rows
        ]),
        "mean_regret": _mean([float(row["regret"]) for row in rows]),
        "oracle_choice_rate": (
            sum(bool(row["oracle_chosen"]) for row in rows) / n
        ),
        "high_quality_rate": (
            sum(bool(row["high_quality"]) for row in rows) / n
        ),
        "overlap_rate": (
            sum(bool(row["any_overlap"]) for row in rows) / n
        ),
        "improved_vs_baseline": improved,
        "degraded_vs_baseline": degraded,
        "unchanged_vs_baseline": n - improved - degraded,
        "mean_delta_vs_baseline": _mean(deltas),
    }


def summarize(
    outputs: list[dict[str, Any]],
    selectors: list[str],
    permutations: int,
    model_calls: int,
    cache_hits: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "selectors": selectors,
        "permutations": permutations,
        "model_calls": model_calls,
        "cache_hits": cache_hits,
        "by_selector": {},
        "pairwise": {},
        "permutation_stability": {},
    }
    for selector in selectors:
        selector_rows = [
            row for row in outputs if row["selector"] == selector
        ]
        summary["by_selector"][selector] = {
            "overall": summarize_group(selector_rows),
            "by_bucket": {
                bucket: summarize_group([
                    row for row in selector_rows if row["bucket"] == bucket
                ])
                for bucket in ("zero", "difficult", "medium", "control")
            },
        }
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in selector_rows:
            grouped[(row["transcript_id"], row["question_id"])].append(row)
        stable = sum(
            len({
                (item["selected_start_word_id"], item["selected_end_word_id"])
                for item in group
            }) == 1
            for group in grouped.values()
        )
        summary["permutation_stability"][selector] = {
            "questions": len(grouped),
            "stable_questions": stable,
            "stable_rate": stable / len(grouped) if grouped else 0.0,
        }

    if set(selectors) == {"semantic", "provenance"}:
        for permutation_index in range(permutations):
            semantic = {
                (row["transcript_id"], row["question_id"]): row
                for row in outputs
                if row["selector"] == "semantic"
                and row["permutation_index"] == permutation_index
            }
            provenance = {
                (row["transcript_id"], row["question_id"]): row
                for row in outputs
                if row["selector"] == "provenance"
                and row["permutation_index"] == permutation_index
            }
            common = sorted(set(semantic) & set(provenance))
            diffs = [
                float(provenance[key]["selected_tiou"]) -
                float(semantic[key]["selected_tiou"])
                for key in common
            ]
            summary["pairwise"][f"permutation_{permutation_index}"] = {
                "n": len(common),
                "mean_provenance_minus_semantic_tiou": _mean(diffs),
                "provenance_better": sum(delta > 1e-12 for delta in diffs),
                "semantic_better": sum(delta < -1e-12 for delta in diffs),
                "equal": sum(abs(delta) <= 1e-12 for delta in diffs),
            }
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    outputs: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Evidence selector comparison",
        "",
        "Both selectors saw identical anonymized candidates and local context. "
        "Gold timestamps and candidate tIoUs were used only for post-hoc scoring.",
        "",
    ]
    for selector, block in summary["by_selector"].items():
        overall = block["overall"]
        lines.extend([
            f"## {selector}",
            "",
            f"- mean selected tIoU: {overall['mean_selected_tiou']:.3f}",
            f"- mean candidate-oracle tIoU: "
            f"{overall['mean_candidate_oracle_tiou']:.3f}",
            f"- mean regret: {overall['mean_regret']:.3f}",
            f"- oracle-choice rate: {overall['oracle_choice_rate']:.3f}",
            f"- high-quality rate (tIoU >= .75): "
            f"{overall['high_quality_rate']:.3f}",
            f"- mean delta vs Pass 1: "
            f"{overall['mean_delta_vs_baseline']:+.3f}",
            "",
        ])
    if summary["pairwise"]:
        lines.extend(["## Pairwise semantic vs provenance", ""])
        for key, block in summary["pairwise"].items():
            lines.append(
                f"- {key}: provenance-semantic "
                f"{block['mean_provenance_minus_semantic_tiou']:+.3f}; "
                f"provenance better {block['provenance_better']}, "
                f"semantic better {block['semantic_better']}, "
                f"equal {block['equal']}"
            )
        lines.append("")

    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in outputs:
        grouped[
            (row["transcript_id"], row["question_id"], row["permutation_index"])
        ][row["selector"]] = row

    lines.extend(["## Per-question choices", ""])
    for key in sorted(grouped):
        group = grouped[key]
        first = next(iter(group.values()))
        lines.extend([
            f"### {first['question_id'] or first['transcript_id']} "
            f"(permutation {first['permutation_index']})",
            "",
            f"**Question:** {first['question']}",
            "",
            f"Pass-1 tIoU: {first['baseline_tiou']:.3f}; "
            f"candidate oracle: {first['candidate_oracle_tiou']:.3f}",
            "",
        ])
        for selector in SELECTOR_VARIANTS:
            if selector not in group:
                continue
            row = group[selector]
            lines.extend([
                f"**{selector}:** {row['selected_label']} | "
                f"role={row['source_role']} | "
                f"tIoU={row['selected_tiou']:.3f} | "
                f"source={row['selected_source']}",
                "",
                row["selected_text"] or "(empty)",
                "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print("Evidence selector comparison")
    print(f"  model calls     {summary['model_calls']}")
    print(f"  cache hits      {summary['cache_hits']}")
    print(f"  permutations    {summary['permutations']}")
    print()
    for selector, block in summary["by_selector"].items():
        overall = block["overall"]
        stability = summary["permutation_stability"][selector]
        print(selector)
        print(f"  mean selected tIoU        {overall['mean_selected_tiou']:.3f}")
        print(f"  candidate oracle          {overall['mean_candidate_oracle_tiou']:.3f}")
        print(f"  mean regret               {overall['mean_regret']:.3f}")
        print(f"  oracle-choice rate        {overall['oracle_choice_rate']:.3f}")
        print(f"  tIoU>=.75 rate            {overall['high_quality_rate']:.3f}")
        print(f"  delta vs Pass 1           {overall['mean_delta_vs_baseline']:+.3f}")
        print(f"  permutation stability     {stability['stable_rate']:.3f}")
        print()
    for key, block in summary["pairwise"].items():
        print(
            f"{key}: provenance-semantic "
            f"{block['mean_provenance_minus_semantic_tiou']:+.3f} | "
            f"P better {block['provenance_better']} | "
            f"S better {block['semantic_better']} | "
            f"equal {block['equal']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".diagnostics/evidence-provenance-selector"),
    )
    parser.add_argument(
        "--selector",
        choices=["semantic", "provenance", "both"],
        default="both",
    )
    parser.add_argument("--include-pass1", action="store_true")
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--seed", default="nordic-provenance-v1")
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
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
    parser.add_argument("--num-predict", type=int, default=128)
    args = parser.parse_args()

    if args.permutations < 1:
        raise ValueError("--permutations must be >= 1")
    if args.context_radius < 0:
        raise ValueError("--context-radius must be >= 0")

    selectors = (
        list(SELECTOR_VARIANTS)
        if args.selector == "both"
        else [args.selector]
    )

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
    outputs, model_calls, cache_hits = run_selector(
        rows=rows,
        payloads=payloads,
        selectors=selectors,
        include_pass1=args.include_pass1,
        permutations=args.permutations,
        seed=args.seed,
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
    summary = summarize(
        outputs, selectors, args.permutations, model_calls, cache_hits
    )
    summary.update({
        "candidate_results": str(args.candidate_results),
        "diagnostics_dir": str(args.diagnostics_dir),
        "include_pass1": args.include_pass1,
        "seed": args.seed,
        "context_radius": args.context_radius,
        "model": args.model,
    })

    summary_path = args.output_dir / "summary.json"
    rows_path = args.output_dir / "selector_results.csv"
    markdown_path = args.output_dir / "selector_results.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(outputs, rows_path)
    write_markdown(outputs, summary, markdown_path)

    print()
    print_summary(summary)
    print()
    print(f"Wrote {summary_path}")
    print(f"Wrote {rows_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
