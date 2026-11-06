"""Typed experience compression, retrieval, and graph writing for online adaptation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


EXPERIENCE_PROTOCOL = "matrecall-experience-memory"
TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?")
MEMORY_VALUE_CHAR_LIMIT = 1600


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _short(value: Any, limit: int) -> str:
    text = canonical_json(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _bounded_value(value: Any, limit: int = MEMORY_VALUE_CHAR_LIMIT) -> Any:
    """Keep short typed values intact and hash-bound long structured values."""
    text = canonical_json(value)
    if len(text) <= limit:
        return value
    return {
        "compressed_json_prefix": text[: limit - 3] + "...",
        "full_value_sha256": digest(value),
        "full_value_characters": len(text),
    }


def _tokens(value: Any) -> set[str]:
    text = canonical_json(value).lower()
    return {match.group(0) for match in TOKEN_RE.finditer(text)}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def prediction_from_result(result: dict[str, Any]) -> Any | None:
    if result.get("status") != "success":
        return None
    try:
        return result["response"]["reasoning_results"]["prediction"]
    except (KeyError, TypeError):
        return None


def feedback_summary(task_kind: str, prediction: Any, target: Any) -> dict[str, Any]:
    """Return a generic, per-case metric used only for memory and learning curves.

    Native task metrics remain authoritative in the existing evaluators.  This
    generic score makes heterogeneous sequential behavior inspectable without
    silently pretending that task-specific units are directly comparable.
    """
    if prediction is None:
        return {
            "metric_name": "unscoreable",
            "score": None,
            "loss": None,
            "exact_match": False,
            "residual": None,
        }
    if task_kind in ("regression", "scalar_regression"):
        try:
            predicted = float(prediction["value"])
            expected = float(target["value"])
            if not (math.isfinite(predicted) and math.isfinite(expected)):
                raise ValueError("non-finite")
            residual = expected - predicted
            loss = abs(residual)
            return {
                "metric_name": "absolute_error",
                "score": -loss,
                "loss": loss,
                "exact_match": predicted == expected,
                "residual": residual,
            }
        except (KeyError, TypeError, ValueError):
            pass
    if task_kind in ("binary_classification", "multiclass_classification"):
        predicted = prediction.get("value") if isinstance(prediction, dict) else prediction
        expected = target.get("value") if isinstance(target, dict) else target
        exact = str(predicted) == str(expected)
        return {
            "metric_name": "exact_match",
            "score": float(exact),
            "loss": float(not exact),
            "exact_match": exact,
            "residual": None,
        }
    predicted_tokens, target_tokens = _tokens(prediction), _tokens(target)
    similarity = _jaccard(predicted_tokens, target_tokens)
    exact = canonical_json(prediction) == canonical_json(target)
    return {
        "metric_name": "structured_token_jaccard",
        "score": similarity,
        "loss": 1.0 - similarity,
        "exact_match": exact,
        "residual": None,
    }


def compress_experience(
    *,
    case: dict[str, Any],
    gold: dict[str, Any],
    result: dict[str, Any],
    sequence_id: str,
    sequence_step: int,
) -> dict[str, Any]:
    """Compress one completed test-then-learn interaction into verified memory."""
    prediction = prediction_from_result(result)
    feedback = feedback_summary(case["task_kind"], prediction, gold["target"])
    anchor = case.get("deterministic_tool_estimate")
    anchor_feedback = feedback_summary(case["task_kind"], anchor, gold["target"])
    improvement = None
    if feedback["score"] is not None and anchor_feedback["score"] is not None:
        improvement = float(feedback["score"] - anchor_feedback["score"])
    residual = feedback.get("residual")
    if residual is None:
        correction = "exact" if feedback.get("exact_match") else "structured_mismatch"
    elif residual > 0:
        correction = "past_prediction_was_too_low"
    elif residual < 0:
        correction = "past_prediction_was_too_high"
    else:
        correction = "no_numeric_correction"
    planner = result.get("planner_record")
    requested_view = None
    if isinstance(planner, dict):
        requested_view = planner.get("requested_view")
    selected_evidence = list(result.get("selected_evidence_ids", []))
    selected_experiences = list(result.get("selected_experience_ids", []))
    query_summary = _short(case["query_visible"], 1200)
    lesson = {
        "correction": correction,
        "requested_view": requested_view,
        "improved_over_static_anchor": improvement is not None and improvement > 0,
        "score_delta_vs_static_anchor": improvement,
        "response_succeeded": result.get("status") == "success",
    }
    episode_id = f"experience:{case['task']}:{sequence_step}:{case['case_id']}"
    raw_trace = {
        "agent_trace": result.get("agent_trace", []),
        "planner_record": planner,
        "selected_evidence_ids": selected_evidence,
        "selected_experience_ids": selected_experiences,
        "response": result.get("response"),
        "error": result.get("error"),
    }
    return {
        "experience_protocol": EXPERIENCE_PROTOCOL,
        "experience_id": episode_id,
        "sequence_id": sequence_id,
        "sequence_step": sequence_step,
        "case_id": case["case_id"],
        "task": case["task"],
        "task_kind": case["task_kind"],
        "query_summary": query_summary,
        "query_tokens": sorted(_tokens(case["query_visible"])),
        "prediction": _bounded_value(prediction),
        "ground_truth": _bounded_value(gold["target"]),
        "feedback": feedback,
        "static_anchor_feedback": anchor_feedback,
        "lesson": lesson,
        "trajectory_summary": {
            "planner_action": planner.get("action") if isinstance(planner, dict) else None,
            "requested_view": requested_view,
            "selected_evidence_ids": selected_evidence,
            "selected_experience_ids": selected_experiences,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "usage": result.get("usage"),
        },
        "raw_trace_sha256": digest(raw_trace),
        "verified_after_prediction": True,
        "epistemic_status": "verified_feedback",
        "memory_text": " ".join((
            query_summary,
            correction,
            str(requested_view or "no_requested_view"),
            "improved" if lesson["improved_over_static_anchor"] else "not_improved",
        )),
    }


def _similarity(query: Any, episode: dict[str, Any], index: int, total: int) -> dict[str, float]:
    query_tokens = _tokens(query)
    episode_tokens = set(episode.get("query_tokens", [])) or _tokens(episode["query_summary"])
    lexical = _jaccard(query_tokens, episode_tokens)
    recency = (index + 1) / max(total, 1)
    score = 0.95 * lexical + 0.05 * recency
    return {"score": score, "lexical_jaccard": lexical, "recency": recency}


def retrieve_case_memory(
    memory: list[dict[str, Any]], query: Any, limit: int
) -> list[dict[str, Any]]:
    """Retrieve prior input-ground-truth pairs without trajectory information."""
    if limit < 1:
        return []
    ranked = []
    for index, episode in enumerate(memory):
        trace = _similarity(query, episode, index, len(memory))
        ranked.append((trace["score"], episode["sequence_step"], episode, trace))
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]["experience_id"]))
    return [
        {
            "case_memory_id": episode["experience_id"],
            "prior_input": episode["query_summary"],
            "verified_ground_truth": episode["ground_truth"],
            "retrieval_trace": trace,
        }
        for _, _, episode, trace in ranked[:limit]
    ]


def _role(episode: dict[str, Any]) -> str:
    feedback = episode["feedback"]
    if feedback.get("exact_match") or episode["lesson"].get("improved_over_static_anchor"):
        return "helpful"
    return "corrective"


def retrieve_experience_memory(
    memory: list[dict[str, Any]], query: Any, limit: int
) -> list[dict[str, Any]]:
    """Retrieve a similarity-ranked helpful/corrective experience mixture."""
    if limit < 1 or not memory:
        return []
    ranked: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
    for index, episode in enumerate(memory):
        trace = _similarity(query, episode, index, len(memory))
        ranked.append((trace["score"], episode["sequence_step"], episode, trace))
    ranked.sort(key=lambda value: (-value[0], -value[1], value[2]["experience_id"]))
    helpful = [value for value in ranked if _role(value[2]) == "helpful"]
    corrective = [value for value in ranked if _role(value[2]) == "corrective"]
    chosen = []
    half = max(1, limit // 2)
    chosen.extend(helpful[:half])
    chosen.extend(corrective[:half])
    chosen_ids = {value[2]["experience_id"] for value in chosen}
    chosen.extend(value for value in ranked if value[2]["experience_id"] not in chosen_ids)
    chosen = sorted(chosen[:limit], key=lambda value: (-value[0], -value[1]))
    return [
        {
            "experience_id": episode["experience_id"],
            "prior_input": episode["query_summary"],
            "prior_prediction": episode["prediction"],
            "verified_ground_truth": episode["ground_truth"],
            "feedback": episode["feedback"],
            "trajectory_summary": episode["trajectory_summary"],
            "lesson": episode["lesson"],
            "contrast_role": _role(episode),
            "retrieval_trace": trace,
        }
        for _, _, episode, trace in chosen
    ]


def graph_records(episodes: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create an append-auditable experience graph from compressed episodes."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def node(node_id: str, node_type: str, attributes: dict[str, Any]) -> None:
        if node_id not in seen:
            nodes.append({"node_id": node_id, "node_type": node_type, "attributes": attributes})
            seen.add(node_id)

    for episode in episodes:
        run_id = f"run:{episode['experience_id']}"
        query_id = f"query:{episode['case_id']}"
        prediction_id = f"prediction:{episode['experience_id']}"
        outcome_id = f"outcome:{episode['experience_id']}"
        feedback_id = f"feedback:{episode['experience_id']}"
        lesson_id = f"lesson:{episode['experience_id']}"
        node(run_id, "execution", {"sequence_step": episode["sequence_step"], "task": episode["task"]})
        node(query_id, "query", {"case_id": episode["case_id"], "summary": episode["query_summary"]})
        node(prediction_id, "prediction", {"value": episode["prediction"]})
        node(outcome_id, "verified_outcome", {"value": episode["ground_truth"]})
        node(feedback_id, "feedback", episode["feedback"])
        node(lesson_id, "consolidated_lesson", episode["lesson"])
        edges.extend((
            {"source_id": run_id, "relation": "EXECUTED_ON", "target_id": query_id},
            {"source_id": run_id, "relation": "PRODUCED", "target_id": prediction_id},
            {"source_id": prediction_id, "relation": "EVALUATED_AGAINST", "target_id": outcome_id},
            {"source_id": run_id, "relation": "RECEIVED", "target_id": feedback_id},
            {"source_id": lesson_id, "relation": "CONSOLIDATED_FROM", "target_id": run_id},
        ))
        for evidence_id in episode["trajectory_summary"].get("selected_evidence_ids", []):
            evidence_node = f"evidence:{evidence_id}"
            node(evidence_node, "domain_episode", {"evidence_id": evidence_id})
            edges.append({"source_id": run_id, "relation": "SELECTED_EVIDENCE", "target_id": evidence_node})
        for experience_id in episode["trajectory_summary"].get("selected_experience_ids", []):
            memory_node = f"run:{experience_id}"
            node(memory_node, "prior_execution", {"experience_id": experience_id})
            edges.append({"source_id": run_id, "relation": "REUSED_EXPERIENCE", "target_id": memory_node})
    return nodes, edges


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")

