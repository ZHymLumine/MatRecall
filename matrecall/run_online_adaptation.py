"""Run six aligned prequential methods over the sampled online-adaptation streams.

Every task is sequential internally and tasks run concurrently.  Gold feedback
is joined only after a response row has been produced.  Adaptive methods may
then use that verified history for the next step in the same task sequence.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from matrecall.static_retrieval import canonical_json, digest
from matrecall.episodic_memory import (
    compress_experience,
    graph_records,
    retrieve_case_memory,
    retrieve_experience_memory,
    write_jsonl,
)
from matrecall.llm_client import ResponseError, complete


BASE_METHODS = (
    "llm_only",
    "static_multiview_rag",
    "react_static_memory",
    "rolling_context_icl",
    "dynamic_case_rag",
    "matrecall",
)
EPISODIC_ONLY_METHODS = ("case_episodic_only", "full_episodic_only")
METHODS = (*BASE_METHODS, *EPISODIC_ONLY_METHODS)
CASE_MEMORY_METHODS = ("dynamic_case_rag", "case_episodic_only")
FULL_MEMORY_METHODS = ("matrecall", "full_episodic_only")
STATIC_MEMORY_METHODS = (
    "static_multiview_rag",
    "react_static_memory",
    "rolling_context_icl",
    "dynamic_case_rag",
    "matrecall",
)
PLANNER_METHODS = ("react_static_memory", *FULL_MEMORY_METHODS)
RUN_PROTOCOL = "matrecall-online-adaptation-run"
# Planner prompt sets. "original" is used by the online-adaptation experiment and by the core and
# query-order memory ablations. "revised" is used by the episodic-only ablation, the retrieval-budget
# ablation and the MatText experiment. The two differ only in the planner instruction for
# static-evidence methods.
PROMPT_SETS = ("original", "revised")
PROMPT_REVISIONS = {
    "original": "online-original",
    "revised": "online-revised",
}
PRINT_LOCK = threading.Lock()

SYSTEM = """You are a materials prediction system in a prequential evaluation.
Predict the current query before its answer is revealed. Prior verified outcomes, when supplied,
belong only to earlier queries in the same task sequence. Never infer that the current query has
already been measured. Use only the supplied domain evidence and prior feedback. Return concise
structured JSON matching the schema and cite only supplied IDs."""

PLANNER_SYSTEM = """You are the evidence controller for MatRecall.
The current query answer is hidden. Select one physically relevant static retrieval view and, when
available, verified prior execution experiences. Prefer a mixture of helpful and corrective
experiences. The planner selects evidence; it does not predict the current answer."""

EPISODIC_ONLY_PLANNER_SYSTEM = """You are the experience-memory controller for MatRecall.
The current query answer is hidden. Select only verified earlier execution experiences that transfer
a compatible mechanism or strategy. Static evidence is unavailable, and the planner does not predict
the current answer."""


def prediction_schema(case: dict[str, Any]) -> dict[str, Any]:
    if "prediction_schema" in case:
        return case["prediction_schema"]
    kind = case["task_kind"]
    if kind == "regression":
        return {
            "type": "object", "properties": {"value": {"type": "number"}},
            "required": ["value"], "additionalProperties": False,
        }
    if kind == "binary_classification":
        return {
            "type": "object",
            "properties": {
                "value": {"type": "integer", "enum": [0, 1]},
                "probability_success": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["value", "probability_success"], "additionalProperties": False,
        }
    if kind == "multiclass_classification":
        return {
            "type": "object",
            "properties": {"value": {"type": "string", "enum": case["allowed_labels"]}},
            "required": ["value"], "additionalProperties": False,
        }
    if kind == "structured_prediction":
        return {
            "type": "object",
            "properties": {"value": {"type": "string", "minLength": 1, "maxLength": 120}},
            "required": ["value"], "additionalProperties": False,
        }
    raise ValueError(f"Unknown task kind: {kind}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line: {path}:{number}")
            rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _id_array(identifiers: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "maxItems": len(identifiers),
        "items": {"type": "string"},
    }
    if identifiers:
        schema["items"]["enum"] = identifiers
    return schema


def response_schema(
    case: dict[str, Any], evidence_ids: list[str], experience_ids: list[str]
) -> dict[str, Any]:
    step = {
        "type": "object",
        "properties": {
            "comparison": {"type": "string", "minLength": 1, "maxLength": 800},
            "evidence_ids": _id_array(evidence_ids),
            "experience_ids": _id_array(experience_ids),
            "implication": {"type": "string", "minLength": 1, "maxLength": 800},
        },
        "required": ["comparison", "evidence_ids", "experience_ids", "implication"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "analysis_process": {
                "type": "array", "minItems": 1, "maxItems": 4, "items": step,
            },
            "reasoning_results": {
                "type": "object",
                "properties": {
                    "prediction": prediction_schema(case),
                    "evidence_ids": _id_array(evidence_ids),
                    "experience_ids": _id_array(experience_ids),
                },
                "required": ["prediction", "evidence_ids", "experience_ids"],
                "additionalProperties": False,
            },
        },
        "required": ["analysis_process", "reasoning_results"],
        "additionalProperties": False,
    }


def _compact_reference(reference: dict[str, Any]) -> dict[str, Any]:
    observed = reference.get("observed_output", reference.get("observed_outcome"))
    return {
        "evidence_id": reference["evidence_id"],
        "input_summary": canonical_json(reference.get("input", {}))[:700],
        "observed_outcome": observed,
        "retrieval_trace": reference.get("retrieval_trace", {}),
        "graph_paths": reference.get("graph_paths", []),
    }


def planner_schema(
    case: dict[str, Any], experience_ids: list[str], evidence_budget: int,
    experience_budget: int,
) -> dict[str, Any]:
    evidence_ids = [row["evidence_id"] for row in case["references"]]
    selected_experience = _id_array(experience_ids)
    selected_experience["maxItems"] = min(experience_budget, len(experience_ids))
    if experience_ids:
        selected_experience["minItems"] = 1
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["predict", "inspect_view"]},
            "requested_view": {
                "type": "string", "enum": ["none", *list(case["reference_views"])],
            },
            "selected_evidence_ids": {
                "type": "array", "maxItems": evidence_budget,
                "items": {"type": "string", "enum": evidence_ids},
            },
            "selected_experience_ids": selected_experience,
            "comparison": {"type": "string", "minLength": 1, "maxLength": 800},
            "conflict_assessment": {"type": "string", "minLength": 1, "maxLength": 800},
            "memory_use": {"type": "string", "minLength": 1, "maxLength": 800},
        },
        "required": [
            "action", "requested_view", "selected_evidence_ids",
            "selected_experience_ids", "comparison", "conflict_assessment", "memory_use",
        ],
        "additionalProperties": False,
    }


def planner_messages(
    case: dict[str, Any], experience_candidates: list[dict[str, Any]],
    evidence_budget: int, experience_budget: int, *, prompt_set: str,
    include_static: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if prompt_set == "original":
        if not include_static:
            raise ValueError("Episodic-only planners exist only in the revised prompt set")
        choice = "Choose static evidence and verified prior experiences"
    elif prompt_set == "revised":
        choice = "Choose verified prior experiences" + (
            " and static evidence" if include_static else " without using static evidence"
        )
    else:
        raise ValueError(f"Unknown prompt set: {prompt_set}")
    payload = {
        "task": case["task_instruction"],
        "current_query": case["query_visible"],
        "verified_experience_catalog": experience_candidates,
        "instruction": (
            choice + " for the current prediction. "
            "The current query ground truth is unavailable."
        ),
    }
    if include_static:
        payload.update(
            static_anchor=case["deterministic_tool_estimate"],
            initial_evidence_ids=case["initial_evidence_ids"],
            available_retrieval_views=case["reference_views"],
            static_candidate_catalog=[
                _compact_reference(row) for row in case["references"]
            ],
        )
    ids = [row["experience_id"] for row in experience_candidates]
    schema = planner_schema(case, ids, evidence_budget, experience_budget)
    if not include_static:
        schema["properties"]["action"] = {"type": "string", "enum": ["predict"]}
        schema["properties"]["requested_view"] = {
            "type": "string", "enum": ["none"],
        }
        schema["properties"]["selected_evidence_ids"] = {
            "type": "array", "maxItems": 0, "items": {"type": "string"},
        }
    return [
        {
            "role": "system",
            "content": PLANNER_SYSTEM if include_static else EPISODIC_ONLY_PLANNER_SYSTEM,
        },
        {"role": "user", "content": canonical_json(payload)},
    ], schema


def select_static_evidence(
    case: dict[str, Any], planner: dict[str, Any] | None, evidence_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["evidence_id"]: row for row in case["references"]}
    ordered = [value for value in case["initial_evidence_ids"] if value in by_id]
    action, requested_view = "predict", "none"
    if planner is not None:
        action, requested_view = planner["action"], planner["requested_view"]
        for evidence_id in planner["selected_evidence_ids"]:
            if evidence_id not in by_id:
                raise ValueError(f"Planner selected unavailable evidence: {evidence_id}")
            if evidence_id not in ordered:
                ordered.append(evidence_id)
        if action == "inspect_view" and requested_view != "none":
            if requested_view not in case["reference_views"]:
                raise ValueError(f"Unknown retrieval view: {requested_view}")
            for evidence_id in case["reference_views"][requested_view]:
                if evidence_id in by_id and evidence_id not in ordered:
                    ordered.append(evidence_id)
    ordered = ordered[:evidence_budget]
    if not ordered:
        raise ValueError("No static evidence selected")
    observation = {
        "action": action,
        "requested_view": requested_view,
        "returned_evidence_ids": ordered,
        "returned_count": len(ordered),
    }
    return [by_id[value] for value in ordered], observation


def final_messages(
    *, case: dict[str, Any], method: str, references: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]], experiences: list[dict[str, Any]],
    planner: dict[str, Any] | None, observation: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    evidence_ids = [row["evidence_id"] for row in references]
    experience_ids = [
        row.get("experience_id", row.get("case_memory_id"))
        for row in [*prior_cases, *experiences]
    ]
    payload: dict[str, Any] = {
        "method": method,
        "task": case["task_instruction"],
        "current_query": case["query_visible"],
        "instruction": (
            "Predict the current query before its answer is revealed. Earlier feedback may be "
            "used only when supplied. Do not treat prior cases as the current answer."
        ),
    }
    if references:
        payload.update(
            deterministic_static_anchor=case["deterministic_tool_estimate"],
            static_evidence_tool_observation=observation,
            static_training_episodes=references,
        )
    if prior_cases:
        payload["verified_prior_case_memory"] = prior_cases
    if experiences:
        payload["verified_execution_experience_memory"] = experiences
    if planner is not None:
        payload["planner_record"] = planner
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ], response_schema(case, evidence_ids, experience_ids)


def validate_response(
    case: dict[str, Any], response: dict[str, Any], evidence_ids: set[str],
    experience_ids: set[str],
) -> None:
    used_evidence = set(response["reasoning_results"]["evidence_ids"])
    used_experience = set(response["reasoning_results"]["experience_ids"])
    for step in response["analysis_process"]:
        used_evidence.update(step["evidence_ids"])
        used_experience.update(step["experience_ids"])
    if not used_evidence <= evidence_ids or not used_experience <= experience_ids:
        raise ValueError("Response cited evidence or experience outside the supplied observation")
    prediction = response["reasoning_results"]["prediction"]
    if case["task_kind"] in ("regression", "scalar_regression"):
        if not math.isfinite(float(prediction["value"])):
            raise ValueError("Non-finite regression prediction")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")


def _method_uses_feedback(method: str) -> bool:
    return method in (
        "rolling_context_icl", *CASE_MEMORY_METHODS, *FULL_MEMORY_METHODS
    )


def run_task(
    *, task_record: dict[str, Any], release: Path, output: Path, method: str,
    call_api: Any, config_sha: str, evidence_budget: int, history_k: int,
    experience_k: int, model: str, model_revision: str, dry_run: bool,
    resume: bool, max_steps: int | None, prompt_set: str,
) -> dict[str, Any]:
    source = release / task_record["relative_path"]
    cases = read_jsonl(source / "cases.jsonl")
    gold_rows = read_jsonl(source / "gold.jsonl")
    order = read_jsonl(source / "order.jsonl")
    if max_steps is not None:
        cases, gold_rows, order = cases[:max_steps], gold_rows[:max_steps], order[:max_steps]
    if not (len(cases) == len(gold_rows) == len(order)):
        raise ValueError(f"Sequence length mismatch: {task_record['task']}")
    if any(c["case_id"] != g["case_id"] or c["case_id"] != o["case_id"] for c, g, o in zip(cases, gold_rows, order)):
        raise ValueError(f"Sequence alignment mismatch: {task_record['task']}")

    task_output = output / "test" / method / task_record["family"] / task_record["task"]
    prediction_path = task_output / "predictions.jsonl"
    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    if existing and not resume:
        raise FileExistsError(prediction_path)
    if len(existing) > len(cases):
        raise ValueError("Resume output is longer than the sequence")

    active_memory: list[dict[str, Any]] = []
    for index, row in enumerate(existing):
        case, gold, sequence = cases[index], gold_rows[index], order[index]
        if (
            row.get("sequence_step") != sequence["sequence_step"]
            or row.get("case_id") != case["case_id"]
            or row.get("case_sha256") != digest(case)
            or row.get("run_config_sha256") != config_sha
        ):
            raise ValueError(f"Resume prefix mismatch: {task_record['task']}/{index + 1}")
        if _method_uses_feedback(method):
            active_memory.append(compress_experience(
                case=case, gold=gold, result=row, sequence_id=sequence["sequence_id"],
                sequence_step=sequence["sequence_step"],
            ))

    for index in range(len(existing), len(cases)):
        case, gold, sequence = cases[index], gold_rows[index], order[index]
        started = time.monotonic()
        row: dict[str, Any] = {
            "protocol": case["protocol"],
            "sequential_run_protocol": RUN_PROTOCOL,
            "prompt_revision": PROMPT_REVISIONS[prompt_set],
            "case_id": case["case_id"],
            "case_sha256": digest(case),
            "task": case["task"],
            "task_kind": case["task_kind"],
            "family": task_record["family"],
            "method": method,
            "mode": method,
            "model": model,
            "model_revision": model_revision,
            "sequence_id": sequence["sequence_id"],
            "sequence_step": sequence["sequence_step"],
            "memory_size_before": len(active_memory),
            "run_config_sha256": config_sha,
        }
        planner = None
        planner_prompt = None
        prior_cases: list[dict[str, Any]] = []
        experience_candidates: list[dict[str, Any]] = []
        selected_experiences: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        observation: dict[str, Any] = {
            "action": "predict", "requested_view": "none",
            "returned_evidence_ids": [], "returned_count": 0,
        }
        try:
            if method == "rolling_context_icl":
                prior_cases = [
                    {
                        "case_memory_id": episode["experience_id"],
                        "prior_input": episode["query_summary"],
                        "verified_ground_truth": episode["ground_truth"],
                        "sequence_step": episode["sequence_step"],
                    }
                    for episode in active_memory[-history_k:]
                ]
            elif method in CASE_MEMORY_METHODS:
                prior_cases = retrieve_case_memory(active_memory, case["query_visible"], history_k)
            elif method in FULL_MEMORY_METHODS:
                experience_candidates = retrieve_experience_memory(
                    active_memory, case["query_visible"], experience_k
                )

            include_static = method in STATIC_MEMORY_METHODS
            if method != "llm_only":
                if method in PLANNER_METHODS:
                    planner_prompt, planner_schema_value = planner_messages(
                        case,
                        experience_candidates if method in FULL_MEMORY_METHODS else [],
                        evidence_budget, experience_k, prompt_set=prompt_set,
                        include_static=include_static,
                    )
                    if dry_run:
                        planner = {
                            "action": "inspect_view" if include_static else "predict",
                            "requested_view": (
                                case["retrieval"]["view"] if include_static else "none"
                            ),
                            "selected_evidence_ids": (
                                case["initial_evidence_ids"][:evidence_budget]
                                if include_static else []
                            ),
                            "selected_experience_ids": [
                                value["experience_id"] for value in experience_candidates[:experience_k]
                            ],
                            "comparison": "dry-run planner",
                            "conflict_assessment": "dry-run planner",
                            "memory_use": "dry-run planner",
                        }
                        planner_usage = None
                        planner_raw_response = None
                    else:
                        planner, _, planner_usage, planner_raw_response = call_api(
                            planner_prompt, planner_schema_value, planner=True
                        )
                    row.update(
                        planner_record=planner,
                        planner_prompt_sha256=digest(planner_prompt),
                        planner_usage=planner_usage,
                    )
                    if planner_raw_response is not None:
                        row["planner_raw_response"] = planner_raw_response
                if include_static:
                    references, observation = select_static_evidence(
                        case, planner, evidence_budget
                    )

            if method in FULL_MEMORY_METHODS and planner is not None:
                by_id = {value["experience_id"]: value for value in experience_candidates}
                for experience_id in planner["selected_experience_ids"]:
                    if experience_id not in by_id:
                        raise ValueError(f"Planner selected unavailable experience: {experience_id}")
                    selected_experiences.append(by_id[experience_id])

            messages, schema = final_messages(
                case=case, method=method, references=references, prior_cases=prior_cases,
                experiences=selected_experiences, planner=planner, observation=observation,
            )
            evidence_ids = {value["evidence_id"] for value in references}
            experience_ids = {
                value.get("experience_id", value.get("case_memory_id"))
                for value in [*prior_cases, *selected_experiences]
            }
            row.update(
                selected_evidence_ids=sorted(evidence_ids),
                selected_experience_ids=sorted(value for value in experience_ids if value),
                agent_trace=[
                    {"step": 1, "kind": "planner", "record": planner},
                    {"step": 2, "kind": "static_tool_observation", "record": observation},
                    {"step": 3, "kind": "experience_observation", "record": selected_experiences or prior_cases},
                ],
                final_prompt_sha256=digest(messages),
            )
            if dry_run:
                row.update(
                    status="dry_run",
                    planner_messages=planner_prompt,
                    final_messages=messages,
                    final_schema=schema,
                )
            else:
                response, raw, final_usage, raw_response = call_api(messages, schema, planner=False)
                validate_response(case, response, evidence_ids, {value for value in experience_ids if value})
                row.update(
                    status="success", response=response, raw_content=raw,
                    usage={"planner": row.get("planner_usage"), "final": final_usage},
                    raw_response=raw_response,
                )
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, ResponseError):
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.monotonic() - started

        experience = compress_experience(
            case=case, gold=gold, result=row, sequence_id=sequence["sequence_id"],
            sequence_step=sequence["sequence_step"],
        )
        row.update(
            feedback=experience["feedback"],
            static_anchor_feedback=experience["static_anchor_feedback"],
            ground_truth_revealed_after_prediction=True,
        )
        if _method_uses_feedback(method):
            active_memory.append(experience)
        row["memory_size_after"] = len(active_memory)
        _append_jsonl(prediction_path, row)
        with PRINT_LOCK:
            print(canonical_json({
                "event": "progress", "method": method, "task": case["task"],
                "step": sequence["sequence_step"], "total": len(cases),
                "status": row["status"],
            }), flush=True)

    if _method_uses_feedback(method):
        write_jsonl(task_output / "verified_feedback_memory.jsonl", active_memory)
    if method in FULL_MEMORY_METHODS:
        nodes, edges = graph_records(active_memory)
        write_jsonl(task_output / "experience_graph_nodes.jsonl", nodes)
        write_jsonl(task_output / "experience_graph_edges.jsonl", edges)
    rows = read_jsonl(prediction_path)
    return {
        "method": method,
        "family": task_record["family"],
        "task": task_record["task"],
        "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "dry_run": sum(row["status"] == "dry_run" for row in rows),
        "predictions_sha256": file_sha256(prediction_path),
    }


def combine_outputs(
    *, release: Path, output: Path, methods: list[str], task_records: list[dict[str, Any]],
    max_steps: int | None,
) -> None:
    for method in methods:
        for family in sorted({row["family"] for row in task_records}):
            predictions, gold = [], []
            for task in [row for row in task_records if row["family"] == family]:
                predictions.extend(read_jsonl(
                    output / "test" / method / family / task["task"] / "predictions.jsonl"
                ))
                values = read_jsonl(release / task["relative_path"] / "gold.jsonl")
                gold.extend(values[:max_steps] if max_steps is not None else values)
            target = output / "combined" / method / family
            write_jsonl(target / "predictions.jsonl", predictions)
            write_jsonl(target / "gold.jsonl", gold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=list(BASE_METHODS)
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Optional audited task subset for supplemental runs.",
    )
    parser.add_argument(
        "--prompt-set", choices=PROMPT_SETS, required=True,
        help="Planner prompt set; see PROMPT_SETS for which runs used which.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-tokens", type=int, default=1024)
    parser.add_argument("--evidence-budget", type=int, default=8)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--experience-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--allow-unescaped-control-chars", action="store_true")
    parser.add_argument("--task-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-steps-per-task", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_tokens, args.planner_max_tokens, args.evidence_budget, args.history_k,
        args.experience_k, args.task_concurrency, args.timeout,
    ) < 1:
        raise ValueError("All budgets and concurrency values must be positive")
    if args.max_steps_per_task is not None and args.max_steps_per_task < 1:
        raise ValueError("--max-steps-per-task must be positive")
    if args.prompt_set == "original" and set(args.methods) & set(EPISODIC_ONLY_METHODS):
        raise ValueError("Episodic-only methods require --prompt-set revised")

    release_manifest = json.loads((args.release / "manifest.json").read_text(encoding="utf-8"))
    if release_manifest.get("status") != "frozen" or release_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task sequential release")
    all_task_records = release_manifest["tasks"]
    if args.tasks:
        requested = set(args.tasks)
        available = {row["task"] for row in all_task_records}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown task subset: {sorted(unknown)}")
        task_records = [row for row in all_task_records if row["task"] in requested]
    else:
        task_records = all_task_records
    config = {
        "run_protocol": RUN_PROTOCOL,
        "prompt_revision": PROMPT_REVISIONS[args.prompt_set],
        "model": args.model,
        "model_revision": args.model_revision,
        "methods": args.methods,
        "tasks": [row["task"] for row in task_records],
        "sampling": {
            "max_tokens": args.max_tokens, "planner_max_tokens": args.planner_max_tokens,
            "seed": args.seed, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "presence_penalty": args.presence_penalty,
            "disable_thinking": args.disable_thinking,
        },
        "memory": {
            "evidence_budget": args.evidence_budget, "history_k": args.history_k,
            "experience_k": args.experience_k,
        },
        "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
        "max_steps_per_task": args.max_steps_per_task,
        "dry_run": args.dry_run,
    }
    config_sha = digest(config)
    manifest_path = args.output / "run_manifest.json"
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("run_config_sha256") != config_sha:
            raise ValueError("Resume run configuration mismatch")
    else:
        manifest_path.write_text(
            json.dumps({**config, "run_config_sha256": config_sha}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def call_api(messages: list[dict[str, str]], schema: dict[str, Any], *, planner: bool):
        return complete(
            messages, schema, model=args.model, base_url=args.base_url, api_key=None,
            max_tokens=args.planner_max_tokens if planner else args.max_tokens,
            seed=args.seed, timeout=args.timeout, response_format="json_schema",
            disable_thinking=args.disable_thinking, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, reasoning_effort=None,
            presence_penalty=args.presence_penalty,
            allow_unescaped_control_chars=args.allow_unescaped_control_chars,
        )

    summaries: list[dict[str, Any]] = []
    for method in args.methods:
        with ThreadPoolExecutor(max_workers=min(args.task_concurrency, len(task_records))) as executor:
            futures = {
                executor.submit(
                    run_task, task_record=record, release=args.release, output=args.output,
                    method=method, call_api=call_api, config_sha=config_sha,
                    evidence_budget=args.evidence_budget, history_k=args.history_k,
                    experience_k=args.experience_k, model=args.model,
                    model_revision=args.model_revision, dry_run=args.dry_run,
                    resume=args.resume, max_steps=args.max_steps_per_task,
                    prompt_set=args.prompt_set,
                ): record["task"]
                for record in task_records
            }
            for future in as_completed(futures):
                summaries.append(future.result())

    combine_outputs(
        release=args.release, output=args.output, methods=args.methods,
        task_records=task_records, max_steps=args.max_steps_per_task,
    )
    completion = {
        "status": "dry_run" if args.dry_run else (
            "completed_with_recorded_response_failures"
            if any(row["failed"] for row in summaries) else "completed"
        ),
        "run_config_sha256": config_sha,
        "task_count": len(task_records),
        "source_task_count": len(all_task_records),
        "method_count": len(args.methods),
        "methods": args.methods,
        "counts": summaries,
    }
    (args.output / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json({
        "event": "complete", "status": completion["status"],
        "methods": len(args.methods), "tasks": len(task_records),
    }), flush=True)


if __name__ == "__main__":
    main()
