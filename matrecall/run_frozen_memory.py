"""Evaluate six methods against one immutable online-adaptation experience snapshot.

This runner never opens transfer gold files and never writes experience
memory.  Gold is joined by a separate post-prediction preparation step.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import time
from typing import Any

from matrecall.static_retrieval import canonical_json, digest
from matrecall.episodic_memory import (
    retrieve_case_memory,
    retrieve_experience_memory,
    write_jsonl,
)
from matrecall.llm_client import ResponseError, complete
from matrecall.run_online_adaptation import (
    file_sha256,
    planner_schema,
    read_jsonl,
    response_schema,
    select_static_evidence,
    validate_response,
)


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
RUN_PROTOCOL = "matrecall-frozen-memory-run"
# Prompt sets. "original" is used by the memory-retention experiment and supports only same-task
# memory. "revised" adds the cross-task transfer instruction and source/target task labels on
# memory items; it is used by the episodic-only retention ablation and by cross-task transfer.
PROMPT_SETS = ("original", "revised")
PROMPT_REVISIONS = {
    "original": "frozen-original",
    "revised": "frozen-revised",
}
PRINT_LOCK = threading.Lock()

SYSTEM = """You are a materials prediction system in a frozen-memory transfer evaluation.
The current query answer is hidden. Any verified cases or execution experiences were completed
during an earlier exposure phase and are now immutable. Use only supplied evidence and memory.
Do not assume that a prior case is identical to the current query. Return concise structured JSON
matching the schema and cite only supplied IDs."""

PLANNER_SYSTEM = """You are the evidence controller for frozen-memory MatRecall.
The current query answer is hidden and the exposure memory is read-only. Select physically
relevant static evidence and, when available, prior verified execution experiences. Prefer a
helpful/corrective mixture. The planner selects evidence; it does not predict the answer."""

EPISODIC_ONLY_PLANNER_SYSTEM = """You are the experience-memory controller for frozen-memory MatRecall.
The current query answer is hidden and the exposure memory is read-only. Select only prior verified
execution experiences that transfer a compatible mechanism or strategy. Static evidence is unavailable,
and the planner does not predict the answer."""


def _compact_reference(reference: dict[str, Any]) -> dict[str, Any]:
    observed = reference.get("observed_output", reference.get("observed_outcome"))
    return {
        "evidence_id": reference["evidence_id"],
        "input_summary": canonical_json(reference.get("input", {}))[:700],
        "observed_outcome": observed,
        "retrieval_trace": reference.get("retrieval_trace", {}),
        "graph_paths": reference.get("graph_paths", []),
    }


def planner_messages(
    case: dict[str, Any], experiences: list[dict[str, Any]],
    evidence_budget: int, experience_budget: int, *, prompt_set: str,
    include_static: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if prompt_set == "original":
        if not include_static:
            raise ValueError("Episodic-only planners exist only in the revised prompt set")
        choice = "Choose static evidence and frozen verified exposure experiences"
    elif prompt_set == "revised":
        choice = "Choose frozen verified exposure experiences" + (
            " and static evidence" if include_static else " without using static evidence"
        )
    else:
        raise ValueError(f"Unknown prompt set: {prompt_set}")
    payload = {
        "task": case["task_instruction"],
        "current_query": case["query_visible"],
        "frozen_verified_experience_catalog": experiences,
        "instruction": (
            choice + ". "
            "No transfer-query ground truth is available and memory cannot be modified."
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
    identifiers = [row["experience_id"] for row in experiences]
    schema = planner_schema(case, identifiers, evidence_budget, experience_budget)
    if not include_static:
        schema["properties"]["action"] = {"type": "string", "enum": ["predict"]}
        schema["properties"]["requested_view"] = {
            "type": "string", "enum": ["none"],
        }
        schema["properties"]["selected_evidence_ids"] = {
            "type": "array", "maxItems": 0, "items": {"type": "string"},
        }
    for field in ("comparison", "conflict_assessment", "memory_use"):
        schema["properties"][field]["maxLength"] = 240
    return [
        {
            "role": "system",
            "content": PLANNER_SYSTEM if include_static else EPISODIC_ONLY_PLANNER_SYSTEM,
        },
        {"role": "user", "content": canonical_json(payload)},
    ], schema


def final_messages(
    *, case: dict[str, Any], method: str, references: list[dict[str, Any]],
    prior_cases: list[dict[str, Any]], experiences: list[dict[str, Any]],
    planner: dict[str, Any] | None, observation: dict[str, Any], prompt_set: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if prompt_set == "original":
        cross_task = ""
    elif prompt_set == "revised":
        cross_task = (
            "A memory item can come from a different source task; transfer only compatible "
            "mechanisms or strategy lessons and never copy an incompatible output schema. "
        )
    else:
        raise ValueError(f"Unknown prompt set: {prompt_set}")
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
            "Predict independently using the immutable exposure memory supplied here. "
            + cross_task
            + "The current answer is unavailable; do not write or infer new feedback."
        ),
    }
    if references:
        payload.update(
            deterministic_static_anchor=case["deterministic_tool_estimate"],
            static_evidence_tool_observation=observation,
            static_training_episodes=[_compact_reference(row) for row in references],
        )
    if prior_cases:
        payload["frozen_verified_prior_cases"] = prior_cases
    if experiences:
        payload["frozen_verified_execution_experiences"] = experiences
    if planner is not None:
        payload["planner_record"] = planner
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": canonical_json(payload)},
    ], response_schema(case, evidence_ids, [value for value in experience_ids if value])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")


def run_task(
    *, task_record: dict[str, Any], release: Path, output: Path, method: str,
    call_api: Any, config_sha: str, evidence_budget: int, history_k: int,
    experience_k: int, model: str, model_revision: str, dry_run: bool,
    resume: bool, max_cases: int | None, prompt_set: str,
) -> dict[str, Any]:
    source = release / task_record["relative_path"]
    cases = read_jsonl(source / "cases.jsonl")
    transfer_index = read_jsonl(source / "transfer_index.jsonl")
    if max_cases is not None:
        cases, transfer_index = cases[:max_cases], transfer_index[:max_cases]
    if len(cases) != len(transfer_index):
        raise ValueError(f"Transfer alignment mismatch: {task_record['task']}")
    if any(
        case["case_id"] != index["case_id"]
        for case, index in zip(cases, transfer_index)
    ):
        raise ValueError(f"Transfer case/index mismatch: {task_record['task']}")
    memory_path = source / "frozen_exposure_memory.jsonl"
    memory_sha = file_sha256(memory_path)
    if memory_sha != task_record["frozen_memory_sha256"]:
        raise ValueError(f"Frozen memory hash mismatch: {task_record['task']}")
    frozen_memory = read_jsonl(memory_path)
    expected_memory_cases = task_record.get(
        "memory_cases", task_record.get("exposure_cases")
    )
    if len(frozen_memory) != expected_memory_cases:
        raise ValueError(f"Frozen memory size mismatch: {task_record['task']}")
    frozen_memory.sort(key=lambda row: row["sequence_step"])

    evaluation_unit = task_record.get("pair_id", task_record["task"])
    task_output = output / "test" / method / task_record["family"] / evaluation_unit
    prediction_path = task_output / "predictions.jsonl"
    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    if existing and not resume:
        raise FileExistsError(prediction_path)
    if len(existing) > len(cases):
        raise ValueError("Resume output is longer than transfer cohort")
    for offset, row in enumerate(existing):
        if (
            row.get("case_id") != cases[offset]["case_id"]
            or row.get("case_sha256") != digest(cases[offset])
            or row.get("run_config_sha256") != config_sha
            or row.get("frozen_memory_sha256") != memory_sha
            or row.get("memory_write_performed") is not False
        ):
            raise ValueError(f"Resume prefix mismatch: {task_record['task']}/{offset + 1}")

    for offset in range(len(existing), len(cases)):
        case, index = cases[offset], transfer_index[offset]
        started = time.monotonic()
        row: dict[str, Any] = {
            "protocol": case["protocol"],
            "transfer_run_protocol": RUN_PROTOCOL,
            "prompt_revision": PROMPT_REVISIONS[prompt_set],
            "case_id": case["case_id"],
            "case_sha256": digest(case),
            "task": case["task"],
            "target_task": task_record.get("target_task", case["task"]),
            "memory_source_task": task_record.get("source_task", case["task"]),
            "memory_relation": task_record.get("memory_relation", "same_task"),
            "pair_id": task_record.get("pair_id"),
            "task_kind": case["task_kind"],
            "family": task_record["family"],
            "method": method,
            "mode": method,
            "model": model,
            "model_revision": model_revision,
            "transfer_index": index["transfer_index"],
            "transfer_group_id": index["transfer_group_id"],
            "grouping_is_proxy": index["grouping_is_proxy"],
            "exposure_case_overlap": index["exposure_case_overlap"],
            "exposure_group_overlap": index["exposure_group_overlap"],
            "frozen_memory_size": len(frozen_memory),
            "frozen_memory_sha256": memory_sha,
            "memory_write_performed": False,
            "transfer_feedback_revealed": False,
            "run_config_sha256": config_sha,
        }
        planner = None
        planner_prompt = None
        prior_cases: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        selected_experiences: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        observation = {
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
                        "exposure_sequence_step": episode["sequence_step"],
                    }
                    for episode in frozen_memory[-history_k:]
                ]
            elif method in CASE_MEMORY_METHODS:
                prior_cases = retrieve_case_memory(
                    frozen_memory, case["query_visible"], history_k
                )
            elif method in FULL_MEMORY_METHODS:
                candidates = retrieve_experience_memory(
                    frozen_memory, case["query_visible"], experience_k
                )

            if prompt_set == "revised":
                memory_source_task = task_record.get("source_task", case["task"])
                target_task = task_record.get("target_task", case["task"])
                for memory_row in [*prior_cases, *candidates]:
                    memory_row["source_task"] = memory_source_task
                    memory_row["target_task"] = target_task

            include_static = method in STATIC_MEMORY_METHODS
            if method != "llm_only":
                if method in PLANNER_METHODS:
                    planner_prompt, planner_schema_value = planner_messages(
                        case, candidates if method in FULL_MEMORY_METHODS else [],
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
                                value["experience_id"] for value in candidates[:experience_k]
                            ],
                            "comparison": "dry-run planner",
                            "conflict_assessment": "dry-run planner",
                            "memory_use": "dry-run planner",
                        }
                        planner_usage = None
                        planner_raw_response = None
                        planner_attempts = []
                    else:
                        (
                            planner, _, planner_usage, planner_raw_response,
                            planner_attempts,
                        ) = call_api(planner_prompt, planner_schema_value, planner=True)
                    row.update(
                        planner_record=planner,
                        planner_prompt_sha256=digest(planner_prompt),
                        planner_usage=planner_usage,
                        planner_attempts=planner_attempts,
                    )
                    if planner_raw_response is not None:
                        row["planner_raw_response"] = planner_raw_response
                if include_static:
                    references, observation = select_static_evidence(
                        case, planner, evidence_budget
                    )

            if method in FULL_MEMORY_METHODS and planner is not None:
                by_id = {value["experience_id"]: value for value in candidates}
                for experience_id in planner["selected_experience_ids"]:
                    if experience_id not in by_id:
                        raise ValueError(
                            f"Planner selected unavailable experience: {experience_id}"
                        )
                    selected_experiences.append(by_id[experience_id])

            messages, schema = final_messages(
                case=case, method=method, references=references,
                prior_cases=prior_cases, experiences=selected_experiences,
                planner=planner, observation=observation, prompt_set=prompt_set,
            )
            evidence_ids = {value["evidence_id"] for value in references}
            experience_ids = {
                value.get("experience_id", value.get("case_memory_id"))
                for value in [*prior_cases, *selected_experiences]
            }
            row.update(
                selected_evidence_ids=sorted(evidence_ids),
                selected_experience_ids=sorted(
                    value for value in experience_ids if value
                ),
                agent_trace=[
                    {"step": 1, "kind": "planner", "record": planner},
                    {"step": 2, "kind": "static_tool_observation", "record": observation},
                    {
                        "step": 3, "kind": "frozen_memory_observation",
                        "record": selected_experiences or prior_cases,
                    },
                ],
                final_prompt_sha256=digest(messages),
            )
            if dry_run:
                row.update(
                    status="dry_run", planner_messages=planner_prompt,
                    final_messages=messages, final_schema=schema,
                )
            else:
                response, raw, usage, raw_response, final_attempts = call_api(
                    messages, schema, planner=False
                )
                validate_response(
                    case, response, evidence_ids,
                    {value for value in experience_ids if value},
                )
                row.update(
                    status="success", response=response, raw_content=raw,
                    usage={"planner": row.get("planner_usage"), "final": usage},
                    raw_response=raw_response, final_attempts=final_attempts,
                )
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, ResponseError):
                row["raw_response"] = exc.raw_response
                row["retry_attempts"] = getattr(exc, "retry_attempts", [])
        row["elapsed_seconds"] = time.monotonic() - started
        if any(key in row for key in ("feedback", "ground_truth", "target")):
            raise AssertionError("Transfer prediction row contains forbidden target feedback")
        _append_jsonl(prediction_path, row)
        with PRINT_LOCK:
            print(canonical_json({
                "event": "progress", "method": method, "task": case["task"],
                "transfer_index": index["transfer_index"], "total": len(cases),
                "status": row["status"],
            }), flush=True)

    rows = read_jsonl(prediction_path)
    if file_sha256(memory_path) != memory_sha:
        raise ValueError(f"Frozen memory changed during evaluation: {task_record['task']}")
    return {
        "method": method,
        "family": task_record["family"],
        "task": task_record["task"],
        "target_task": task_record.get("target_task", task_record["task"]),
        "source_task": task_record.get("source_task", task_record["task"]),
        "memory_relation": task_record.get("memory_relation", "same_task"),
        "pair_id": task_record.get("pair_id"),
        "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "dry_run": sum(row["status"] == "dry_run" for row in rows),
        "retried_success": sum(
            row.get("status") == "success" and len(row.get("final_attempts", [])) > 1
            for row in rows
        ),
        "predictions_sha256": file_sha256(prediction_path),
        "frozen_memory_sha256_before_after": memory_sha,
        "memory_writes": 0,
    }


def combine_predictions(
    *, output: Path, methods: list[str], task_records: list[dict[str, Any]],
    max_cases: int | None,
) -> None:
    for method in methods:
        for family in sorted({row["family"] for row in task_records}):
            predictions = []
            for task in [row for row in task_records if row["family"] == family]:
                values = read_jsonl(
                    output / "test" / method / family /
                    task.get("pair_id", task["task"]) / "predictions.jsonl"
                )
                predictions.extend(values[:max_cases] if max_cases is not None else values)
            write_jsonl(output / "combined" / method / family / "predictions.jsonl", predictions)


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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--evidence-budget", type=int, default=8)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--experience-k", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--allow-unescaped-control-chars", action="store_true")
    parser.add_argument("--task-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-cases-per-task", type=int)
    parser.add_argument(
        "--prompt-set", choices=PROMPT_SETS, required=True,
        help="Prompt set; see PROMPT_SETS for which runs used which.",
    )
    parser.add_argument(
        "--exclude-task", action="append", default=[],
        help="Exclude a task from an additive run without altering the frozen release.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_tokens, args.planner_max_tokens, args.evidence_budget,
        args.history_k, args.experience_k, args.max_attempts,
        args.task_concurrency, args.timeout,
    ) < 1:
        raise ValueError("All budgets, retries, concurrency, and timeout must be positive")
    if args.max_cases_per_task is not None and args.max_cases_per_task < 1:
        raise ValueError("--max-cases-per-task must be positive")

    release_manifest = json.loads((args.release / "manifest.json").read_text())
    if release_manifest.get("status") != "frozen":
        raise ValueError("Expected a frozen transfer release")
    if release_manifest.get("release_id") not in {
        "matrecall-memory-retention",
        "matrecall-cross-task-transfer",
    }:
        raise ValueError("Unexpected transfer release protocol")
    if args.prompt_set == "original":
        if set(args.methods) & set(EPISODIC_ONLY_METHODS):
            raise ValueError("Episodic-only methods require --prompt-set revised")
        if any(
            row.get("memory_relation", "same_task") != "same_task"
            for row in release_manifest["tasks"]
        ):
            raise ValueError("Cross-task memory requires --prompt-set revised")
    all_task_records = release_manifest["tasks"]
    excluded_tasks = set(args.exclude_task)
    available_tasks = {row["task"] for row in all_task_records}
    unknown_tasks = excluded_tasks - available_tasks
    if unknown_tasks:
        raise ValueError(f"Unknown task exclusions: {sorted(unknown_tasks)}")
    task_records = [
        row for row in all_task_records if row["task"] not in excluded_tasks
    ]
    if not task_records:
        raise ValueError("Task exclusions removed every transfer task")
    config = {
        "run_protocol": RUN_PROTOCOL,
        "prompt_revision": PROMPT_REVISIONS[args.prompt_set],
        "model": args.model,
        "model_revision": args.model_revision,
        "methods": args.methods,
        "tasks": [row.get("pair_id", row["task"]) for row in task_records],
        "excluded_tasks": sorted(excluded_tasks),
        "sampling": {
            "max_tokens": args.max_tokens,
            "planner_max_tokens": args.planner_max_tokens,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "disable_thinking": args.disable_thinking,
            "max_attempts": args.max_attempts,
        },
        "memory": {
            "policy": "read_only_fixed_snapshot",
            "evidence_budget": args.evidence_budget,
            "history_k": args.history_k,
            "experience_k": args.experience_k,
            "writes_per_transfer_query": 0,
        },
        "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
        "max_cases_per_task": args.max_cases_per_task,
        "dry_run": args.dry_run,
    }
    config_sha = digest(config)
    manifest_path = args.output / "run_manifest.json"
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_config_sha256") != config_sha:
            raise ValueError("Resume run configuration mismatch")
    else:
        manifest_path.write_text(
            json.dumps(
                {**config, "run_config_sha256": config_sha},
                ensure_ascii=False, indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

    def call_api(messages: list[dict[str, str]], schema: dict[str, Any], *, planner: bool):
        attempts = []
        current_messages = list(messages)
        for attempt in range(1, args.max_attempts + 1):
            try:
                response, raw, usage, raw_response = complete(
                    current_messages, schema, model=args.model,
                    base_url=args.base_url, api_key=None,
                    max_tokens=args.planner_max_tokens if planner else args.max_tokens,
                    seed=args.seed + attempt - 1, timeout=args.timeout,
                    response_format="json_schema",
                    disable_thinking=args.disable_thinking,
                    temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, reasoning_effort=None,
                    presence_penalty=args.presence_penalty,
                    allow_unescaped_control_chars=args.allow_unescaped_control_chars,
                )
                attempts.append({"attempt": attempt, "status": "success"})
                return response, raw, usage, raw_response, attempts
            except ResponseError as exc:
                attempts.append({
                    "attempt": attempt, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if attempt >= args.max_attempts:
                    exc.retry_attempts = attempts
                    raise
                current_messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "The previous response was invalid or truncated. Return only concise "
                            "JSON matching the schema; do not add extra explanation."
                        ),
                    },
                ]
        raise AssertionError("Unreachable retry loop")

    summaries = []
    for method in args.methods:
        with ThreadPoolExecutor(
            max_workers=min(args.task_concurrency, len(task_records))
        ) as executor:
            futures = {
                executor.submit(
                    run_task,
                    task_record=record, release=args.release, output=args.output,
                    method=method, call_api=call_api, config_sha=config_sha,
                    evidence_budget=args.evidence_budget, history_k=args.history_k,
                    experience_k=args.experience_k, model=args.model,
                    model_revision=args.model_revision, dry_run=args.dry_run,
                    resume=args.resume, max_cases=args.max_cases_per_task,
                    prompt_set=args.prompt_set,
                ): record["task"]
                for record in task_records
            }
            for future in as_completed(futures):
                summaries.append(future.result())

    combine_predictions(
        output=args.output, methods=args.methods, task_records=task_records,
        max_cases=args.max_cases_per_task,
    )
    completion = {
        "status": "dry_run" if args.dry_run else (
            "completed_with_recorded_response_failures"
            if any(row["failed"] for row in summaries) else "completed"
        ),
        "run_config_sha256": config_sha,
        "task_count": len(task_records),
        "source_task_count": len(all_task_records),
        "excluded_tasks": sorted(excluded_tasks),
        "transfer_cases_per_method": sum(
            min(row["transfer_cases"], args.max_cases_per_task)
            if args.max_cases_per_task is not None else row["transfer_cases"]
            for row in task_records
        ),
        "method_count": len(args.methods),
        "methods": args.methods,
        "memory_policy": "read_only_fixed_snapshot",
        "memory_writes": 0,
        "transfer_feedback_reveals": 0,
        "counts": summaries,
    }
    (args.output / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json({
        "event": "complete", "status": completion["status"],
        "methods": len(args.methods), "tasks": len(task_records),
        "cases_per_method": completion["transfer_cases_per_method"],
    }), flush=True)


if __name__ == "__main__":
    main()
