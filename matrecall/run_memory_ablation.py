"""Run controlled episodic-memory ablations.

All ``react_*`` and ``full_*`` methods use the same planner, static evidence,
model, prompt budget, and retry policy.  They differ only in writable-memory
content or feedback timing, making memory attribution reviewable.
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
    compress_experience,
    graph_records,
    retrieve_case_memory,
    retrieve_experience_memory,
    write_jsonl,
)
from matrecall.llm_client import ResponseError, complete
from matrecall.run_online_adaptation import (
    PROMPT_REVISIONS as PLANNER_PROMPT_REVISIONS,
    PROMPT_SETS,
    _append_jsonl,
    _compact_reference,
    _id_array,
    file_sha256,
    planner_messages,
    read_jsonl,
    select_static_evidence,
    validate_response,
)


METHODS = (
    "llm_only",
    "static_multiview_rag",
    "react_static_memory",
    "rolling_context_icl",
    "dynamic_case_rag",
    "react_case_only",
    "react_prediction_feedback",
    "react_trajectory_feedback",
    "full_score_only",
    "full_delayed_feedback",
    "matrecall",
)
ORDER_ROBUSTNESS_METHODS = (
    "react_static_memory", "react_case_only", "matrecall",
)
RUN_PROTOCOL = "matrecall-memory-ablation-run"
PROMPT_REVISION = "controlled-memory-ablation"
PRINT_LOCK = threading.Lock()

SYSTEM = """You are a materials prediction system in a prequential evaluation.
Predict the current query before its answer is revealed. Earlier verified feedback may be used
only when explicitly supplied. Return concise structured JSON matching the schema. Cite only
supplied evidence and memory IDs."""


METHOD_POLICIES: dict[str, dict[str, Any]] = {
    "llm_only": {"controller": "direct", "memory": "none"},
    "static_multiview_rag": {"controller": "fixed_static", "memory": "none"},
    "react_static_memory": {"controller": "react", "memory": "none"},
    "rolling_context_icl": {"controller": "fixed_static", "memory": "rolling_case"},
    "dynamic_case_rag": {"controller": "fixed_static", "memory": "retrieved_case"},
    "react_case_only": {"controller": "react", "memory": "case_only"},
    "react_prediction_feedback": {
        "controller": "react", "memory": "prediction_feedback",
    },
    "react_trajectory_feedback": {
        "controller": "react", "memory": "trajectory_feedback",
    },
    "full_score_only": {"controller": "react", "memory": "score_only"},
    "full_delayed_feedback": {"controller": "react", "memory": "full_delayed"},
    "matrecall": {"controller": "react", "memory": "full"},
}


def compact_response_schema(
    case: dict[str, Any], evidence_ids: list[str], experience_ids: list[str]
) -> dict[str, Any]:
    """Bound response verbosity identically across every ablation method."""
    step = {
        "type": "object",
        "properties": {
            "comparison": {"type": "string", "minLength": 1, "maxLength": 400},
            "evidence_ids": _id_array(evidence_ids),
            "experience_ids": _id_array(experience_ids),
            "implication": {"type": "string", "minLength": 1, "maxLength": 400},
        },
        "required": ["comparison", "evidence_ids", "experience_ids", "implication"],
        "additionalProperties": False,
    }
    from matrecall.run_online_adaptation import prediction_schema
    return {
        "type": "object",
        "properties": {
            "analysis_process": {
                "type": "array", "minItems": 1, "maxItems": 2, "items": step,
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
        "memory_policy": METHOD_POLICIES[method],
        "task": case["task_instruction"],
        "current_query": case["query_visible"],
        "instruction": (
            "Predict the current query before its answer is revealed. Keep reasoning concise. "
            "Do not treat any prior case as the current answer."
        ),
    }
    if references:
        payload.update(
            deterministic_static_anchor=case["deterministic_tool_estimate"],
            static_evidence_tool_observation=observation,
            static_training_episodes=[_compact_reference(row) for row in references],
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
    ], compact_response_schema(case, evidence_ids, [value for value in experience_ids if value])


def project_episode(episode: dict[str, Any], policy: str) -> dict[str, Any]:
    """Expose exactly the fields permitted by one controlled memory policy."""
    base = {
        "experience_id": episode["experience_id"],
        "prior_input": episode["query_summary"],
        "sequence_step": episode["sequence_step"],
    }
    if policy == "case_only":
        return {**base, "verified_ground_truth": episode["ground_truth"]}
    if policy == "prediction_feedback":
        return {
            **base,
            "prior_prediction": episode["prediction"],
            "verified_ground_truth": episode["ground_truth"],
            "feedback": episode["feedback"],
        }
    if policy == "trajectory_feedback":
        return {
            **project_episode(episode, "prediction_feedback"),
            "trajectory_summary": episode["trajectory_summary"],
        }
    if policy == "score_only":
        score_feedback = {
            key: episode["feedback"].get(key)
            for key in ("metric_name", "score", "loss", "exact_match")
        }
        score_lesson = {
            key: episode["lesson"].get(key)
            for key in (
                "improved_over_static_anchor",
                "score_delta_vs_static_anchor",
                "response_succeeded",
            )
        }
        return {
            **base,
            "prior_prediction": episode["prediction"],
            "feedback": score_feedback,
            "trajectory_summary": episode["trajectory_summary"],
            "lesson": score_lesson,
        }
    if policy in ("full", "full_delayed"):
        return {
            **base,
            "prior_prediction": episode["prediction"],
            "verified_ground_truth": episode["ground_truth"],
            "feedback": episode["feedback"],
            "trajectory_summary": episode["trajectory_summary"],
            "lesson": episode["lesson"],
        }
    raise ValueError(f"Unsupported projection policy: {policy}")


def eligible_memory(
    memory: list[dict[str, Any]], *, policy: str, current_step: int, delay_steps: int
) -> list[dict[str, Any]]:
    if policy == "full_delayed":
        threshold = current_step - delay_steps
        return [row for row in memory if row["sequence_step"] <= threshold]
    return list(memory)


def retrieve_projected_memory(
    memory: list[dict[str, Any]], query: Any, *, policy: str, limit: int
) -> list[dict[str, Any]]:
    if not memory or limit < 1:
        return []
    if policy == "case_only":
        rows = retrieve_case_memory(memory, query, limit)
        return [
            {
                "experience_id": row["case_memory_id"],
                "prior_input": row["prior_input"],
                "verified_ground_truth": row["verified_ground_truth"],
                "retrieval_trace": row["retrieval_trace"],
            }
            for row in rows
        ]
    full_rows = retrieve_experience_memory(memory, query, limit)
    by_id = {row["experience_id"]: row for row in memory}
    projected = []
    for row in full_rows:
        value = project_episode(by_id[row["experience_id"]], policy)
        value["retrieval_trace"] = row["retrieval_trace"]
        value["contrast_role"] = row["contrast_role"]
        projected.append(value)
    return projected


def method_uses_writable_memory(method: str) -> bool:
    return METHOD_POLICIES[method]["memory"] not in ("none",)


def run_task(
    *, task_record: dict[str, Any], release: Path, output: Path, method: str,
    call_api: Any, config_sha: str, evidence_budget: int, history_k: int,
    experience_k: int, feedback_delay_steps: int, model: str,
    model_revision: str, dry_run: bool, resume: bool, max_steps: int | None,
    prompt_set: str,
) -> dict[str, Any]:
    source = release / task_record["relative_path"]
    cases = read_jsonl(source / "cases.jsonl")
    gold_rows = read_jsonl(source / "gold.jsonl")
    order = read_jsonl(source / "order.jsonl")
    if max_steps is not None:
        cases, gold_rows, order = cases[:max_steps], gold_rows[:max_steps], order[:max_steps]
    if not (len(cases) == len(gold_rows) == len(order)):
        raise ValueError(f"Sequence length mismatch: {task_record['task']}")
    if any(
        case["case_id"] != gold["case_id"] or case["case_id"] != sequence["case_id"]
        for case, gold, sequence in zip(cases, gold_rows, order)
    ):
        raise ValueError(f"Sequence alignment mismatch: {task_record['task']}")

    task_output = output / "test" / method / task_record["family"] / task_record["task"]
    prediction_path = task_output / "predictions.jsonl"
    existing = read_jsonl(prediction_path) if prediction_path.exists() else []
    if existing and not resume:
        raise FileExistsError(prediction_path)
    if len(existing) > len(cases):
        raise ValueError("Resume output is longer than the sequence")

    active_memory: list[dict[str, Any]] = []
    for index, result in enumerate(existing):
        case, gold, sequence = cases[index], gold_rows[index], order[index]
        if (
            result.get("sequence_step") != sequence["sequence_step"]
            or result.get("case_id") != case["case_id"]
            or result.get("case_sha256") != digest(case)
            or result.get("run_config_sha256") != config_sha
        ):
            raise ValueError(f"Resume prefix mismatch: {task_record['task']}/{index + 1}")
        if method_uses_writable_memory(method):
            active_memory.append(compress_experience(
                case=case, gold=gold, result=result,
                sequence_id=sequence["sequence_id"],
                sequence_step=sequence["sequence_step"],
            ))

    policy = METHOD_POLICIES[method]
    memory_policy = policy["memory"]
    for index in range(len(existing), len(cases)):
        case, gold, sequence = cases[index], gold_rows[index], order[index]
        started = time.monotonic()
        available_memory = eligible_memory(
            active_memory, policy=memory_policy,
            current_step=sequence["sequence_step"], delay_steps=feedback_delay_steps,
        )
        row: dict[str, Any] = {
            "protocol": case["protocol"],
            "sequential_run_protocol": RUN_PROTOCOL,
            "prompt_revision": PROMPT_REVISION,
            "case_id": case["case_id"],
            "case_sha256": digest(case),
            "task": case["task"],
            "task_kind": case["task_kind"],
            "family": task_record["family"],
            "method": method,
            "mode": method,
            "method_policy": policy,
            "model": model,
            "model_revision": model_revision,
            "sequence_id": sequence["sequence_id"],
            "sequence_regime": sequence["sequence_regime"],
            "order_seed": sequence["order_seed"],
            "sequence_step": sequence["sequence_step"],
            "sequence_group_id": sequence["sequence_group_id"],
            "group_first_occurrence": sequence["group_first_occurrence"],
            "same_group_seen_before": sequence["same_group_seen_before"],
            "memory_size_written_before": len(active_memory),
            "memory_size_available_before": len(available_memory),
            "run_config_sha256": config_sha,
        }
        planner = None
        planner_prompt = None
        planner_attempts = None
        prior_cases: list[dict[str, Any]] = []
        experience_candidates: list[dict[str, Any]] = []
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
                        "sequence_step": episode["sequence_step"],
                    }
                    for episode in available_memory[-history_k:]
                ]
            elif method == "dynamic_case_rag":
                prior_cases = retrieve_case_memory(
                    available_memory, case["query_visible"], history_k
                )
            elif memory_policy not in ("none", "rolling_case", "retrieved_case"):
                projection = "full" if memory_policy == "full_delayed" else memory_policy
                experience_candidates = retrieve_projected_memory(
                    available_memory, case["query_visible"],
                    policy=projection, limit=experience_k,
                )

            if method != "llm_only":
                if policy["controller"] == "react":
                    planner_prompt, planner_schema_value = planner_messages(
                        case, experience_candidates, evidence_budget, experience_k,
                        prompt_set=prompt_set,
                    )
                    if dry_run:
                        planner = {
                            "action": "inspect_view",
                            "requested_view": case["retrieval"]["view"],
                            "selected_evidence_ids": case["initial_evidence_ids"][:evidence_budget],
                            "selected_experience_ids": [
                                value["experience_id"]
                                for value in experience_candidates[:experience_k]
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
                references, observation = select_static_evidence(
                    case, planner, evidence_budget
                )

            if policy["controller"] == "react" and planner is not None:
                by_id = {value["experience_id"]: value for value in experience_candidates}
                for experience_id in planner["selected_experience_ids"]:
                    if experience_id not in by_id:
                        raise ValueError(
                            f"Planner selected unavailable experience: {experience_id}"
                        )
                    selected_experiences.append(by_id[experience_id])

            messages, schema = final_messages(
                case=case, method=method, references=references,
                prior_cases=prior_cases, experiences=selected_experiences,
                planner=planner, observation=observation,
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
                    {
                        "step": 3, "kind": "experience_observation",
                        "record": selected_experiences or prior_cases,
                    },
                ],
                final_prompt_sha256=digest(messages),
            )
            if dry_run:
                row.update(
                    status="dry_run", planner_messages=planner_prompt,
                    final_messages=messages, final_schema=schema,
                    final_attempts=[],
                )
            else:
                response, raw, final_usage, raw_response, final_attempts = call_api(
                    messages, schema, planner=False
                )
                validate_response(
                    case, response, evidence_ids,
                    {value for value in experience_ids if value},
                )
                row.update(
                    status="success", response=response, raw_content=raw,
                    usage={"planner": row.get("planner_usage"), "final": final_usage},
                    raw_response=raw_response, final_attempts=final_attempts,
                )
        except Exception as exc:
            row.update(
                status="failed", error=f"{type(exc).__name__}: {exc}",
                failed_attempts=getattr(exc, "retry_attempts", None),
            )
            if isinstance(exc, ResponseError):
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.monotonic() - started

        experience = compress_experience(
            case=case, gold=gold, result=row,
            sequence_id=sequence["sequence_id"],
            sequence_step=sequence["sequence_step"],
        )
        row.update(
            feedback=experience["feedback"],
            static_anchor_feedback=experience["static_anchor_feedback"],
            ground_truth_revealed_after_prediction=True,
        )
        if method_uses_writable_memory(method):
            active_memory.append(experience)
        row["memory_size_written_after"] = len(active_memory)
        _append_jsonl(prediction_path, row)
        with PRINT_LOCK:
            print(canonical_json({
                "event": "progress", "method": method, "task": case["task"],
                "step": sequence["sequence_step"], "total": len(cases),
                "status": row["status"],
            }), flush=True)

    if method_uses_writable_memory(method):
        write_jsonl(task_output / "verified_feedback_memory.jsonl", active_memory)
        (task_output / "memory_policy.json").write_text(
            json.dumps({
                "method": method, "policy": policy,
                "feedback_delay_steps": feedback_delay_steps,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if method == "matrecall":
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
        "retried_success": sum(
            row.get("status") == "success"
            and (
                len(row.get("planner_attempts") or []) > 1
                or len(row.get("final_attempts") or []) > 1
            )
            for row in rows
        ),
        "predictions_sha256": file_sha256(prediction_path),
    }


def combine_outputs(
    *, release: Path, output: Path, methods: list[str],
    task_records: list[dict[str, Any]], max_steps: int | None,
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
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--prompt-set", choices=PROMPT_SETS, required=True,
        help="Planner prompt set from run_online_adaptation.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--evidence-budget", type=int, default=8)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--experience-k", type=int, default=6)
    parser.add_argument("--feedback-delay-steps", type=int, default=4)
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
    parser.add_argument("--max-steps-per-task", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_tokens, args.planner_max_tokens, args.evidence_budget,
        args.history_k, args.experience_k, args.feedback_delay_steps,
        args.max_attempts, args.task_concurrency, args.timeout,
    ) < 1:
        raise ValueError("All budgets, delay, retries, and concurrency values must be positive")
    if args.max_steps_per_task is not None and args.max_steps_per_task < 1:
        raise ValueError("--max-steps-per-task must be positive")

    release_manifest = json.loads((args.release / "manifest.json").read_text(encoding="utf-8"))
    if release_manifest.get("status") != "frozen" or release_manifest.get("task_count") != 22:
        raise ValueError("Expected a frozen 22-task memory-ablation stream release")
    task_records = release_manifest["tasks"]
    config = {
        "run_protocol": RUN_PROTOCOL,
        "prompt_revision": PROMPT_REVISION,
        "planner_prompt_revision": PLANNER_PROMPT_REVISIONS[args.prompt_set],
        "release_path": str(args.release.resolve()),
        "sequence_regime": release_manifest.get("sequence_regime"),
        "order_seed": release_manifest.get("order_seed"),
        "model": args.model,
        "model_revision": args.model_revision,
        "methods": args.methods,
        "method_policies": {method: METHOD_POLICIES[method] for method in args.methods},
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
            "evidence_budget": args.evidence_budget,
            "history_k": args.history_k,
            "experience_k": args.experience_k,
            "feedback_delay_steps": args.feedback_delay_steps,
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
                    current_messages, schema, model=args.model, base_url=args.base_url,
                    api_key=None,
                    max_tokens=args.planner_max_tokens if planner else args.max_tokens,
                    seed=args.seed + attempt - 1, timeout=args.timeout,
                    response_format="json_schema", disable_thinking=args.disable_thinking,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    reasoning_effort=None, presence_penalty=args.presence_penalty,
                    allow_unescaped_control_chars=args.allow_unescaped_control_chars,
                )
                attempts.append({"attempt": attempt, "status": "success"})
                return response, raw, usage, raw_response, attempts
            except ResponseError as exc:
                attempts.append({
                    "attempt": attempt,
                    "status": "failed",
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
                            "JSON matching the supplied schema; do not add extra explanation."
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
                    experience_k=args.experience_k,
                    feedback_delay_steps=args.feedback_delay_steps,
                    model=args.model, model_revision=args.model_revision,
                    dry_run=args.dry_run, resume=args.resume,
                    max_steps=args.max_steps_per_task, prompt_set=args.prompt_set,
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
