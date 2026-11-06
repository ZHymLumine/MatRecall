"""Run MatRecall x MatText exposure and frozen-transfer evaluations."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

from matrecall.mattext import (
    EXPOSURE_METHODS,
    FROZEN_METHODS,
    PROMPT_REVISION,
    RUN_PROTOCOL,
    UNRELATED_TASK,
    append_jsonl,
    digest,
    file_sha256,
    read_jsonl,
    retrieve_full_experience,
    runtime_case,
    shuffle_episode_feedback,
    write_jsonl,
)
from matrecall.episodic_memory import (
    compress_experience,
    graph_records,
    retrieve_case_memory,
)
from matrecall.llm_client import ResponseError, complete
from matrecall.run_online_adaptation import (
    final_messages,
    planner_messages,
    select_static_evidence,
    validate_response,
)


PRINT_LOCK = threading.Lock()

V08_NUMERIC_PATTERN = r"^-?(0|[1-9][0-9]{0,5})(\.[0-9]{1,6})?$"


def compact_planner_contract(
    messages: list[dict[str, str]], schema: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Bound planner trace verbosity without changing its selectable evidence."""
    messages = deepcopy(messages)
    schema = deepcopy(schema)
    for field in ("comparison", "conflict_assessment", "memory_use"):
        schema["properties"][field]["maxLength"] = 240
    payload = json.loads(messages[-1]["content"])
    payload["response_contract"] = (
        "Keep comparison, conflict_assessment, and memory_use concise (at most two "
        "sentences each). Select IDs only from the supplied catalogs."
    )
    messages[-1]["content"] = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return messages, schema


def compact_final_contract(
    messages: list[dict[str, str]], schema: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Use a bounded decimal string during constrained decoding.

    JSON-schema numeric terminals can continue emitting trailing decimal zeros
    until the token budget is exhausted.  The bounded string is converted back
    to a float immediately after decoding, so stored predictions and metrics
    retain the existing numeric interface.
    """
    messages = deepcopy(messages)
    schema = deepcopy(schema)
    analysis = schema["properties"]["analysis_process"]
    analysis["maxItems"] = 2
    for field in ("comparison", "implication"):
        analysis["items"]["properties"][field]["maxLength"] = 320
    value_schema = schema["properties"]["reasoning_results"]["properties"][
        "prediction"
    ]["properties"]["value"]
    if value_schema.get("type") != "number":
        raise ValueError("MatText requires a scalar numeric prediction schema")
    value_schema.clear()
    value_schema.update({"type": "string", "pattern": V08_NUMERIC_PATTERN})
    payload = json.loads(messages[-1]["content"])
    payload["response_contract"] = (
        "Use no more than two concise analysis steps. Return prediction.value as a "
        "plain decimal string with at most six digits after the decimal point; do "
        "not use scientific notation or append trailing zeros."
    )
    messages[-1]["content"] = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return messages, schema


def normalize_mattext_prediction(response: dict[str, Any]) -> None:
    """Restore the public numeric prediction representation after decoding."""
    prediction = response["reasoning_results"]["prediction"]
    value = prediction.get("value")
    if not isinstance(value, str):
        raise ValueError("MatText constrained prediction must be a decimal string")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("MatText prediction is non-finite")
    prediction["value"] = parsed


def _task_record(
    manifest: dict[str, Any], task: str, fold: int,
) -> dict[str, Any]:
    matches = [
        row for row in manifest["task_fold_records"]
        if row["task"] == task and int(row["fold"]) == fold
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one task/fold record: {task}/{fold}")
    return matches[0]


def _experience(
    case: dict[str, Any], gold: dict[str, Any], result: dict[str, Any],
    representation: str, sequence_id: str, step: int,
) -> dict[str, Any]:
    episode = compress_experience(
        case=case,
        gold=gold,
        result=result,
        sequence_id=sequence_id,
        sequence_step=step,
    )
    episode["memory_metadata"] = deepcopy(case["memory_metadata"])
    episode["memory_metadata"]["source_representation"] = representation
    episode["query_summary"] = json.dumps(
        case["query_visible"], ensure_ascii=False, sort_keys=True
    )[:1600]
    episode["query_tokens"] = sorted(set(
        episode["query_summary"].lower().replace("\n", " ").split()
    ))
    return episode


def _call_prediction(
    *, base_case: dict[str, Any], case: dict[str, Any], method: str,
    active_memory: list[dict[str, Any]], call_api: Any, dry_run: bool,
    evidence_budget: int, history_k: int, experience_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    planner = None
    planner_prompt = None
    planner_usage = None
    prior_cases: list[dict[str, Any]] = []
    experience_candidates: list[dict[str, Any]] = []
    selected_experiences: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    observation: dict[str, Any] = {
        "action": "predict", "requested_view": "none",
        "returned_evidence_ids": [], "returned_count": 0,
    }
    if method == "dynamic_case_rag":
        prior_cases = retrieve_case_memory(
            active_memory, case["query_visible"], history_k
        )
    elif method.startswith("matrecall"):
        experience_candidates = retrieve_full_experience(
            active_memory, base_case, experience_k
        )
    include_static = method != "llm_only"
    if method.startswith("matrecall"):
        # The MatText experiment uses the revised planner prompt.
        planner_prompt, schema = planner_messages(
            case, experience_candidates, evidence_budget, experience_k,
            prompt_set="revised", include_static=True,
        )
        planner_prompt, schema = compact_planner_contract(
            planner_prompt, schema
        )
        if dry_run:
            planner = {
                "action": "inspect_view",
                "requested_view": "hybrid",
                "selected_evidence_ids": case["initial_evidence_ids"][:evidence_budget],
                "selected_experience_ids": [
                    row["experience_id"] for row in experience_candidates[:experience_k]
                ],
                "comparison": "dry-run planner",
                "conflict_assessment": "dry-run planner",
                "memory_use": "dry-run planner",
            }
        else:
            planner, _, planner_usage, _ = call_api(
                planner_prompt, schema, planner=True
            )
        by_id = {row["experience_id"]: row for row in experience_candidates}
        for identifier in planner["selected_experience_ids"]:
            if identifier not in by_id:
                raise ValueError(f"Planner selected unknown experience: {identifier}")
            selected_experiences.append(by_id[identifier])
    if include_static:
        references, observation = select_static_evidence(
            case, planner, evidence_budget
        )
    messages, schema = final_messages(
        case=case,
        method=method,
        references=references,
        prior_cases=prior_cases,
        experiences=selected_experiences,
        planner=planner,
        observation=observation,
    )
    messages, schema = compact_final_contract(messages, schema)
    evidence_ids = {row["evidence_id"] for row in references}
    experience_ids = {
        row.get("experience_id", row.get("case_memory_id"))
        for row in [*prior_cases, *selected_experiences]
    }
    result: dict[str, Any] = {
        "planner_record": planner,
        "planner_prompt_sha256": digest(planner_prompt) if planner_prompt else None,
        "planner_usage": planner_usage,
        "selected_evidence_ids": sorted(evidence_ids),
        "selected_experience_ids": sorted(value for value in experience_ids if value),
        "agent_trace": [
            {"step": 1, "kind": "planner", "record": planner},
            {"step": 2, "kind": "static_tool_observation", "record": observation},
            {"step": 3, "kind": "experience_observation", "record": selected_experiences or prior_cases},
        ],
        "final_prompt_sha256": digest(messages),
    }
    if dry_run:
        result.update(
            status="dry_run",
            planner_messages=planner_prompt,
            final_messages=messages,
            final_schema=schema,
        )
    else:
        response, raw, usage, raw_response = call_api(messages, schema, planner=False)
        normalize_mattext_prediction(response)
        validate_response(
            case, response, evidence_ids, {value for value in experience_ids if value}
        )
        result.update(
            status="success",
            response=response,
            raw_content=raw,
            usage={"planner": planner_usage, "final": usage},
            raw_response=raw_response,
        )
    return result, planner_prompt, messages


def _recover_memory(
    prediction_rows: list[dict[str, Any]], base_cases: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]], static_library: list[dict[str, Any]],
    representation: str,
) -> list[dict[str, Any]]:
    memory = []
    for row, base_case, gold in zip(prediction_rows, base_cases, gold_rows):
        case = runtime_case(base_case, static_library, representation)
        memory.append(_experience(
            case, gold, row, representation,
            row["sequence_id"], int(row["sequence_step"]),
        ))
    return memory


def run_exposure_cell(
    *, release: Path, task_record: dict[str, Any], output: Path,
    representation: str, method: str, call_api: Any, config_sha: str,
    model: str, model_revision: str, dry_run: bool, resume: bool,
    evidence_budget: int, history_k: int, experience_k: int,
    max_cases: int | None,
) -> dict[str, Any]:
    source = release / task_record["relative_path"]
    static_library = read_jsonl(source / "static_library.jsonl")
    cases = read_jsonl(source / "exposure" / "cases.jsonl")
    gold = read_jsonl(source / "exposure" / "gold.jsonl")
    order = read_jsonl(source / "exposure" / "order.jsonl")
    if max_cases is not None:
        cases, gold, order = cases[:max_cases], gold[:max_cases], order[:max_cases]
    if not (len(cases) == len(gold) == len(order)):
        raise ValueError("Exposure alignment mismatch")
    cell = output / "exposure" / method / task_record["task"] / f"fold_{task_record['fold']}" / representation
    predictions_path = cell / "predictions.jsonl"
    commits_path = cell / "prediction_commits.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    if existing and not resume:
        raise FileExistsError(predictions_path)
    if len(existing) > len(cases):
        raise ValueError("Resume prefix exceeds exposure sequence")
    active_memory = _recover_memory(
        existing, cases[:len(existing)], gold[:len(existing)], static_library,
        representation,
    ) if method in ("dynamic_case_rag", "matrecall") else []

    for index in range(len(existing), len(cases)):
        base_case, target, sequence = cases[index], gold[index], order[index]
        case = runtime_case(base_case, static_library, representation)
        started = time.monotonic()
        row: dict[str, Any] = {
            "protocol": RUN_PROTOCOL,
            "prompt_revision": PROMPT_REVISION,
            "stage": "exposure",
            "case_id": base_case["case_id"],
            "case_sha256": digest(base_case),
            "task": task_record["task"],
            "task_kind": "scalar_regression",
            "fold": int(task_record["fold"]),
            "representation": representation,
            "method": method,
            "model": model,
            "model_revision": model_revision,
            "sequence_id": f"{task_record['task']}:fold-{task_record['fold']}:{representation}",
            "sequence_step": int(sequence["sequence_step"]),
            "memory_size_before": len(active_memory),
            "run_config_sha256": config_sha,
        }
        try:
            result, _, _ = _call_prediction(
                base_case=base_case,
                case=case,
                method=method,
                active_memory=active_memory,
                call_api=call_api,
                dry_run=dry_run,
                evidence_budget=evidence_budget,
                history_k=history_k,
                experience_k=experience_k,
            )
            row.update(result)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, ResponseError):
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.monotonic() - started
        # This append happens before target access and makes the reveal boundary auditable.
        append_jsonl(commits_path, {
            "case_id": row["case_id"],
            "sequence_step": row["sequence_step"],
            "prediction_record_sha256_before_reveal": digest(row),
            "status": row["status"],
        })
        episode = _experience(
            case, target, row, representation,
            row["sequence_id"], row["sequence_step"],
        )
        row.update(
            feedback=episode["feedback"],
            static_anchor_feedback=episode["static_anchor_feedback"],
            ground_truth_revealed_after_prediction=True,
        )
        if method in ("dynamic_case_rag", "matrecall"):
            active_memory.append(episode)
        row["memory_size_after"] = len(active_memory)
        append_jsonl(predictions_path, row)
        with PRINT_LOCK:
            print(json.dumps({
                "event": "progress", "stage": "exposure", "method": method,
                "task": task_record["task"], "fold": task_record["fold"],
                "representation": representation, "step": index + 1,
                "total": len(cases), "status": row["status"],
            }), flush=True)
    if method in ("dynamic_case_rag", "matrecall"):
        memory_path = cell / "verified_feedback_memory.jsonl"
        if memory_path.exists():
            memory_path.unlink()
        write_jsonl(memory_path, active_memory)
    if method == "matrecall":
        nodes, edges = graph_records(active_memory)
        for name, values in (
            ("experience_graph_nodes.jsonl", nodes),
            ("experience_graph_edges.jsonl", edges),
        ):
            path = cell / name
            if path.exists():
                path.unlink()
            write_jsonl(path, values)
    rows = read_jsonl(predictions_path)
    return {
        "stage": "exposure", "task": task_record["task"],
        "fold": int(task_record["fold"]), "representation": representation,
        "method": method, "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "dry_run": sum(row["status"] == "dry_run" for row in rows),
        "predictions_sha256": file_sha256(predictions_path),
    }


def _memory_path(
    exposure_root: Path, task: str, method: str, fold: int, representation: str,
) -> Path:
    return (
        exposure_root / task / "exposure" / method / task /
        f"fold_{fold}" / representation / "verified_feedback_memory.jsonl"
    )


def run_frozen_cell(
    *, release: Path, task_record: dict[str, Any], output: Path,
    exposure_root: Path, source_representation: str, target_representation: str,
    method: str, call_api: Any, config_sha: str, model: str,
    model_revision: str, dry_run: bool, resume: bool, evidence_budget: int,
    history_k: int, experience_k: int, max_cases: int | None,
) -> dict[str, Any]:
    source = release / task_record["relative_path"]
    static_library = read_jsonl(source / "static_library.jsonl")
    cases = read_jsonl(source / "frozen_test" / "cases.jsonl")
    if max_cases is not None:
        cases = cases[:max_cases]
    task, fold = task_record["task"], int(task_record["fold"])
    memory: list[dict[str, Any]] = []
    actual_memory_task = task
    if method == "dynamic_case_rag":
        memory = read_jsonl(_memory_path(
            exposure_root, task, "dynamic_case_rag", fold, source_representation
        ))
    elif method.startswith("matrecall"):
        if method == "matrecall_unrelated":
            if task not in UNRELATED_TASK:
                raise ValueError(f"No preregistered unrelated task for {task}")
            actual_memory_task = UNRELATED_TASK[task]
        memory = read_jsonl(_memory_path(
            exposure_root, actual_memory_task, "matrecall", fold,
            source_representation,
        ))
        if method == "matrecall_shuffled":
            memory = shuffle_episode_feedback(
                memory,
                namespace=f"{task}:{fold}:{source_representation}:{target_representation}",
            )
    source_name = source_representation if memory else "none"
    cell = (
        output / "frozen" / method / task / f"fold_{fold}" /
        f"source_{source_name}" / f"target_{target_representation}"
    )
    predictions_path = cell / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    if existing and not resume:
        raise FileExistsError(predictions_path)
    for index in range(len(existing), len(cases)):
        base_case = cases[index]
        case = runtime_case(base_case, static_library, target_representation)
        started = time.monotonic()
        row: dict[str, Any] = {
            "protocol": RUN_PROTOCOL,
            "prompt_revision": PROMPT_REVISION,
            "stage": "frozen_test",
            "case_id": base_case["case_id"],
            "case_sha256": digest(base_case),
            "task": task,
            "task_kind": "scalar_regression",
            "fold": fold,
            "source_representation": source_name,
            "target_representation": target_representation,
            "memory_source_task": actual_memory_task if memory else None,
            "method": method,
            "model": model,
            "model_revision": model_revision,
            "memory_size_before": len(memory),
            "run_config_sha256": config_sha,
            "frozen_memory": True,
            "ground_truth_revealed_during_prediction": False,
            "memory_write_after_prediction": False,
        }
        try:
            result, _, _ = _call_prediction(
                base_case=base_case,
                case=case,
                method=method,
                active_memory=memory,
                call_api=call_api,
                dry_run=dry_run,
                evidence_budget=evidence_budget,
                history_k=history_k,
                experience_k=experience_k,
            )
            row.update(result)
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, ResponseError):
                row["raw_response"] = exc.raw_response
        row["elapsed_seconds"] = time.monotonic() - started
        append_jsonl(predictions_path, row)
        with PRINT_LOCK:
            print(json.dumps({
                "event": "progress", "stage": "frozen", "method": method,
                "task": task, "fold": fold, "source": source_name,
                "target": target_representation, "step": index + 1,
                "total": len(cases), "status": row["status"],
            }), flush=True)
    rows = read_jsonl(predictions_path)
    return {
        "stage": "frozen_test", "task": task, "fold": fold,
        "source_representation": source_name,
        "target_representation": target_representation, "method": method,
        "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "dry_run": sum(row["status"] == "dry_run" for row in rows),
        "predictions_sha256": file_sha256(predictions_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("exposure", "frozen"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--exposure-root", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--representations", nargs="+")
    parser.add_argument("--source-representations", nargs="+")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--planner-max-tokens", type=int, default=768)
    parser.add_argument("--evidence-budget", type=int, default=6)
    parser.add_argument("--history-k", type=int, default=4)
    parser.add_argument("--experience-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--allow-unescaped-control-chars", action="store_true")
    parser.add_argument("--cell-concurrency", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.release / "manifest.json").read_text())
    if manifest.get("status") != "frozen":
        raise ValueError("Expected frozen MatText release")
    tasks = set(manifest["tasks"])
    if args.task not in tasks:
        raise ValueError(f"Unknown task {args.task}")
    available_representations = [row["name"] for row in manifest["representations"]]
    representations = args.representations or available_representations
    if not set(representations) <= set(available_representations):
        raise ValueError("Unknown representation requested")
    methods = args.methods or list(
        EXPOSURE_METHODS if args.stage == "exposure" else FROZEN_METHODS
    )
    allowed = set(EXPOSURE_METHODS if args.stage == "exposure" else FROZEN_METHODS)
    if not set(methods) <= allowed:
        raise ValueError(f"Unsupported methods for {args.stage}: {methods}")
    if args.stage == "frozen" and args.exposure_root is None:
        raise ValueError("--exposure-root is required for frozen evaluation")
    source_representations = args.source_representations or [
        value for value in ("composition", "crystal_text_llm")
        if value in available_representations
    ]
    config = {
        "protocol": RUN_PROTOCOL,
        "prompt_revision": PROMPT_REVISION,
        "release_manifest_sha256": file_sha256(args.release / "manifest.json"),
        "stage": args.stage,
        "task": args.task,
        "folds": args.folds,
        "representations": representations,
        "source_representations": source_representations,
        "methods": methods,
        "model": args.model,
        "model_revision": args.model_revision,
        "sampling": {
            "max_tokens": args.max_tokens,
            "planner_max_tokens": args.planner_max_tokens,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "presence_penalty": args.presence_penalty,
            "disable_thinking": args.disable_thinking,
        },
        "memory": {
            "evidence_budget": args.evidence_budget,
            "history_k": args.history_k,
            "experience_k": args.experience_k,
        },
        "max_cases": args.max_cases,
        "dry_run": args.dry_run,
    }
    config_sha = digest(config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"run_manifest_{args.stage}.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["run_config_sha256"] != config_sha and not args.resume:
            raise ValueError("Existing run manifest differs")
    else:
        manifest_path.write_text(json.dumps({
            **config, "run_config_sha256": config_sha,
        }, indent=2, sort_keys=True) + "\n")

    def call_api(messages: list[dict[str, str]], schema: dict[str, Any], *, planner: bool):
        return complete(
            messages, schema, model=args.model, base_url=args.base_url,
            api_key=None,
            max_tokens=args.planner_max_tokens if planner else args.max_tokens,
            seed=args.seed, timeout=args.timeout, response_format="json_schema",
            disable_thinking=args.disable_thinking,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            reasoning_effort=None, presence_penalty=args.presence_penalty,
            allow_unescaped_control_chars=args.allow_unescaped_control_chars,
        )

    cells: list[dict[str, Any]] = []
    for fold in args.folds:
        record = _task_record(manifest, args.task, fold)
        if args.stage == "exposure":
            for representation in representations:
                for method in methods:
                    cells.append({
                        "release": args.release, "task_record": record,
                        "output": args.output, "representation": representation,
                        "method": method, "call_api": call_api,
                        "config_sha": config_sha, "model": args.model,
                        "model_revision": args.model_revision,
                        "dry_run": args.dry_run, "resume": args.resume,
                        "evidence_budget": args.evidence_budget,
                        "history_k": args.history_k,
                        "experience_k": args.experience_k,
                        "max_cases": args.max_cases,
                    })
        else:
            for target_representation in representations:
                for method in methods:
                    sources = (
                        source_representations
                        if method not in ("llm_only", "static_multiview_rag")
                        else ["none"]
                    )
                    for source_representation in sources:
                        cells.append({
                            "release": args.release, "task_record": record,
                            "output": args.output,
                            "exposure_root": args.exposure_root,
                            "source_representation": source_representation,
                            "target_representation": target_representation,
                            "method": method, "call_api": call_api,
                            "config_sha": config_sha, "model": args.model,
                            "model_revision": args.model_revision,
                            "dry_run": args.dry_run, "resume": args.resume,
                            "evidence_budget": args.evidence_budget,
                            "history_k": args.history_k,
                            "experience_k": args.experience_k,
                            "max_cases": args.max_cases,
                        })
    summaries: list[dict[str, Any]] = []
    runner = run_exposure_cell if args.stage == "exposure" else run_frozen_cell
    with ThreadPoolExecutor(max_workers=min(args.cell_concurrency, len(cells))) as executor:
        futures = [executor.submit(runner, **cell) for cell in cells]
        for future in as_completed(futures):
            summaries.append(future.result())
    completion = {
        "status": "dry_run" if args.dry_run else (
            "completed_with_recorded_response_failures"
            if any(row["failed"] for row in summaries) else "completed"
        ),
        "stage": args.stage,
        "task": args.task,
        "cell_count": len(summaries),
        "counts": sorted(
            summaries,
            key=lambda row: tuple(str(row.get(key, "")) for key in (
                "task", "fold", "method", "representation",
                "source_representation", "target_representation",
            )),
        ),
        "run_config_sha256": config_sha,
    }
    (args.output / f"completion_{args.stage}.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "event": "complete", "stage": args.stage,
        "task": args.task, "cells": len(summaries),
        "status": completion["status"],
    }), flush=True)


if __name__ == "__main__":
    main()

