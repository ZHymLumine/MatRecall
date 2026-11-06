"""Train matched simple specialized baselines for eight unified tasks.

These models use only each task's training memory and select hyperparameters on
the validation split.  They are deliberately labelled as matched simple
baselines; they are not representations of the published paper models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from matrecall.unified_evaluation import (
    CONDITION_FIELDS,
    FECONI_ELEMENTS,
    PROTOCOL,
    TASK_ORDER,
    canonical_json,
    digest,
    envelope,
    file_sha256,
    gold_targets,
    graph_scores,
    load_task,
    memory_target,
    primary_selection_score,
    read_jsonl,
    write_jsonl,
)


TASKS = tuple(task for task in TASK_ORDER if task not in {
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "mbj_bandgap",
    "bulk_modulus_kv",
    "magmom_oszicar",
})


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _formula_features(formula: str, prefix: str = "") -> dict[str, float]:
    from pymatgen.core import Composition

    try:
        fractions = Composition(str(formula)).fractional_composition.get_el_amt_dict()
    except Exception:
        return {prefix + "formula=" + str(formula): 1.0}
    return {prefix + "el=" + key: float(value) for key, value in fractions.items()}


def structured_features(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "element_fractions" in payload:
        result.update({"el=" + key: float(value) for key, value in payload["element_fractions"].items()})
    if "target_properties" in payload:
        result.update({"property=" + key: float(value) for key, value in payload["target_properties"].items()})
    for key in ("target_formula",):
        if key in payload:
            result.update(_formula_features(payload[key], "target:"))
    for key in ("target_formulas", "precursor_formulas"):
        for formula in payload.get(key, []):
            for name, value in _formula_features(formula, key + ":").items():
                result[name] = result.get(name, 0.0) + value
        if key in payload:
            result[key + ":count"] = len(payload[key])
    for key, value in payload.items():
        if key not in {"element_fractions", "target_properties", "target_formula", "target_formulas", "precursor_formulas"}:
            result[key + "=" + str(value)] = 1.0
    return result


def frame_features(frame) -> list[dict[str, Any]]:
    return [structured_features(_json(value)) for value in frame["input_json"]]


def _ranked_predictions(model: Any, matrix: Any) -> list[dict[str, Any]]:
    probabilities = model.predict_proba(matrix)
    classes = np.asarray(model.classes_)
    return [
        {"ranked_space_group_numbers": [int(classes[index]) for index in np.argsort(-row)[:5]]}
        for row in probabilities
    ]


def classification_task(release: Path, task: str, seed: int, jobs: int):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    vectorizer = DictVectorizer(sparse=True)
    train_x = vectorizer.fit_transform(frame_features(memory))
    validation_x = vectorizer.transform(frame_features(validation))
    test_x = vectorizer.transform(frame_features(test))
    if task == "composition_crystal_system_context_to_space_group":
        train_y = memory["space_group_number"].astype(int).to_numpy()
    else:
        train_y = np.asarray([canonical_json(sorted(_json(value)["phase_set"])) for value in memory["target_json"]])
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    models = [
        ("dummy_prior", DummyClassifier(strategy="prior", random_state=seed)),
        ("random_forest", RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced")),
        ("extra_trees", ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced")),
        ("extra_trees_leaf2", ExtraTreesClassifier(n_estimators=400, random_state=seed, n_jobs=jobs, class_weight="balanced", min_samples_leaf=2)),
    ]
    trials, fitted = [], {}
    for name, model in models:
        model.fit(train_x, train_y)
        if task == "composition_crystal_system_context_to_space_group":
            predicted = _ranked_predictions(model, validation_x)
        else:
            predicted = [{"phase_set": json.loads(value)} for value in model.predict(validation_x)]
        score, maximize = primary_selection_score(task, expected, predicted)
        trials.append({"name": name, "selection_value": score, "larger_is_better": maximize})
        fitted[name] = model
    selected = sorted(trials, key=lambda row: (-row["selection_value"] if row["larger_is_better"] else row["selection_value"], row["name"]))[0]
    model = fitted[selected["name"]]
    if task == "composition_crystal_system_context_to_space_group":
        predictions = _ranked_predictions(model, test_x)
    else:
        predictions = [{"phase_set": json.loads(value)} for value in model.predict(test_x)]
    return predictions, {
        "task": task,
        "baseline_family": "composition_descriptor_tree_classifier",
        "selected": selected,
        "candidates": trials,
        "feature_count": len(vectorizer.feature_names_),
    }


def inverse_composition_task(release: Path, task: str, seed: int, jobs: int):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    vectorizer = DictVectorizer(sparse=False)
    train_x = vectorizer.fit_transform(frame_features(memory))
    validation_x = vectorizer.transform(frame_features(validation))
    test_x = vectorizer.transform(frame_features(test))
    train_y = np.asarray([
        [_json(value)["element_fractions"].get(element, 0.0) for element in FECONI_ELEMENTS]
        for value in memory["target_json"]
    ])
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    trials, fitted = [], {}
    for leaf in (1, 2, 4):
        name = f"extra_trees_leaf{leaf}"
        model = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=leaf, random_state=seed, n_jobs=jobs)
        model.fit(train_x, train_y)
        values = _composition_predictions(model.predict(validation_x))
        score, maximize = primary_selection_score(task, expected, values)
        trials.append({"name": name, "selection_value": score, "larger_is_better": maximize})
        fitted[name] = model
    selected = min(trials, key=lambda row: (row["selection_value"], row["name"]))
    predictions = _composition_predictions(fitted[selected["name"]].predict(test_x))
    return predictions, {
        "task": task,
        "baseline_family": "property_vector_to_composition_extra_trees",
        "selected": selected,
        "candidates": trials,
        "feature_count": len(vectorizer.feature_names_),
    }


def _composition_predictions(matrix: Any) -> list[dict[str, Any]]:
    output = []
    for row in np.asarray(matrix):
        clipped = np.maximum(row.astype(float), 0.0)
        if clipped.sum() <= 0:
            clipped[0] = 1.0
        clipped /= clipped.sum()
        output.append({"composition": [
            {"element": element, "fraction": float(value)}
            for element, value in zip(FECONI_ELEMENTS, clipped) if value >= 1e-5
        ]})
    return output


def precursor_task(release: Path, task: str):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    vectorizer = DictVectorizer(sparse=True)
    train_x = vectorizer.fit_transform(frame_features(memory))
    validation_x = vectorizer.transform(frame_features(validation))
    test_x = vectorizer.transform(frame_features(test))
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    trials = []
    for k in (1, 3, 5, 10):
        model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
        prediction = _precursor_neighbors(model, validation_x, targets)
        score, maximize = primary_selection_score(task, expected, prediction)
        trials.append({"name": f"composition_knn_k{k}", "k": k, "selection_value": score, "larger_is_better": maximize})
    selected = max(trials, key=lambda row: (row["selection_value"], -row["k"]))
    model = NearestNeighbors(n_neighbors=selected["k"], metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
    predictions = _precursor_neighbors(model, test_x, targets)
    return predictions, {
        "task": task,
        "baseline_family": "composition_descriptor_knn_precursor_model",
        "selected": selected,
        "candidates": trials,
        "feature_count": len(vectorizer.feature_names_),
    }


def _precursor_neighbors(model: Any, matrix: Any, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    distances, indices = model.kneighbors(matrix)
    output = []
    for row_distances, row_indices in zip(distances, indices):
        weights = [max(1.0 - float(value), 0.0) + 1e-9 for value in row_distances]
        votes: dict[str, float] = {}
        sizes = []
        for index, weight in zip(row_indices, weights):
            formulas = targets[int(index)]["precursor_formulas"]
            sizes.append(len(formulas))
            for formula in set(formulas):
                votes[formula] = votes.get(formula, 0.0) + weight
        count = max(1, int(round(float(np.median(sizes)))))
        ranked = sorted(votes, key=lambda value: (-votes[value], value))[:count]
        output.append({"precursor_formulas": ranked})
    return output


def conditions_task(release: Path, task: str, seed: int, jobs: int):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    vectorizer = DictVectorizer(sparse=False)
    train_x = vectorizer.fit_transform(frame_features(memory))
    validation_x = vectorizer.transform(frame_features(validation))
    test_x = vectorizer.transform(frame_features(test))
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    trials, fitted = [], {}
    for leaf in (1, 2, 4):
        models = {}
        validation_predictions = [dict() for _ in range(len(validation))]
        for field in CONDITION_FIELDS:
            present = memory[field].notna().to_numpy()
            classifier = RandomForestClassifier(n_estimators=300, min_samples_leaf=leaf, random_state=seed, n_jobs=jobs, class_weight="balanced")
            classifier.fit(train_x, present.astype(int))
            regressor = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=leaf, random_state=seed, n_jobs=jobs)
            regressor.fit(train_x[present], memory.loc[present, field].astype(float).to_numpy())
            predicted_present = classifier.predict(validation_x).astype(bool)
            predicted_value = regressor.predict(validation_x)
            for index in range(len(validation)):
                validation_predictions[index][field] = float(predicted_value[index]) if predicted_present[index] else None
            models[field] = (classifier, regressor)
        score, maximize = primary_selection_score(task, expected, validation_predictions)
        name = f"two_stage_forest_leaf{leaf}"
        trials.append({"name": name, "leaf": leaf, "selection_value": score, "larger_is_better": maximize})
        fitted[name] = models
    selected = min(trials, key=lambda row: (row["selection_value"], row["name"]))
    predictions = [dict() for _ in range(len(test))]
    for field, (classifier, regressor) in fitted[selected["name"]].items():
        present = classifier.predict(test_x).astype(bool)
        values = regressor.predict(test_x)
        for index in range(len(test)):
            predictions[index][field] = float(values[index]) if present[index] else None
    return predictions, {
        "task": task,
        "baseline_family": "two_stage_presence_classifier_and_condition_regressor",
        "selected": selected,
        "candidates": trials,
        "feature_count": len(vectorizer.feature_names_),
    }


def sequence_task(release: Path, task: str):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    vectorizer = DictVectorizer(sparse=True)
    train_x = vectorizer.fit_transform(frame_features(memory))
    validation_x = vectorizer.transform(frame_features(validation))
    test_x = vectorizer.transform(frame_features(test))
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    trials = []
    for k in (1, 3, 5, 10):
        model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
        predicted = _majority_transfer(model, validation_x, targets, "operation_sequence")
        score, maximize = primary_selection_score(task, expected, predicted)
        trials.append({"name": f"composition_knn_k{k}", "k": k, "selection_value": score, "larger_is_better": maximize})
    selected = max(trials, key=lambda row: (row["selection_value"], -row["k"]))
    model = NearestNeighbors(n_neighbors=selected["k"], metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
    predictions = _majority_transfer(model, test_x, targets, "operation_sequence")
    return predictions, {
        "task": task,
        "baseline_family": "composition_descriptor_knn_sequence_model",
        "selected": selected,
        "candidates": trials,
        "feature_count": len(vectorizer.feature_names_),
    }


def _majority_transfer(model: Any, matrix: Any, targets: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    distances, indices = model.kneighbors(matrix)
    output = []
    for row_distances, row_indices in zip(distances, indices):
        votes: dict[str, float] = {}
        representatives = {}
        for distance, index in zip(row_distances, row_indices):
            target = targets[int(index)]
            key = canonical_json(target[field])
            votes[key] = votes.get(key, 0.0) + max(1.0 - float(distance), 0.0) + 1e-9
            representatives.setdefault(key, target)
        selected = max(votes, key=lambda key: (votes[key], key))
        output.append(representatives[selected])
    return output


def route_task(release: Path, task: str):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    train_payload = [_json(value) for value in memory["input_json"]]
    validation_payload = [_json(value) for value in validation["input_json"]]
    test_payload = [_json(value) for value in test["input_json"]]
    dict_vectorizer = DictVectorizer(sparse=True)
    train_cat = dict_vectorizer.fit_transform([{"zeolite=" + row["zeolite_code"]: 2.0} for row in train_payload])
    validation_cat = dict_vectorizer.transform([{"zeolite=" + row["zeolite_code"]: 2.0} for row in validation_payload])
    test_cat = dict_vectorizer.transform([{"zeolite=" + row["zeolite_code"]: 2.0} for row in test_payload])
    smiles = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1, max_features=40000, dtype=np.float32)
    train_smiles = smiles.fit_transform([row["osda_smiles"] for row in train_payload])
    validation_smiles = smiles.transform([row["osda_smiles"] for row in validation_payload])
    test_smiles = smiles.transform([row["osda_smiles"] for row in test_payload])
    train_x, validation_x, test_x = hstack([train_cat, train_smiles]), hstack([validation_cat, validation_smiles]), hstack([test_cat, test_smiles])
    targets = [memory_target(task, row) for _, row in memory.iterrows()]
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    trials = []
    for k in (3, 5, 10):
        model = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
        predicted = _route_neighbors(model, validation_x, targets)
        score, maximize = primary_selection_score(task, expected, predicted)
        trials.append({"name": f"zeolite_smiles_descriptor_knn_k{k}", "k": k, "selection_value": score, "larger_is_better": maximize})
    selected = min(trials, key=lambda row: (row["selection_value"], row["k"]))
    model = NearestNeighbors(n_neighbors=selected["k"], metric="cosine", algorithm="brute", n_jobs=4).fit(train_x)
    predictions = _route_neighbors(model, test_x, targets)
    return predictions, {
        "task": task,
        "baseline_family": "zeolite_and_smiles_descriptor_knn_distribution_model",
        "selected": selected,
        "candidates": trials,
        "feature_count": train_x.shape[1],
    }


def _route_neighbors(model: Any, matrix: Any, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, indices = model.kneighbors(matrix)
    return [{"routes": [targets[int(index)]["routes"][0] for index in row[:5]]} for row in indices]


def _token_string(value: Any) -> str:
    return value if isinstance(value, str) else "<MISSING_TOKEN>"


def token_features(tokens: list[Any], index: int) -> dict[str, Any]:
    token = _token_string(tokens[index])
    lower = token.lower()
    return {
        "token=" + lower: 1,
        "prefix2=" + lower[:2]: 1,
        "prefix3=" + lower[:3]: 1,
        "suffix2=" + lower[-2:]: 1,
        "suffix3=" + lower[-3:]: 1,
        "shape=" + "".join("A" if c.isupper() else "a" if c.islower() else "0" if c.isdigit() else c for c in token): 1,
        "prev=" + (_token_string(tokens[index - 1]).lower() if index else "<BOS>"): 1,
        "next=" + (_token_string(tokens[index + 1]).lower() if index + 1 < len(tokens) else "<EOS>"): 1,
        "has_digit": any(c.isdigit() for c in token),
        "index_bucket": min(index // 5, 10),
    }


def bio_labels(tokens: list[str], graph: dict[str, Any]) -> list[str]:
    labels = ["O"] * len(tokens)
    for node in sorted(graph.get("nodes", []), key=lambda row: (row["token_start"], -row["token_end"])):
        start, end, kind = int(node["token_start"]), int(node["token_end"]), str(node["type"])
        for index in range(max(start, 0), min(end, len(tokens))):
            if labels[index] == "O":
                labels[index] = ("B-" if index == start else "I-") + kind
    return labels


def graph_from_labels(tokens: list[str], labels: list[str]) -> dict[str, Any]:
    nodes, index = [], 0
    while index < len(tokens):
        label = labels[index]
        if label == "O":
            index += 1
            continue
        prefix, kind = label.split("-", 1)
        end = index + 1
        while end < len(tokens) and labels[end] == "I-" + kind:
            end += 1
        nodes.append({"id": f"n{len(nodes)}", "type": kind, "token_start": index, "token_end": end})
        index = end
    return {"nodes": nodes, "edges": []}


def relation_feature(tokens: list[Any], source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    a, b = int(source["token_start"]), int(target["token_start"])
    lo, hi = sorted((int(source["token_end"]), int(target["token_start"])))
    between = " ".join(_token_string(value) for value in tokens[max(0, lo):max(0, hi)]).lower()[:120]
    return {
        "source=" + source["type"]: 1,
        "target=" + target["type"]: 1,
        "type_pair=" + source["type"] + "->" + target["type"]: 1,
        "direction=" + ("forward" if a < b else "backward"): 1,
        "distance_bucket": min(abs(a - b), 20),
        "between=" + between: 1,
    }


def add_relations(tokens: list[str], graph: dict[str, Any], vectorizer: DictVectorizer, model: Any) -> None:
    pairs = [(left, right) for left in graph["nodes"] for right in graph["nodes"] if left["id"] != right["id"]]
    if not pairs:
        return
    matrix = vectorizer.transform([relation_feature(tokens, left, right) for left, right in pairs])
    labels = model.predict(matrix)
    graph["edges"] = [
        {"source": left["id"], "target": right["id"], "type": str(label)}
        for (left, right), label in zip(pairs, labels) if label != "__NONE__"
    ]


def action_graph_task(release: Path, task: str, seed: int, jobs: int):
    memory, validation, validation_labels, _ = load_task(release, task, "validation")
    _, test, _, _ = load_task(release, task, "test")
    token_x, token_y, relation_x, relation_y = [], [], [], []
    for _, row in memory.iterrows():
        payload, graph = _json(row["input_json"]), _json(row["target_json"])
        tokens = payload["tokens"]
        labels = bio_labels(tokens, graph)
        token_x.extend(token_features(tokens, index) for index in range(len(tokens)))
        token_y.extend(labels)
        by_pair = {(edge["source"], edge["target"]): edge["type"] for edge in graph.get("edges", [])}
        for source in graph.get("nodes", []):
            for target in graph.get("nodes", []):
                if source["id"] == target["id"]:
                    continue
                relation_x.append(relation_feature(tokens, source, target))
                relation_y.append(by_pair.get((source["id"], target["id"]), "__NONE__"))
    token_vectorizer = DictVectorizer(sparse=True)
    token_matrix = token_vectorizer.fit_transform(token_x)
    relation_vectorizer = DictVectorizer(sparse=True)
    relation_matrix = relation_vectorizer.fit_transform(relation_x)
    expected_map = gold_targets(task, validation_labels)
    expected = [expected_map[str(value)] for value in validation["query_id"]]
    validation_payload = [_json(value) for value in validation["input_json"]]
    trials, fitted = [], {}
    for c in (0.5, 2.0):
        token_model = LogisticRegression(C=c, max_iter=300, class_weight="balanced", random_state=seed, n_jobs=jobs)
        relation_model = LogisticRegression(C=c, max_iter=300, class_weight="balanced", random_state=seed, n_jobs=jobs)
        token_model.fit(token_matrix, token_y)
        relation_model.fit(relation_matrix, relation_y)
        predictions = _predict_graphs(validation_payload, token_vectorizer, token_model, relation_vectorizer, relation_model)
        scores = [graph_scores(pred, gold) for pred, gold in zip(predictions, expected)]
        value = float(np.mean([(row["entity_f1"] + row["relation_f1"]) / 2 for row in scores]))
        name = f"token_relation_logistic_c{c:g}"
        trials.append({"name": name, "C": c, "selection_value": value, "larger_is_better": True})
        fitted[name] = (token_model, relation_model)
    selected = max(trials, key=lambda row: (row["selection_value"], -row["C"]))
    token_model, relation_model = fitted[selected["name"]]
    test_payload = [_json(value) for value in test["input_json"]]
    predictions = _predict_graphs(test_payload, token_vectorizer, token_model, relation_vectorizer, relation_model)
    return predictions, {
        "task": task,
        "baseline_family": "supervised_token_and_relation_logistic_model",
        "selected": selected,
        "candidates": trials,
        "token_feature_count": len(token_vectorizer.feature_names_),
        "relation_feature_count": len(relation_vectorizer.feature_names_),
    }


def _predict_graphs(payloads, token_vectorizer, token_model, relation_vectorizer, relation_model):
    output = []
    for payload in payloads:
        tokens = payload["tokens"]
        if not tokens:
            output.append({"nodes": [], "edges": []})
            continue
        matrix = token_vectorizer.transform([token_features(tokens, index) for index in range(len(tokens))])
        graph = graph_from_labels(tokens, token_model.predict(matrix).tolist())
        add_relations(tokens, graph, relation_vectorizer, relation_model)
        output.append(graph)
    return output


def run_task(release: Path, task: str, seed: int, jobs: int):
    if task in {"composition_crystal_system_context_to_space_group", "composition_to_phase_assemblage"}:
        return classification_task(release, task, seed, jobs)
    if task == "target_property_to_formulation_composition":
        return inverse_composition_task(release, task, seed, jobs)
    if task == "target_composition_to_precursor_set":
        return precursor_task(release, task)
    if task == "target_and_precursors_to_heating_conditions":
        return conditions_task(release, task, seed, jobs)
    if task == "target_and_precursors_to_operations_and_conditions":
        return sequence_task(release, task)
    if task == "zeolite_and_osda_to_synthesis_route_distribution":
        return route_task(release, task)
    if task == "synthesis_text_to_action_graph":
        return action_graph_task(release, task, seed, jobs)
    raise KeyError(task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--evaluation-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cases = read_jsonl(args.evaluation_release / "test" / "cases.jsonl")
    cases_by_task = {task: [case for case in cases if case["task"] == task] for task in TASKS}
    rows, development = [], []
    for task in TASKS:
        print(canonical_json({"event": "train-specialized-baseline", "task": task}), flush=True)
        predictions, report = run_task(args.release, task, args.seed, args.jobs)
        task_cases = cases_by_task[task]
        if len(predictions) != len(task_cases):
            raise ValueError(f"Prediction count mismatch: {task}")
        model_name = report["selected"]["name"]
        for case, prediction in zip(task_cases, predictions):
            rows.append({
                "protocol": PROTOCOL,
                "case_id": case["case_id"],
                "case_sha256": digest(case),
                "task": task,
                "mode": "matched_retrained_simple_specialized_baseline",
                "model": model_name,
                "baseline_family": report["baseline_family"],
                "status": "success",
                "response": envelope(prediction, [], "Prediction from a train-only task-specific model selected on validation."),
            })
        development.append(report)
    prediction_path = args.output / "test_predictions.jsonl"
    write_jsonl(prediction_path, rows)
    (args.output / "validation_model_selection.json").write_text(
        json.dumps(development, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "scope": "Eight matched-data simple specialized baselines; not published-model reproductions.",
        "training_data": "task training memory only",
        "selection_data": "task validation split only",
        "test_labels_used_during_fit_or_selection": False,
        "seed": args.seed,
        "tasks": list(TASKS),
        "prediction_rows": len(rows),
        "source_release_manifest_sha256": file_sha256(args.release / "manifest.json"),
        "evaluation_release_manifest_sha256": file_sha256(args.evaluation_release / "manifest.json"),
        "predictions_sha256": file_sha256(prediction_path),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(canonical_json({"event": "completed", "tasks": len(TASKS), "predictions": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
