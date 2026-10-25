"""Train FAST-style models and combine them with the official baselines.

The organizer repository is read as an external checkout; no organizer data or
code is copied into this project. See MODEL_PLAN.md for the validation limits.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from data import PanelInfo, StationDay, link_features, station_window_features

STATE_KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
QUEUE_KEYS = ["window_id", "timestamp", "link_id"]
ODME_KEYS = ["panel", "departure_time", "path_id"]
STATE_COLUMNS = STATE_KEYS + ["speed_kmh", "flow_vph"]
QUEUE_COLUMNS = QUEUE_KEYS + ["queue_pred"]
ODME_COLUMNS = ODME_KEYS + ["origin_zone", "destination_zone", "path_flow"]


def panels(release: Path, selected: list[str] | None) -> list[str]:
    available = sorted(p.name for p in (release / "corridors").iterdir() if p.is_dir())
    if selected:
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"Unknown panels: {sorted(unknown)}")
        return sorted(set(selected))
    return available


def train_files(release: Path, panel: str, period: str) -> list[Path]:
    from learning import period_files
    return period_files(release, panel, period)


def day_masks(day: StationDay, panel: PanelInfo, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    rate = (0.20, 0.30, 0.50)[seed % 3]
    hidden_links = generator.random((len(day.timestamps), len(panel.links))) < rate
    visible = day.eligible & ~hidden_links[:, day.station_to_link]
    targets = day.eligible & hidden_links[:, day.station_to_link]
    return visible, targets


def center_windows(steps: int):
    for start in range(0, steps, 12):
        index = np.clip(np.arange(start - 30, start + 42), 0, steps - 1)
        yield index, slice(30, 30 + min(12, steps - start)), slice(start, min(start + 12, steps))


def tensors(device: str, *arrays: np.ndarray):
    import torch
    return [torch.as_tensor(array, device=device) for array in arrays]


def fit_state(args: argparse.Namespace) -> None:
    from learning import train
    train(args, "state")


def queue_windows(day: StationDay, panel: PanelInfo, per_condition: int, rng: np.random.Generator):
    steps, nodes = len(day.timestamps), len(panel.links)
    speed = np.zeros((steps, nodes), dtype=np.float32)
    count = np.zeros((steps, nodes), dtype=np.float32)
    for station, link in enumerate(day.station_to_link):
        good = day.eligible[:, station]
        speed[good, link] += day.speed[good, station]
        count[good, link] += 1
    valid = count > 0
    speed /= np.maximum(count, 1)
    queue = (speed * 25.0 <= panel.cutoffs[None, :]) & valid
    candidates: dict[str, list[int]] = {"onset": [], "ongoing": []}
    for origin in range(12, steps - 6):
        # The release gives [T-60, T); targets start at T+5, not at T.
        future = queue[origin + 1:origin + 7]
        if not future.any():
            continue
        history = queue[origin - 12:origin]
        if not history.any():
            candidates["onset"].append(origin)
        elif history[-1].any():
            candidates["ongoing"].append(origin)
    chosen = []
    for condition, origins in candidates.items():
        if origins:
            chosen.extend((condition, int(origin)) for origin in rng.choice(origins, min(per_condition, len(origins)), replace=False))
    for condition, origin in chosen:
        indices = slice(origin - 12, origin)
        history = link_features(day.timestamps[indices], speed[indices], np.zeros_like(speed[indices]),
                                valid[indices], np.arange(nodes), panel)
        future = slice(origin + 1, origin + 7)
        yield condition, history, queue[future], speed[future], valid[future]


def mean_iou(logits: list[np.ndarray], labels: list[np.ndarray], valid: list[np.ndarray], threshold: float) -> float:
    scores = []
    for output, truth, mask in zip(logits, labels, valid):
        predicted = (1.0 / (1.0 + np.exp(-np.clip(output, -30, 30))) >= threshold) & mask
        actual = truth & mask
        union = np.logical_or(predicted, actual).sum()
        scores.append(float(np.logical_and(predicted, actual).sum() / union) if union else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def fit_queue(args: argparse.Namespace) -> None:
    from learning import train
    train(args, "queue")


def load_model(path: Path, task: str, device: str):
    import torch
    from fast_model import QueueModel, StateModel
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["task"] != task:
        raise ValueError(f"Expected {task} checkpoint: {path}")
    if checkpoint.get("format_version") == 2:
        from learning import model_from_checkpoint
        return model_from_checkpoint(checkpoint, device).eval(), checkpoint
    model = (StateModel if task == "state" else QueueModel)(checkpoint["width"], checkpoint["blocks"])
    model.load_state_dict(checkpoint["weights"])
    return model.to(device).eval(), checkpoint


def predict_state(release: Path, split: str, checkpoint: Path, output: Path) -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(checkpoint, "state", device)
    if config.get("format_version") == 2:
        from learning import predict_state_v2
        predict_state_v2(model, config, release, split, output)
        return
    parts = []
    for name in panels(release, None):
        panel = PanelInfo.load(release, name)
        template_path = release / "task1" / name / split / "sample_submission_state.csv"
        template = pd.read_csv(template_path, dtype=str)
        target_by_date = {date: group.copy() for date, group in template.groupby(template.timestamp.str[:10])}
        files = sorted((release / "corridors" / name / split / "mainline_states_masked").glob("**/*.parquet"))
        found = set()
        for path in files:
            day = StationDay.load(path, panel)
            date = day.timestamps[0][:10]
            if date not in target_by_date:
                continue
            found.add(date)
            visible = day.eligible
            prediction = np.zeros((*day.speed.shape, 2), dtype=np.float32)
            station_link = torch.as_tensor(day.station_to_link, device=device)
            with torch.no_grad():
                for index, middle, absolute in center_windows(len(day.timestamps)):
                    inputs, context = station_window_features(day, panel, visible, index)
                    x, station = tensors(device, inputs[None], context[None])
                    values = model(x, station, station_link, panel.segments, config["bidirectional"])
                    prediction[absolute] = values[0, middle].cpu().numpy()
            target = target_by_date[date]
            timestamp_index = {value: i for i, value in enumerate(day.timestamps)}
            station_index = {value: i for i, value in enumerate(day.station_ids)}
            if not set(target.timestamp).issubset(timestamp_index) or not set(target.station_id).issubset(station_index):
                raise ValueError(f"Template keys absent from masked observations: {name} {date}")
            ti = target.timestamp.map(timestamp_index).to_numpy(dtype=int)
            si = target.station_id.map(station_index).to_numpy(dtype=int)
            if any(day.station_links[s] != link for s, link in zip(si, target.link_id)):
                raise ValueError(f"Station/link mismatch: {name} {date}")
            target["speed_kmh"] = prediction[ti, si, 0] * 25.0
            target["flow_vph"] = prediction[ti, si, 1] * 600.0 * day.lanes[si]
            parts.append(target[STATE_COLUMNS])
        if found != set(target_by_date):
            raise ValueError(f"Missing masked days for {name} {split}: {sorted(set(target_by_date) - found)}")
    result = pd.concat(parts, ignore_index=True)
    if result[STATE_KEYS].duplicated().any() or result[["speed_kmh", "flow_vph"]].isna().any().any():
        raise ValueError("Invalid State predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def predict_queue(release: Path, split: str, checkpoint: Path, output: Path) -> None:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(checkpoint, "queue", device)
    if config.get("format_version") == 2:
        from learning import predict_queue_v2
        predict_queue_v2(model, config, release, split, output)
        return
    parts = []
    for name in panels(release, None):
        root = release / "task2" / name / split
        if not (root / "sample_submission_queue.csv").exists():
            continue
        template = pd.read_csv(root / "sample_submission_queue.csv", dtype=str)
        history = pd.read_parquet(root / "window_history.parquet", columns=["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        history["window_id"] = history.window_id.astype(str)
        history["link_id"] = history.link_id.astype(str)
        history["timestamp"] = history.timestamp.astype(str)
        panel = PanelInfo.load(release, name)
        windows = {key: group for key, group in history.groupby("window_id")}
        for window_id, target in template.groupby("window_id", sort=False):
            if window_id not in windows:
                raise ValueError(f"No history for {window_id}")
            frame = windows[window_id]
            times = sorted(frame.timestamp.unique().tolist())
            if len(times) != 12 or times[-1] >= min(target.timestamp):
                raise ValueError(f"Unexpected Task 2 history boundary: {window_id}")
            nodes = len(panel.links)
            speed = np.zeros((12, nodes), dtype=np.float32)
            count = np.zeros((12, nodes), dtype=np.float32)
            time_index = {value: i for i, value in enumerate(times)}
            for row in frame.itertuples(index=False):
                if row.link_id in panel.link_index and bool(row.is_score_eligible) and pd.notna(row.speed_kmh):
                    index = time_index[row.timestamp], panel.link_index[row.link_id]
                    speed[index] += float(row.speed_kmh) / 25.0
                    count[index] += 1
            valid = count > 0
            speed /= np.maximum(count, 1)
            x = link_features(times, speed, np.zeros_like(speed), valid, np.arange(nodes), panel)
            with torch.no_grad():
                logits, _ = model(tensors(device, x[None])[0], panel.segments, config["bidirectional"])
            result = target.copy()
            future = sorted(result.timestamp.unique().tolist())
            if len(future) != 6 or not set(result.link_id).issubset(panel.link_index):
                raise ValueError(f"Unexpected Task 2 targets: {window_id}")
            future_index = result.timestamp.map({value: i for i, value in enumerate(future)}).to_numpy(dtype=int)
            link_index = result.link_id.map(panel.link_index).to_numpy(dtype=int)
            probability = torch.sigmoid(logits[0]).cpu().numpy()
            result["queue_pred"] = (probability[future_index, link_index] >= config["threshold"]).astype(int)
            parts.append(result[QUEUE_COLUMNS])
    if not parts:
        raise ValueError(f"No Task 2 templates for {split}")
    result = pd.concat(parts, ignore_index=True)
    if result[QUEUE_KEYS].duplicated().any():
        raise ValueError("Duplicate queue keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def odme_problem(release: Path, name: str, split: str):
    from scipy.sparse import csr_matrix
    network = release / "corridors" / name / "network"
    root = release / "task4" / name / split
    template = pd.read_csv(root / "sample_submission_path_flow.csv", dtype=str)
    if template.empty or template[ODME_COLUMNS[:-1]].isna().any().any():
        raise ValueError(f"Empty or incomplete ODME template: {name} {split}")
    if not template.panel.eq(name).all() or template.departure_time.nunique() != 1:
        raise ValueError(f"Mismatched ODME template period/panel: {name} {split}")
    if template[ODME_KEYS].duplicated().any() or template.path_id.duplicated().any():
        raise ValueError(f"Duplicate ODME template paths: {name} {split}")
    counts = pd.read_csv(root / "synthetic_link_counts.csv", dtype={"link_id": str})
    prior = pd.read_csv(root / "synthetic_weak_prior.csv", dtype={"path_id": str, "departure_time": str})
    incidence = pd.read_csv(network / "path_link_incidence.csv", dtype={"path_id": str, "link_id": str})
    paths = template.path_id.tolist()
    measured = counts.dropna(subset=["count"])
    if measured.link_id.duplicated().any() or prior.path_id.duplicated().any():
        raise ValueError("Duplicate measured links or priors")
    measured = measured[measured.link_id.isin(incidence.link_id)]
    if measured.empty:
        raise ValueError(f"No measured ODME links: {name} {split}")
    row = {link: i for i, link in enumerate(measured.link_id)}
    column = {path: i for i, path in enumerate(paths)}
    included = incidence[incidence.link_id.isin(row) & incidence.path_id.isin(column)]
    matrix = csr_matrix((np.ones(len(included)),
                         (included.link_id.map(row), included.path_id.map(column))),
                        shape=(len(row), len(column)))
    matrix.data[:] = 1.0  # Repeated incidence rows still mean one traversal.
    prior_values = pd.to_numeric(prior.path_flow, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(prior_values).all() or (prior_values < 0).any():
        raise ValueError(f"Invalid ODME prior: {name} {split}")
    if "departure_time" in prior and not prior.departure_time.eq(template.departure_time.iloc[0]).all():
        raise ValueError(f"Mismatched ODME prior period: {name} {split}")
    base = pd.Series(prior_values, index=prior.path_id).reindex(paths, fill_value=0).to_numpy(dtype=np.float64)
    observed = pd.to_numeric(measured["count"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(observed).all() or not np.isfinite(base).all() or (observed < 0).any() or (base < 0).any():
        raise ValueError(f"Invalid counts/prior: {name} {split}")
    return template, matrix, observed, base


def odme_prior_variance(base, power=0, floor_ratio=0.1):
    base = np.asarray(base, dtype=np.float64)
    if base.ndim != 1 or not base.size or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("ODME prior must be a nonempty finite nonnegative vector")
    if power not in (0, 1, 2) or not np.isfinite(floor_ratio) or floor_ratio <= 0:
        raise ValueError("ODME prior power must be 0, 1, or 2, and floor ratio must be positive")
    if power == 0:
        return np.ones_like(base)
    positive = base[base > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    variance = (base / scale + floor_ratio) ** power
    variance /= np.median(variance)
    if not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("ODME prior variance is outside the numerical range")
    return variance


def solve_odme(matrix, observed, base, lam, prior_power=0, prior_floor_ratio=0.1):
    from scipy.optimize import lsq_linear, minimize
    from scipy.sparse import csr_matrix
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("ODME lambda must be finite and nonnegative")
    variance = odme_prior_variance(base, prior_power, prior_floor_ratio)
    base = np.asarray(base, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    design = csr_matrix(matrix, dtype=np.float64)
    if (observed.ndim != 1 or not observed.size or design.shape != (len(observed), len(base))
            or not np.isfinite(observed).all() or (observed < 0).any()
            or not np.isfinite(design.data).all() or (design.data < 0).any()):
        raise ValueError("Invalid ODME matrix/count dimensions or values")
    if lam == 0:
        result = lsq_linear(design, observed, bounds=(0, np.inf), tol=1e-8, max_iter=2000)
        if not result.success:
            raise RuntimeError(f"ODME solver did not converge: {result.message}")
        return result.x
    precision = lam / variance
    if not np.isfinite(precision).all() or (precision <= 0).any():
        raise ValueError("ODME prior precision is outside the numerical range")
    # The dual has one variable per measured link, independent of path count.
    def objective(dual):
        flow = np.maximum(base - np.asarray(design.T @ dual).ravel() / precision, 0)
        cost = 0.5 * (dual @ dual) + observed @ dual + 0.5 * np.dot(precision * (flow - base), flow + base)
        return cost, dual + observed - design @ flow
    result = minimize(objective, np.zeros(len(observed)), jac=True, method="BFGS",
                      options={"gtol": 1e-5, "maxiter": 2000})
    flow = np.maximum(base - np.asarray(design.T @ result.x).ravel() / precision, 0)
    residual, difference = design @ flow - observed, flow - base
    primal = 0.5 * (residual @ residual + np.dot(precision * difference, difference))
    # Clipped flows minimize the Lagrangian; this gap avoids subtracting large costs.
    gap = 0.5 * np.sum((result.x - residual) ** 2)
    if not np.isfinite(flow).all() or not np.isfinite(primal + gap) or gap > 1e-7 * (1 + primal):
        raise RuntimeError(f"ODME solver did not converge: {result.message}")
    return flow


def predict_odme(release: Path, split: str, output: Path, lam: float = 0.05,
                 prior_power=0, prior_floor_ratio=0.1, selected=None, prior_only=False) -> None:
    parts = []
    for name in panels(release, selected):
        template, matrix, observed, base = odme_problem(release, name, split)
        solution = base.copy() if prior_only else solve_odme(matrix, observed, base, lam, prior_power, prior_floor_ratio)
        template["path_flow"] = solution
        parts.append(template[ODME_COLUMNS])
    if not parts:
        raise ValueError(f"No ODME panels for {split}")
    result = pd.concat(parts, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def odme_structure(template, matrix):
    columns = matrix.tocsc()
    signatures = {}
    for index in range(columns.shape[1]):
        links = tuple(columns.indices[columns.indptr[index]:columns.indptr[index + 1]])
        signatures[links] = signatures.get(links, 0) + 1
    od_sizes = template.groupby(["origin_zone", "destination_zone"]).size()
    return {"measured_links": matrix.shape[0], "paths": matrix.shape[1],
            "unobserved_paths": int(np.sum(np.diff(columns.indptr) == 0)),
            "distinct_measured_path_signatures": len(signatures),
            "ambiguous_path_groups": sum(size > 1 for size in signatures.values()),
            "paths_in_ambiguous_groups": sum(size for size in signatures.values() if size > 1),
            "od_pairs": len(od_sizes), "od_pairs_with_multiple_paths": int((od_sizes > 1).sum()),
            "max_paths_per_od": int(od_sizes.max()),
            "destinations": int(template.destination_zone.nunique())}


def odme_metrics(template, matrix, observed, base, estimate, fit):
    loaded = np.asarray(matrix @ estimate).ravel()
    metrics = {}
    for label, mask in (("fit", fit), ("hidden", ~fit)):
        residual = loaded[mask] - observed[mask]
        normalized_l1 = float(np.sum(np.abs(residual)) / max(float(observed[mask].sum()), 1e-9))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        metrics.update({f"{label}_link_normalized_l1": normalized_l1,
                        f"{label}_link_s_link": max(0.0, 1.0 - normalized_l1),
                        f"{label}_link_rmse_vph": rmse,
                        f"{label}_link_normalized_rmse": rmse / max(float(observed[mask].mean()), 1)})
    movement = estimate - base
    prior_total, estimate_total = float(base.sum()), float(estimate.sum())
    destinations = template.destination_zone.to_numpy()
    prior_attr = pd.Series(base).groupby(destinations).sum().to_numpy() / max(prior_total, 1e-9)
    estimate_attr = pd.Series(estimate).groupby(destinations).sum().to_numpy() / max(estimate_total, 1e-9)
    metrics.update({"prior_movement_l1": float(np.abs(movement).sum()),
                    "prior_relative_l1": float(np.abs(movement).sum() / max(prior_total, 1e-9)),
                    "prior_relative_l2": float(np.linalg.norm(movement) / max(np.linalg.norm(base), 1)),
                    "destination_distribution_l1_from_prior": float(np.abs(estimate_attr - prior_attr).sum())
                    if prior_total > 0 and estimate_total > 0 else None,
                    "zero_path_fraction": float(np.mean(estimate == 0))})
    return metrics


def validate_odme(args):
    from learning import family_mean, stable_seed, write_json
    if args.repeats < 1 or not args.lambdas or not args.prior_powers:
        raise ValueError("ODME validation needs candidates and at least one repeat")
    if any(not np.isfinite(lam) or lam < 0 for lam in args.lambdas):
        raise ValueError("ODME lambdas must be finite and nonnegative")
    configurations = sorted({(float(lam), power) for lam in args.lambdas for power in args.prior_powers}
                            | {(0.05, 0)})
    candidates = [{"method": "prior", "lambda": None, "prior_power": None}]
    candidates += [{"method": "nnls", "lambda": lam, "prior_power": power} for lam, power in configurations]
    rows, structures = [], {}
    for name in panels(args.release_root, args.panel):
        template, matrix, observed, base = odme_problem(args.release_root, name, args.split)
        if len(observed) < 2:
            raise ValueError("ODME link holdout needs at least two measured links")
        odme_prior_variance(base, 0, args.prior_floor_ratio)
        structures[name] = odme_structure(template, matrix)
        for repeat in range(args.repeats):
            # The first mask stays comparable with earlier lambda-only reports.
            seed = stable_seed(args.seed, name, args.split) if repeat == 0 else stable_seed(args.seed, name, args.split, repeat)
            rng = np.random.default_rng(seed)
            hidden = rng.choice(len(observed), max(1, len(observed) // 5), replace=False)
            fit = np.ones(len(observed), bool)
            fit[hidden] = False
            for candidate in candidates:
                estimate = base.copy() if candidate["method"] == "prior" else solve_odme(
                    matrix[fit], observed[fit], base, candidate["lambda"], candidate["prior_power"], args.prior_floor_ratio)
                rows.append({"panel": name, "repeat": repeat, **candidate,
                             "hidden_link_indices": hidden.tolist(),
                             **odme_metrics(template, matrix, observed, base, estimate, fit)})
    if not rows:
        raise ValueError("No ODME panels to validate")
    summaries = []
    for candidate in candidates:
        matches = [row for row in rows if all(row[key] == value for key, value in candidate.items())]
        summary = dict(candidate)
        for metric in ("hidden_link_normalized_l1", "hidden_link_normalized_rmse", "hidden_link_s_link",
                       "fit_link_normalized_l1", "prior_movement_l1", "prior_relative_l1", "prior_relative_l2",
                       "destination_distribution_l1_from_prior", "zero_path_fraction"):
            usable = [row for row in matches if row[metric] is not None]
            panel_values = {}
            for row in usable:
                panel_values.setdefault(row["panel"], []).append(row[metric])
            summary[metric] = family_mean({name: float(np.mean(values)) for name, values in panel_values.items()}) if usable else None
        repeat_errors = [family_mean({row["panel"]: row["hidden_link_normalized_l1"] for row in matches
                                     if row["repeat"] == repeat}) for repeat in range(args.repeats)]
        summary["hidden_link_normalized_l1_std"] = float(np.std(repeat_errors))
        summary["worst_panel_hidden_link_normalized_l1"] = max(
            float(np.mean([row["hidden_link_normalized_l1"] for row in matches if row["panel"] == name]))
            for name in structures)
        summaries.append(summary)
    best = min(summaries, key=lambda row: row["hidden_link_normalized_l1"])
    report = {"split": args.split, "proxy_only": True, "selection_metric": "hidden_link_normalized_l1",
              "seed": args.seed, "repeats": args.repeats, "prior_floor_ratio": args.prior_floor_ratio,
              "best_proxy_candidate": {key: best[key] for key in ("method", "lambda", "prior_power")},
              "candidates": summaries, "structure": structures, "panels": rows}
    write_json(args.output, report)
    print(report)


def ensemble_predictions(release, split, checkpoints, output, task):
    import torch
    if len(checkpoints) == 1:
        (predict_state if task == "state" else predict_queue)(release, split, checkpoints[0], output)
        return
    keys, values = (STATE_KEYS, ["speed_kmh", "flow_vph"]) if task == "state" else (QUEUE_KEYS, ["queue_pred"])
    reference, total, thresholds = None, None, []
    for number, path in enumerate(checkpoints):
        member = output.with_name(f"{output.stem}.member{number}.csv")
        if task == "state":
            predict_state(release, split, path, member)
        else:
            from learning import predict_queue_v2
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, config = load_model(path, task, device)
            if config.get("format_version") != 2:
                raise ValueError("Queue probability averaging requires v2 checkpoints")
            predict_queue_v2(model, config, release, split, member, probabilities=True)
            thresholds.append(config["threshold"])
        frame = pd.read_csv(member, dtype={key: str for key in keys}).sort_values(keys).reset_index(drop=True)
        if frame[keys].duplicated().any():
            raise ValueError("Duplicate ensemble keys")
        if reference is None:
            reference = frame[keys]
            total = frame[values].to_numpy(np.float64)
        elif not frame[keys].equals(reference):
            raise ValueError("Ensemble members have different target keys")
        else:
            total += frame[values].to_numpy(np.float64)
    total /= len(checkpoints)
    if task == "queue":
        total = (total >= np.mean(thresholds)).astype(int)
    result = reference.copy()
    result[values] = total
    result.to_csv(output, index=False)


def official(python: str, script: Path, *arguments: str) -> None:
    subprocess.run([python, str(script), *arguments], check=True)


def external_state_inputs(release: Path, paths: list[Path]) -> dict[str, pd.DataFrame]:
    parts = []
    for path in paths:
        frame = pd.read_csv(path, dtype={key: str for key in STATE_KEYS})
        if set(STATE_COLUMNS) - set(frame):
            raise ValueError(f"Missing Task 1 columns: {path}")
        parts.append(frame[STATE_COLUMNS])
    state = pd.concat(parts, ignore_index=True)
    if state.empty or state[STATE_KEYS].isna().any().any() or state.duplicated(STATE_KEYS).any():
        raise ValueError("Empty, incomplete, or duplicate external Task 1 keys")
    for column in ("speed_kmh", "flow_vph"):
        state[column] = pd.to_numeric(state[column], errors="coerce")
    values = state[["speed_kmh", "flow_vph"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("External Task 1 values must be finite and nonnegative")
    result = {}
    for split in ("validation", "private"):
        templates = [pd.read_csv(release / "task1" / name / split / "sample_submission_state.csv",
                                 usecols=STATE_KEYS, dtype=str) for name in panels(release, None)]
        expected = pd.concat(templates, ignore_index=True)
        aligned = expected.merge(state, on=STATE_KEYS, how="left", validate="one_to_one", sort=False)
        if aligned[["speed_kmh", "flow_vph"]].isna().any().any():
            raise ValueError(f"External Task 1 files do not cover every {split} template key")
        result[split] = aligned[STATE_COLUMNS]
    if sum(len(frame) for frame in result.values()) != len(state):
        raise ValueError("External Task 1 files contain keys outside validation/private templates")
    return result


def make_submission(args: argparse.Namespace) -> None:
    official_root = args.official_root.resolve()
    release = args.release_root.resolve()
    for required in (official_root / "src" / "merge_submissions.py", release / "submission_key.csv"):
        if not required.exists():
            raise FileNotFoundError(required)
    state_inputs = external_state_inputs(release, args.state_csv) if args.state_csv else None
    combined: dict[str, list[Path]] = {"state": [], "queue": [], "odme": []}
    for split in ("validation", "private"):
        folder = args.output_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        state, queue, odme = (folder / f"{task}.csv" for task in combined)
        if state_inputs is not None:
            state_inputs[split].to_csv(state, index=False)
        elif args.state_checkpoint:
            ensemble_predictions(release, split, args.state_checkpoint, state, "state")
        else:
            official(sys.executable, official_root / "src" / "task1" / "build_task1_baseline_submission.py",
                     "--release-root", str(release), "--split", split, "--output", str(state))
        if args.queue_checkpoint:
            ensemble_predictions(release, split, args.queue_checkpoint, queue, "queue")
        else:
            official(sys.executable, official_root / "src" / "task2" / "build_task2_persistence_submission.py",
                     "--release-root", str(release), "--split", split, "--output", str(queue))
        predict_odme(release, split, odme, args.odme_lambda, args.odme_prior_power, args.odme_prior_floor_ratio)
        for task, path in zip(combined, (state, queue, odme)):
            combined[task].append(path)
    merged = {}
    for task, files in combined.items():
        table = pd.concat([pd.read_csv(path, dtype=str) for path in files], ignore_index=True)
        keys = {"state": STATE_KEYS, "queue": QUEUE_KEYS, "odme": ODME_KEYS}[task]
        if table[keys].duplicated().any():
            raise ValueError(f"Duplicate {task} keys across splits")
        destination = args.output_dir / f"{task}_both.csv"
        table.to_csv(destination, index=False)
        merged[task] = destination
    temporary = args.output_dir / "submission.pending.csv"
    command = [sys.executable, str(official_root / "src" / "merge_submissions.py"),
               "--state", str(merged["state"]), "--queue", str(merged["queue"]),
               "--odme", str(merged["odme"]), "--key", str(release / "submission_key.csv"),
               "--output", str(temporary)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    print(completed.stdout)
    if "WARNING:" in completed.stdout or "Every scored cell was filled from your files." not in completed.stdout:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Official merger found missing predictions; upload CSV was not finalized")
    os.replace(temporary, args.output_dir / "submission.csv")
    print(f"Complete submission: {args.output_dir / 'submission.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-state", "train-queue", "submit", "evaluate", "validate-odme", "predict-odme"):
        command = commands.add_parser(name)
        command.add_argument("--release-root", type=Path, required=True)
        if name.startswith("train"):
            command.add_argument("--panel", action="append")
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--epochs", type=int, default=75, help="Total epoch budget, including resumed epochs")
            command.add_argument("--width", type=int, default=64)
            command.add_argument("--blocks", type=int, default=2)
            command.add_argument("--heads", type=int, default=4)
            command.add_argument("--dropout", type=float, default=0.1)
            command.add_argument("--backend", choices=["mamba2", "mamba1", "scan"], default="mamba2")
            command.add_argument("--graph", action="store_true")
            command.add_argument("--graph-mode", choices=["mean", "reliability"], default="mean")
            command.add_argument("--missing-aware", action="store_true", help="Learn separate missing-channel embeddings")
            command.add_argument("--station-decoder", choices=["linear", "gated", "split"], default="linear")
            command.add_argument("--adaptive-graph-k", type=int, default=0, help="0 disables learned edges; otherwise select at least 2 neighbors")
            command.add_argument("--wavelet-levels", type=int, choices=[0, 1, 2, 3], default=0, help="Haar trend/detail levels; 0 disables wavelet")
            command.add_argument("--wavelet-input", choices=["signals", "embedded"], help="New training defaults to signals; resume/transfer inherits the checkpoint")
            command.add_argument("--state-regime-balance", action="store_true", help="Fit-only macro regime MSE weighting for State")
            command.add_argument("--lr", type=float, default=1e-3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--max-days", type=int)
            command.add_argument("--batch-size", type=int, default=0, help="0 selects a panel-size-dependent batch")
            command.add_argument("--panel-batch", action="append", default=[], metavar="PANEL=N")
            command.add_argument("--update-windows", type=int,
                                 help="Windows per optimizer update; 0 selects State=8/Queue=48; omitted keeps legacy updates")
            command.add_argument("--workers", type=int, choices=[0, 1, 2], default=1)
            command.add_argument("--cpu-threads", type=int, default=2)
            command.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
            command.add_argument("--bf16", action="store_true", help="Use CUDA bfloat16 autocast for model forward passes")
            command.add_argument("--patience", type=int, default=5, help="0 disables early stopping")
            command.add_argument("--scheduler-patience", type=int, default=2)
            command.add_argument("--grad-clip", type=float, default=1.0)
            command.add_argument("--log-every", type=int, default=25)
            command.add_argument("--resume", type=Path)
            command.add_argument("--init-state", type=Path, help="Initialize Queue encoder from a v2 State checkpoint")
            command.add_argument("--lambda-n", type=float, default=0.0)
            command.add_argument("--lambda-dn", type=float, default=0.0)
            command.add_argument("--lambda-fd", type=float, default=0.0)
            command.add_argument("--windows-per-day", type=int, default=6)
            command.add_argument("--normal-windows", type=int, default=1)
            command.add_argument("--normal-weight", type=float, default=0.2)
            command.add_argument("--validation-windows", type=int, default=5, help="Fixed maximum per condition/panel")
            command.add_argument("--window-gap-minutes", type=int, default=360)
            command.add_argument("--history-coverage", type=float, default=0.5)
            command.add_argument("--future-coverage", type=float, default=0.5)
            command.add_argument("--label-temperature", type=float, default=0.0, help="km/h; 0 uses hard labels")
            command.add_argument("--uncertainty-margin", type=float, default=0.0, help="km/h; 0 disables uncertainty weighting")
            command.add_argument("--soft-iou-weight", type=float, default=0.2)
            command.add_argument("--speed-weight", type=float, default=0.1)
            command.add_argument("--forward-only", dest="bidirectional", action="store_false")
            command.set_defaults(bidirectional=True)
            command.set_defaults(queue_origin=False, queue_resample=False, queue_condition_gap=False,
                                 queue_changes=False, queue_head="linear", queue_validation_spread=False)
            if name == "train-queue":
                command.add_argument("--queue-origin", action="store_true", help="Append the released masked observation at T")
                command.add_argument("--queue-resample", action="store_true", help="Resample fit windows each epoch; keep evaluation fixed")
                command.add_argument("--queue-condition-gap", action="store_true", help="Apply fit window spacing within each condition")
                command.add_argument("--queue-changes", action="store_true", help="Add threshold margin and masked 5/15-minute changes")
                command.add_argument("--queue-head", choices=["linear", "pooled-mlp"], default="linear")
                command.add_argument("--queue-validation-spread", action="store_true", help="Spread fixed validation windows across dates")
        elif name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument("--period", choices=["select", "holdout"], default="select")
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--panel", action="append")
            command.add_argument("--max-days", type=int)
            command.add_argument("--workers", type=int, choices=[0, 1, 2], default=1)
            command.add_argument("--bootstrap", type=int, default=0)
            command.add_argument("--validation-windows", type=int, help="Override the saved per-condition Queue validation limit")
            command.add_argument("--queue-validation-spread", action="store_true", default=None,
                                 help="Evaluate Queue on fixed windows spread across dates")
        elif name == "validate-odme":
            command.add_argument("--split", choices=["validation", "private"], default="validation")
            command.add_argument("--panel", action="append")
            command.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.05, 0.2])
            command.add_argument("--prior-powers", type=int, choices=[0, 1, 2], nargs="+", default=[0])
            command.add_argument("--prior-floor-ratio", type=float, default=0.1)
            command.add_argument("--repeats", type=int, default=3)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--output", type=Path, required=True)
        elif name == "predict-odme":
            command.add_argument("--split", choices=["validation", "private"], required=True)
            command.add_argument("--panel", action="append")
            command.add_argument("--odme-lambda", type=float, default=0.05)
            command.add_argument("--prior-power", type=int, choices=[0, 1, 2], default=0)
            command.add_argument("--prior-floor-ratio", type=float, default=0.1)
            command.add_argument("--prior-only", action="store_true", help="Write the released prior without solving")
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--official-root", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            state_source = command.add_mutually_exclusive_group()
            state_source.add_argument("--state-checkpoint", type=Path, action="append")
            state_source.add_argument("--state-csv", type=Path, action="append",
                                      help="Task 1 CSVs covering all validation/private targets, including MagiNet outputs")
            command.add_argument("--queue-checkpoint", type=Path, action="append")
            command.add_argument("--odme-lambda", type=float, default=0.05)
            command.add_argument("--odme-prior-power", type=int, choices=[0, 1, 2], default=0)
            command.add_argument("--odme-prior-floor-ratio", type=float, default=0.1)
    args = parser.parse_args()
    if args.command == "train-state":
        fit_state(args)
    elif args.command == "train-queue":
        fit_queue(args)
    elif args.command == "evaluate":
        from learning import evaluate
        evaluate(args)
    elif args.command == "validate-odme":
        validate_odme(args)
    elif args.command == "predict-odme":
        predict_odme(args.release_root, args.split, args.output, args.odme_lambda,
                     args.prior_power, args.prior_floor_ratio, args.panel, args.prior_only)
    else:
        make_submission(args)


if __name__ == "__main__":
    main()
