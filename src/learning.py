"""Versioned training, exact-template State validation, and Queue proxy validation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from data import (FEATURE_NAMES, STATE_CONTEXT_NAMES, PanelInfo, StationDay, TargetIndex, aggregate_channels,
                  features_v2, queue_features, queue_feature_names, ramp_features, slice_day, state_features_v2)
from fast_model import TrafficQueueModel, TrafficStateModel


def stable_seed(*parts):
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest(), "big")


@lru_cache(maxsize=64)
def file_index(release: Path, name: str):
    result = {}
    for path in sorted((release / "corridors" / name / "train" / "mainline_states").glob("**/*.parquet")):
        match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", path.name)
        date = match.group().replace("_", "-") if match else str(pd.read_parquet(path, columns=["timestamp"]).timestamp.iloc[0])[:10]
        if date in result:
            raise ValueError(f"Duplicate day: {name} {date}")
        result[date] = path
    return result


def period_files(release, name, period, maximum=None):
    limits = {"fit": ("2030-06-01", "2030-12-31"), "select": ("2031-01-01", "2031-01-31"),
              "holdout": ("2031-02-01", "2031-02-28")}
    low, high = limits[period]
    paths = [path for date, path in file_index(release, name).items() if low <= date <= high]
    return paths[:maximum] if maximum else paths


def signature(paths):
    return hashlib.sha256(json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]).encode()).hexdigest()


def fit_profile(release, panel, paths):
    """Fit-only weekday/slot means; the checkpoint embeds the resulting array."""
    key = hashlib.sha256(("eligible-v2|" + signature(paths) + json.dumps(panel.links)).encode()).hexdigest()[:16]
    folder = release.parent / "processed_v2"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"{panel.name}-profile-{key}.npy"
    if cache.exists():
        return np.load(cache, allow_pickle=False)
    print(f"Preparing fit-only means: {panel.name} days={len(paths)}", flush=True)
    sums = np.zeros((2016, len(panel.links), 3), np.float64)
    counts = np.zeros_like(sums)
    for number, path in enumerate(paths, 1):
        day = StationDay.load(path, panel)
        values, valid, _ = aggregate_channels(day, panel, day.eligible)
        dates = pd.to_datetime(day.timestamps, utc=True)
        slots = np.asarray(dates.dayofweek * 288 + dates.hour * 12 + dates.minute // 5)
        np.add.at(sums, slots, values * valid)
        np.add.at(counts, slots, valid)
        if number % 50 == 0:
            print(f"fit-only means {panel.name}: {number}/{len(paths)}", flush=True)
    global_mean = sums.sum((0, 1)) / np.maximum(counts.sum((0, 1)), 1)
    link_mean = np.divide(sums.sum(0), counts.sum(0), out=np.broadcast_to(global_mean, sums.shape[1:]).copy(), where=counts.sum(0) > 0)
    result = np.divide(sums, counts, out=np.broadcast_to(link_mean, sums.shape).copy(), where=counts > 0).astype(np.float32)
    temporary = cache.with_suffix(".pending.npy")
    np.save(temporary, result)
    temporary.replace(cache)
    return result


def center_windows(steps):
    for start in range(0, steps, 12):
        yield np.clip(np.arange(start - 30, start + 42), 0, steps - 1), slice(start, min(start + 12, steps))


def state_examples(context, path):
    panel, release = context["panel"], context["release"]
    truth = StationDay.load(path, panel)
    match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", path.name)
    if match and truth.timestamps[0][:10] != match.group().replace("_", "-"):
        raise ValueError(f"File date does not match observations: {path}")
    regime, targets = context["targets"].get(truth)
    masked = StationDay.load(context["masked"][path.name], panel)
    if masked.timestamps != truth.timestamps or masked.station_ids != truth.station_ids:
        raise ValueError(f"Masked/unmasked mapping differs: {path}")
    if np.isfinite(masked.speed[targets]).any() or np.isfinite(masked.flow_lane[targets]).any():
        raise ValueError("A template answer is visible in the masked input")
    ramp = ramp_features(release, panel, "train", truth.timestamps)
    samples = []
    for index, absolute in center_windows(len(truth.timestamps)):
        if not targets[absolute].any():
            continue
        x, station = state_features_v2(masked, panel, index, context["profile"], ramp,
                                      rich=context.get("station_decoder", "linear") != "linear")
        length = absolute.stop - absolute.start
        y = np.zeros((12, len(truth.station_ids), 2), np.float32)
        mask = np.zeros(y.shape[:2], bool)
        eligible = np.zeros_like(mask)
        observed = np.zeros_like(mask)
        y[:length] = np.stack((truth.speed[absolute], truth.flow_lane[absolute]), -1)
        mask[:length] = targets[absolute]
        eligible[:length] = truth.eligible[absolute]
        observed[:length] = np.isfinite(masked.speed[absolute]) & np.isfinite(masked.flow_lane[absolute])
        slots = pd.to_datetime(truth.timestamps[absolute], utc=True)
        slot_ids = np.asarray(slots.dayofweek * 288 + slots.hour * 12 + slots.minute // 5)
        prior = np.zeros_like(y)
        prior[:length] = context["profile"][slot_ids][:, truth.station_to_link, :2]
        samples.append({"x": x, "context": station, "truth": y, "mask": mask,
                        "eligible": eligible, "observed": observed, "baseline": station[30:42, :, :2],
                        "prior": prior, "date": truth.timestamps[0][:10], "regime": regime,
                        "station_link": truth.station_to_link, "lanes": truth.lanes})
    return samples


@lru_cache(maxsize=32)
def masked_file_index(release, name, split):
    folder = release / "corridors" / name / split / "mainline_states_masked"
    if not folder.is_dir():
        raise FileNotFoundError(f"Released Queue origin needs masked inputs: {folder}")
    result = {}
    for path in sorted(folder.glob("**/*.parquet")):
        match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", path.name)
        if not match:
            raise ValueError(f"Masked file has no date: {path}")
        date = match.group().replace("_", "-")
        if date in result:
            raise ValueError(f"Duplicate masked day: {name} {split} {date}")
        result[date] = path
    return result


def released_origin_features(context, timestamp, ramp, cache):
    """Read exactly T from the released masked layer, never fill it from truth."""
    panel, date = context["panel"], timestamp[:10]
    missing = np.full((1, len(panel.links)), np.nan, np.float32)
    origin = StationDay([timestamp], panel.links, panel.links, np.arange(len(panel.links)), panel.lanes,
                        missing, missing.copy(), np.zeros_like(missing, bool), missing.copy(), np.zeros_like(missing))
    path = context["origin_paths"].get(date)
    if path is not None:
        if date not in cache:
            cache[date] = StationDay.load(path, panel)
        day = cache[date]
        if timestamp in day.timestamps:
            origin = slice_day(day, np.array([day.timestamps.index(timestamp)]))
    return features_v2(origin, panel, context["profile"], ramp)


def queue_examples(context, path, args, period, epoch=0):
    panel = context["panel"]
    day = StationDay.load(path, panel)
    steps, nodes = len(day.timestamps), len(panel.links)
    values, _, _ = aggregate_channels(day, panel, day.eligible)
    speed = values[..., 0]
    minimum = np.full((steps, nodes), np.inf, np.float32)
    valid = np.zeros_like(minimum, bool)
    for station, link in enumerate(day.station_to_link):
        good = day.eligible[:, station]
        minimum[:, link] = np.minimum(minimum[:, link], np.where(good, day.speed[:, station] * 25, np.inf))
        valid[:, link] |= good
    queues = (minimum <= panel.cutoffs) & valid  # Any eligible station, not mean speed.
    candidates = {"onset": [], "ongoing": [], "normal": []}
    for origin in range(12, steps - 6):
        history, future = queues[origin - 12:origin], queues[origin + 1:origin + 7]
        if valid[origin - 12:origin].mean() < args.history_coverage or valid[origin + 1:origin + 7].mean() < args.future_coverage:
            continue
        kind = ("onset" if not history.any() else "ongoing" if history[-1].any() else None) if future.any() else "normal"
        if kind is not None:
            candidates[kind].append(origin)
    seed_parts = (args.seed, panel.name, path.name, "queue")
    if period == "fit" and getattr(args, "queue_resample", False):
        seed_parts += (epoch,)
    rng = np.random.default_rng(stable_seed(*seed_parts))
    condition_gap = getattr(args, "queue_condition_gap", False) if period == "fit" else getattr(args, "queue_validation_spread", False)
    chosen = []
    for kind in ("onset", "ongoing", "normal"):
        limit = args.normal_windows if kind == "normal" else args.windows_per_day
        for origin in rng.permutation(candidates[kind]):
            if sum(k == kind for k, _ in chosen) >= limit:
                break
            if all(abs(int(origin) - earlier) * 5 >= args.window_gap_minutes
                   for other_kind, earlier in chosen if not condition_gap or other_kind == kind):
                chosen.append((kind, int(origin)))
    ramp = ramp_features(context["release"], panel, "train", day.timestamps)
    samples, origin_cache = [], {}
    for kind, origin in chosen:
        index = np.arange(origin - 12, origin)
        future = slice(origin + 1, origin + 7)
        released = (released_origin_features(context, day.timestamps[origin], ramp[origin:origin + 1], origin_cache)
                    if getattr(args, "queue_origin", False) else None)
        x = queue_features(slice_day(day, index), panel, context["profile"], ramp[index], released,
                           getattr(args, "queue_changes", False))
        hard = queues[future].astype(np.float32)
        margin = panel.cutoffs[None] - minimum[future]
        soft = 1 / (1 + np.exp(-np.clip(margin / max(args.label_temperature, 1e-6), -30, 30))) if args.label_temperature > 0 else hard
        weight = np.minimum(np.abs(margin) / args.uncertainty_margin, 1) if args.uncertainty_margin > 0 else np.ones_like(hard)
        weight = np.where(valid[future], np.maximum(weight, 0.1), 0).astype(np.float32)
        samples.append({"x": x, "truth": hard, "soft": soft.astype(np.float32), "speed": speed[future],
                        "mask": valid[future], "weight": weight, "baseline": np.broadcast_to(queues[origin - 1], hard.shape).copy(),
                        "condition": kind, "date": day.timestamps[0][:10], "origin": origin})
    return samples


def tensor_batch(samples, keys, device):
    out = {}
    for key in keys:
        value = torch.from_numpy(np.stack([sample[key] for sample in samples]))
        out[key] = value.pin_memory().to(device, non_blocking=True) if device == "cuda" else value
    return out


def physics_loss(prediction, truth, mask, eligible, observed, station_link, station_lanes, panel):
    """Supervise vehicle counts/differences on completed, observable link fields."""
    complete = torch.where(mask[..., None], prediction, torch.nan_to_num(truth))
    active = eligible & (observed | mask)
    count = torch.zeros((*active.shape[:2], len(panel.links)), device=prediction.device)
    count = count.index_add(2, station_link, active.float())
    def links(field):
        return torch.zeros_like(count).index_add(2, station_link, field * active) / count.clamp_min(1)
    v = links(complete[..., 0] * 25)
    q = links(complete[..., 1] * 600 * station_lanes[None, None])
    vt = links(torch.nan_to_num(truth[..., 0]) * 25)
    qt = links(torch.nan_to_num(truth[..., 1]) * 600 * station_lanes[None, None])
    device = prediction.device
    length = torch.as_tensor(panel.lengths, device=device)
    free = torch.as_tensor(panel.free_speed, device=device)
    capacity = torch.as_tensor(panel.capacity if panel.capacity is not None else np.full(len(panel.links), 6000), device=device)
    jam = torch.as_tensor(panel.jam_density if panel.jam_density is not None else np.full(len(panel.links), 400), device=device)
    scale = (capacity / free.clamp_min(1) * length).clamp_min(1)
    n = q / v.clamp_min(1) * length
    nt = qt / vt.clamp_min(1) * length
    good = count > 0
    loss_n = (((n - nt) / scale).square() * good).sum() / good.sum().clamp_min(1)
    pairs = good[:, 1:] & good[:, :-1]
    loss_dn = ((((n[:, 1:] - n[:, :-1]) - (nt[:, 1:] - nt[:, :-1])) / scale).square() * pairs).sum() / pairs.sum().clamp_min(1)
    density = q / v.clamp_min(1)
    wave = capacity / (jam - capacity / free.clamp_min(1)).clamp_min(1)
    fd_flow = torch.minimum(free * density, wave * (jam - density)).clamp_min(0)
    loss_fd = (((q - fd_flow) / capacity.clamp_min(1)).square() * good).sum() / good.sum().clamp_min(1)
    return loss_n, loss_dn, loss_fd


def state_loss(prediction, batch, mapping, lanes, panel, args, normalizer=None, aux_weight=1.0):
    squared = (prediction.float() - torch.nan_to_num(batch["truth"])).square()
    mask = batch["mask"]
    weights = batch.get("regime_weight")
    denominator = (len(prediction) if weights is not None else mask.sum().clamp_min(1)) if normalizer is None else max(normalizer, 1)
    error = (0.54 * squared[..., 0] + 0.46 * squared[..., 1]) * mask
    loss = (error if weights is None else error * weights[:, None, None]).sum() / denominator
    if args.lambda_n or args.lambda_dn or args.lambda_fd:
        n, dn, fd = physics_loss(prediction.float(), batch["truth"], mask, batch["eligible"], batch["observed"], mapping, lanes, panel)
        loss = loss + aux_weight * (args.lambda_n * n + args.lambda_dn * dn + args.lambda_fd * fd)
    return loss


def queue_loss(logits, speeds, batch, conditions, args, condition_counts=None):
    logits, speeds = logits.float(), speeds.float()
    weights = batch["weight"] * batch["mask"]
    bce = (F.binary_cross_entropy_with_logits(logits, batch["soft"], reduction="none") * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)
    probability = logits.sigmoid() * batch["mask"]
    truth = batch["truth"] * batch["mask"]
    intersection = (probability * truth).sum((1, 2))
    union = (probability + truth - probability * truth).sum((1, 2))
    soft_iou = (1 - (intersection + 1e-6) / (union + 1e-6)) * (truth.sum((1, 2)) > 0)
    speed = (F.smooth_l1_loss(speeds, batch["speed"], reduction="none") * batch["mask"]).sum((1, 2)) / batch["mask"].sum((1, 2)).clamp_min(1)
    losses = bce + args.soft_iou_weight * soft_iou + args.speed_weight * speed
    groups = []
    for kind in ("onset", "ongoing", "normal"):
        indices = [i for i, condition in enumerate(conditions) if condition == kind]
        if indices:
            value = losses[indices].mean() if condition_counts is None else losses[indices].sum() / condition_counts[kind]
            groups.append(value * (args.normal_weight if kind == "normal" else 1.0))
    return torch.stack(groups).mean() if condition_counts is None else torch.stack(groups).sum() / len(condition_counts)


def family_mean(scores):
    families = {}
    for panel, score in scores.items():
        families.setdefault(panel.rsplit("_", 1)[0], []).append(score)
    return float(np.mean([np.mean(values) for values in families.values()])) if families else 0.0


def state_metrics(records, method="model"):
    groups = {}
    for row in records:
        key = row["panel"], row["regime"]
        group = groups.setdefault(key, np.zeros(3, np.float64))
        group += [row[f"{method}_speed_sse"], row[f"{method}_flow_sse"], row["count"]]
    detail = []
    panels = {}
    for (panel, regime), (sv, sq, count) in groups.items():
        rv, rq = np.sqrt(sv / max(count, 1)), np.sqrt(sq / max(count, 1))
        score = float(0.54 * max(0, 1 - rv) + 0.46 * max(0, 1 - rq)) if count else 0.0
        panels.setdefault(panel, {})[regime] = score
        detail.append({"panel": panel, "regime": regime, "count": int(count), "rmse_speed_kmh": float(rv * 25),
                       "rmse_flow_vph_per_lane": float(rq * 600), "State": score})
    # An absent regime has no evidence and receives zero, matching the evaluator.
    scores = {panel: float(np.mean([values.get(r, 0) for r in ("R1", "R2", "R3")])) for panel, values in panels.items()}
    return {"State": family_mean(scores), "panels": scores, "regimes": detail}


def iou(predicted, truth, mask):
    predicted, truth = predicted & mask, truth.astype(bool) & mask
    union = np.logical_or(predicted, truth).sum()
    return float(np.logical_and(predicted, truth).sum() / union) if union else 0.0


def queue_metrics(records, threshold, baseline=False):
    groups = {}
    normal = []
    for row in records:
        score = row["persistence_iou"] if baseline else row["ious"][str(threshold)]
        if row["condition"] == "normal":
            normal.append(row["persistence_false_positive"] if baseline else row["normal_false_positive"][str(threshold)])
        else:
            groups.setdefault((row["panel"], row["condition"]), []).append(score)
    panels = {row["panel"] for row in records}
    by_condition = {kind: {panel: float(np.mean(groups.get((panel, kind), [0]))) for panel in panels} for kind in ("onset", "ongoing")}
    scores = {panel: 0.5 * (by_condition["onset"][panel] + by_condition["ongoing"][panel]) for panel in panels}
    return {"proxy_IoU": family_mean(scores), "onset": family_mean(by_condition["onset"]),
            "ongoing": family_mean(by_condition["ongoing"]), "panels": scores,
            "condition_counts": {kind: sum(row["condition"] == kind for row in records) for kind in ("onset", "ongoing", "normal")},
            "normal_false_positive": float(np.mean(normal)) if normal else None}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def event(path, **values):
    values = {"time_utc": pd.Timestamp.now(tz="UTC").isoformat(), **values}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(values, ensure_ascii=False), flush=True)


def model_from_checkpoint(checkpoint, device):
    expected = queue_feature_names(checkpoint["model_config"].get("queue_changes", False)) if checkpoint["task"] == "queue" else FEATURE_NAMES
    if checkpoint.get("format_version") != 2 or tuple(checkpoint["feature_names"]) != expected:
        raise ValueError("Expected a v2 checkpoint with the current feature contract")
    cls = TrafficStateModel if checkpoint["task"] == "state" else TrafficQueueModel
    if checkpoint["task"] == "state" and checkpoint["model_config"].get("station_decoder", "linear") != "linear":
        if tuple(checkpoint.get("state_context_names", ())) != STATE_CONTEXT_NAMES:
            raise ValueError("State checkpoint has an incompatible station context contract")
    model = cls(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["weights"])
    return model


def check_model_requests(args, config, queue=True):
    for key in ("graph", "missing_aware"):
        if getattr(args, key, False) and not config.get(key, False):
            raise ValueError(f"The checkpoint has {key}=False; use a new checkpoint to change the model")
    if args.graph_mode != "mean" and args.graph_mode != config.get("graph_mode", "mean"):
        raise ValueError("The checkpoint has a different graph mode; use a new checkpoint to change the model")
    for key, default in (("station_decoder", "linear"), ("adaptive_graph_k", 0), ("wavelet_levels", 0)):
        requested = getattr(args, key, default)
        if requested != default and requested != config.get(key, default):
            raise ValueError(f"The checkpoint has a different {key}; use a new checkpoint to change the model")
    if getattr(args, "wavelet_input", None) is not None and args.wavelet_input != config.get("wavelet_input", "embedded"):
        raise ValueError("The checkpoint has a different wavelet_input; use a new checkpoint to change the model")
    if queue:
        for key, default in (("queue_head", "linear"), ("queue_changes", False)):
            requested = getattr(args, key, default)
            if requested != default and requested != config.get(key, default):
                raise ValueError(f"The checkpoint has a different {key}; use a new checkpoint to change the model")


def fit_regime_weights(item):
    """Weight SSE so its window mean equals the fit-only macro regime MSE."""
    targets = item["targets"]
    counts, windows = {}, 0
    fit_paths = set(item["fit_paths"])
    for date, path in file_index(item["release"], item["panel"].name).items():
        if path not in fit_paths:
            continue
        regime = targets.regimes[date]
        cells = targets.archive[date]
        counts[regime] = counts.get(regime, 0) + len(cells)
        windows += len(np.unique((cells // len(targets.stations)) // 12))
    counts = {regime: count for regime, count in counts.items() if count}
    if not counts or set(counts) - {"R1", "R2", "R3"}:
        raise ValueError("Regime balancing requires nonempty R1/R2/R3 fit targets")
    return {regime: windows / (len(counts) * count) for regime, count in counts.items()}


def contexts(args, task, checkpoint=None):
    release = args.release_root.resolve()
    available = sorted(path.name for path in (release / "corridors").iterdir() if path.is_dir())
    names = sorted(set(args.panel or (checkpoint["trained_panels"] if checkpoint else available)))
    if set(names) - set(available):
        raise ValueError(f"Unknown panels: {set(names) - set(available)}")
    if task == "queue":
        names = [name for name in names if name not in {"D12_I405_N", "D12_I405_S"}]
    if not names:
        raise ValueError("No panels for this task")
    result = {}
    for name in names:
        panel = PanelInfo.load(release, name)
        fit_paths = period_files(release, name, "fit", args.max_days)
        if not fit_paths:
            raise ValueError(f"No fit days for {name}")
        if checkpoint:
            if name not in checkpoint["profiles"]:
                raise ValueError(f"Checkpoint has no fit-only profile for {name}")
            profile = checkpoint["profiles"][name].cpu().numpy()
            if checkpoint["links"][name] != panel.links:
                raise ValueError(f"Road layout changed: {name}")
        else:
            profile = fit_profile(release, panel, fit_paths)
        context = {"panel": panel, "release": release, "profile": profile, "fit_paths": fit_paths}
        if task == "queue" and getattr(args, "queue_origin", False):
            context["origin_paths"] = masked_file_index(release, name, "train")
            fit_set = set(fit_paths)
            context["origin_fit_paths"] = [context["origin_paths"][date] for date, path in sorted(file_index(release, name).items())
                                           if path in fit_set and date in context["origin_paths"]]
        if task == "state":
            first = StationDay.load(fit_paths[0], panel)
            context["targets"] = TargetIndex(release, name, first.station_ids)
            context["masked"] = {path.name: path for path in (release / "corridors" / name / "train" / "mainline_states_masked").glob("**/*.parquet")}
            config = checkpoint["model_config"] if checkpoint else vars(args)
            context["station_decoder"] = config.get("station_decoder", "linear")
            if getattr(args, "state_regime_balance", False):
                context["regime_weights"] = fit_regime_weights(context)
        result[name] = context
    return result


def schedule(context, period, args, epoch):
    rng = np.random.default_rng(stable_seed(args.seed, epoch, period))
    names = list(context)
    if period == "fit":
        rng.shuffle(names)
    paths = []
    for name in names:
        days = period_files(args.release_root, name, period, args.max_days)
        if period == "fit":
            rng.shuffle(days)
        elif getattr(args, "queue_validation_spread", False) and days:
            # Visit evenly spaced dates first; run_period takes at most one window per condition/day.
            first = np.linspace(0, len(days) - 1, min(len(days), args.validation_windows)).round().astype(int).tolist()
            chosen = set(first)
            days = [days[i] for i in first] + [path for i, path in enumerate(days) if i not in chosen]
        paths.append([(name, path) for path in days])
    return [item for row in itertools.zip_longest(*paths) for item in row if item is not None]


def prepared_days(context, items, args, task, period, epoch=0):
    def prepare(item):
        name, path = item
        start = time.perf_counter()
        samples = state_examples(context[name], path) if task == "state" else queue_examples(context[name], path, args, period, epoch)
        return name, path, samples, time.perf_counter() - start
    if args.workers == 0:
        for item in items:
            yield prepare(item)
    else:
        # Bounded queue: do not retain the whole dataset in WSL's small RAM.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(items)
            pending = [pool.submit(prepare, item) for item in itertools.islice(iterator, args.workers + 1)]
            while pending:
                future = pending.pop(0)
                result = future.result()
                item = next(iterator, None)
                if item is not None:
                    pending.append(pool.submit(prepare, item))
                yield result


def batch_size(args, name, nodes, task):
    overrides = dict(value.split("=", 1) for value in args.panel_batch)
    if name in overrides:
        size = int(overrides[name])
    elif args.batch_size:
        size = args.batch_size
    else:
        size = (8 if nodes <= 128 else 2) if task == "state" else (48 if nodes <= 128 else 12)
    if size < 1:
        raise ValueError("Batch size must be positive")
    return size


def update_size(args, name, nodes, task):
    requested = getattr(args, "update_windows", None)
    if requested is None:
        return batch_size(args, name, nodes, task)
    if requested < 0:
        raise ValueError("--update-windows must be nonnegative")
    return requested or (8 if task == "state" else 48)


def device_layout(context, vocabulary, device):
    for name, item in context.items():
        panel = item["panel"]
        item["node_ids"] = torch.tensor([vocabulary.get(f"{name}|{link}", 0) for link in panel.links], device=device)
        item["edges"] = torch.as_tensor(panel.edges, device=device)


def run_period(model, context, vocabulary, args, task, period, epoch, device, optimizer=None, log=None, global_step=0):
    training = optimizer is not None
    model.train(training)
    records = []
    pending = {name: [] for name in context}
    totals = torch.zeros(2, device=device)
    physics_totals = torch.zeros(4, device=device)
    seen_conditions = {name: {kind: 0 for kind in ("onset", "ongoing", "normal")} for name in context}
    thresholds = tuple(float(t) for t in (0.3, 0.4, 0.5, 0.6))
    if period == "holdout":
        thresholds = (args.fixed_threshold,)
    started = time.perf_counter()
    preparation_seconds = 0.0
    windows = 0
    updates_by_panel = {name: 0 for name in context}
    device_layout(context, vocabulary, device)

    def process_batch(name, samples, normalizer=None, condition_counts=None, aux_weight=1.0):
        nonlocal windows
        panel = context[name]["panel"]
        if log and windows == 0:
            event(log, event="first_batch", epoch=epoch + 1, period=period, panel=name,
                  batch_size=len(samples), backend=args.model_config["backend"],
                  note="The first CUDA call may compile Triton kernels")
        if task == "state":
            mapping = torch.as_tensor(samples[0]["station_link"], device=device)
            lanes = torch.as_tensor(samples[0]["lanes"], device=device)
            if any(not np.array_equal(sample["station_link"], samples[0]["station_link"]) for sample in samples):
                raise ValueError("Station mapping differs inside a batch")
            batch = tensor_batch(samples, ("x", "context", "truth", "mask", "eligible", "observed"), device)
            if training and getattr(args, "state_regime_balance", False):
                batch["regime_weight"] = torch.tensor([context[name]["regime_weights"][s["regime"]]
                                                      for s in samples], device=device)
        else:
            batch = tensor_batch(samples, ("x", "truth", "soft", "speed", "mask", "weight"), device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device, dtype=torch.bfloat16, enabled=args.use_bf16):
                if task == "state":
                    prediction = model(batch["x"], batch["context"], mapping, context[name]["node_ids"], panel.segments, context[name]["edges"], args.bidirectional)[:, 30:42]
                else:
                    logits, speeds = model(batch["x"], context[name]["node_ids"], panel.segments, context[name]["edges"], args.bidirectional)
            loss = (state_loss(prediction, batch, mapping, lanes, panel, args, normalizer, aux_weight)
                    if task == "state" else queue_loss(logits, speeds, batch, [s["condition"] for s in samples], args, condition_counts))
            if training:
                loss.backward()
        windows += len(samples)
        if not training:
            if task == "state":
                physics = physics_loss(prediction.float(), batch["truth"], batch["mask"], batch["eligible"], batch["observed"], mapping, lanes, panel)
                physics_totals[:3] += torch.stack(physics).detach() * len(samples)
                physics_totals[3] += len(samples)
                outputs = prediction.float().cpu().numpy()
                for sample, output in zip(samples, outputs):
                    row = {"panel": name, "date": sample["date"], "regime": sample["regime"], "count": int(sample["mask"].sum())}
                    for method, estimate in (("model", output), ("interpolation", sample["baseline"]), ("historical", sample["prior"])):
                        squared = (estimate - np.nan_to_num(sample["truth"])) ** 2
                        row[f"{method}_speed_sse"] = float(squared[..., 0][sample["mask"]].sum())
                        row[f"{method}_flow_sse"] = float(squared[..., 1][sample["mask"]].sum())
                    records.append(row)
            else:
                probabilities, future_speed = logits.float().sigmoid().cpu().numpy(), speeds.float().cpu().numpy()
                for sample, probability, predicted_speed in zip(samples, probabilities, future_speed):
                    record = {"panel": name, "date": sample["date"], "condition": sample["condition"], "origin": sample["origin"],
                              "persistence_iou": iou(sample["baseline"], sample["truth"], sample["mask"]),
                              "persistence_false_positive": float((sample["baseline"] & sample["mask"]).sum() / max(sample["mask"].sum(), 1)),
                              "ious": {str(t): iou(probability >= t, sample["truth"], sample["mask"]) for t in thresholds},
                              "normal_false_positive": {str(t): float(((probability >= t) & sample["mask"]).sum() / max(sample["mask"].sum(), 1)) for t in thresholds},
                              "speed_sse_kmh": float((((predicted_speed - sample["speed"]) * 25) ** 2)[sample["mask"]].sum()),
                              "count": int(sample["mask"].sum())}
                    records.append(record)
        return loss.detach()

    def process(name, samples):
        nonlocal global_step
        count = len(samples)
        size = batch_size(args, name, len(context[name]["panel"].links), task) if training else count
        accumulated = training and getattr(args, "update_windows", None) is not None
        normalizer = ((count if getattr(args, "state_regime_balance", False) else
                       sum(np.count_nonzero(sample["mask"]) for sample in samples))
                      if accumulated and task == "state" else None)
        conditions = {kind: sum(sample["condition"] == kind for sample in samples)
                      for kind in {sample["condition"] for sample in samples}} if accumulated and task == "queue" else None
        if training:
            optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for start in range(0, count, size):
            part = samples[start:start + size]
            total_loss += process_batch(name, part, normalizer, conditions, len(part) / count)
        if training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            updates_by_panel[name] += 1
        totals[0] += total_loss * count
        totals[1] += count
        if log and training and global_step % args.log_every == 0:
            values = totals.cpu().tolist()
            event(log, event="progress", epoch=epoch + 1, period=period, panel=name, step=global_step,
                  windows=windows, mean_loss=values[0] / max(values[1], 1), elapsed_seconds=time.perf_counter() - started)

    items = schedule(context, period, args, epoch)
    if not items:
        raise ValueError(f"No days in {period}")
    for number, (name, path, samples, prep) in enumerate(prepared_days(context, items, args, task, period, epoch), 1):
        preparation_seconds += prep
        if not training and task == "queue":
            selected = []
            daily_seen = set()
            for sample in samples:
                kind = sample["condition"]
                if (seen_conditions[name][kind] < args.validation_windows
                        and (not getattr(args, "queue_validation_spread", False) or kind not in daily_seen)):
                    selected.append(sample)
                    daily_seen.add(kind)
                    seen_conditions[name][kind] += 1
            samples = selected
        pending[name].extend(samples)
        size = (update_size if training else batch_size)(args, name, len(context[name]["panel"].links), task)
        while len(pending[name]) >= size:
            process(name, pending[name][:size])
            del pending[name][:size]
        if log and (number == 1 or number % 50 == 0 or number == len(items)):
            event(log, event="data_progress", epoch=epoch + 1, period=period, days=number, total_days=len(items), panel=name, windows=windows)
    for name, samples in pending.items():
        if samples:
            process(name, samples)
    values = totals.cpu().tolist()
    if not values[1] or not np.isfinite(values).all():
        raise RuntimeError(f"No usable windows or non-finite loss in {period}")
    elapsed = time.perf_counter() - started
    report = {"task": task, "period": period, "loss": values[0] / values[1], "windows": windows,
              "elapsed_seconds": elapsed, "preparation_seconds_sum": preparation_seconds,
              "windows_per_second": windows / elapsed, "daily_records": records,
              "optimizer_updates": sum(updates_by_panel.values()), "updates_by_panel": updates_by_panel}
    if not training:
        if task == "state":
            report.update(state_metrics(records))
            report["baselines"] = {method: state_metrics(records, method) for method in ("interpolation", "historical")}
            physical = physics_totals.cpu().tolist()
            report["physics_aux_normalized_mse"] = dict(zip(("N", "dN", "FD"), [v / max(physical[3], 1) for v in physical[:3]]))
        else:
            chosen = max(thresholds, key=lambda t: queue_metrics(records, t)["proxy_IoU"])
            report.update(queue_metrics(records, chosen))
            report["threshold"] = chosen
            report["baseline"] = queue_metrics(records, chosen, True)
            report["future_speed_rmse_kmh"] = float(np.sqrt(sum(row["speed_sse_kmh"] for row in records) / max(sum(row["count"] for row in records), 1)))
            keys = sorted((row["panel"], row["date"], row["condition"], row["origin"]) for row in records)
            report["window_signature"] = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
            report["validation_spread"] = getattr(args, "queue_validation_spread", False)
    return report, global_step


def save_checkpoint(path, model, optimizer, scheduler, context, vocabulary, args, task, epoch, best, bad_epochs, step, report):
    payload = {"format_version": 2, "task": task, "weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "model_config": args.model_config,
               "feature_names": list(queue_feature_names(args.model_config.get("queue_changes", False)) if task == "queue" else FEATURE_NAMES),
               "node_vocab": vocabulary,
               "trained_panels": list(context), "links": {name: item["panel"].links for name, item in context.items()},
               "profiles": {name: torch.from_numpy(item["profile"]) for name, item in context.items()},
               "fit_signature": {name: signature(item["fit_paths"]) for name, item in context.items()},
               "training_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "epoch": epoch + 1, "global_step": step, "best_metric": best, "bad_epochs": bad_epochs,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "threshold": report.get("threshold", 0.5), "bidirectional": args.bidirectional,
               "metrics": {key: value for key, value in report.items() if key != "daily_records"},
               "environment": {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda}}
    if task == "state":
        payload["state_context_names"] = list(STATE_CONTEXT_NAMES if model.station_decoder != "linear" else STATE_CONTEXT_NAMES[:3])
    elif getattr(args, "queue_origin", False):
        payload["origin_fit_signature"] = {name: signature(item["origin_fit_paths"]) for name, item in context.items()}
    temporary = path.with_suffix(path.suffix + ".pending")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args, task):
    if min(args.epochs, args.cpu_threads, args.log_every) < 1 or args.grad_clip <= 0:
        raise ValueError("Epochs, threads, log interval, and gradient clipping must be positive")
    if min(args.lambda_n, args.lambda_dn, args.lambda_fd, args.normal_weight, args.soft_iou_weight, args.speed_weight, args.patience) < 0:
        raise ValueError("Loss weights and early-stopping patience must be nonnegative")
    if not 0 <= args.history_coverage <= 1 or not 0 <= args.future_coverage <= 1:
        raise ValueError("Coverage thresholds must lie in [0, 1]")
    if args.update_windows is not None and args.update_windows < 0:
        raise ValueError("--update-windows must be nonnegative")
    if args.graph_mode != "mean" and not args.graph and not (args.resume or args.init_state):
        raise ValueError("--graph-mode reliability requires --graph")
    if args.adaptive_graph_k < 0 or args.adaptive_graph_k == 1 or not 0 <= args.wavelet_levels <= 3:
        raise ValueError("--adaptive-graph-k must be 0 or at least 2; --wavelet-levels must lie in [0, 3]")
    if min(args.validation_windows, args.windows_per_day) < 1 or args.normal_windows < 0 or args.window_gap_minutes < 0:
        raise ValueError("Queue window limits must be positive, and normal windows/gap nonnegative")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.use_bf16 = device == "cuda" and (args.bf16 or args.precision == "bf16" or args.precision == "auto" and torch.cuda.is_bf16_supported())
    if (args.bf16 or args.precision == "bf16") and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training requires a supported CUDA GPU")
    if args.backend != "scan" and device != "cuda":
        raise ValueError("Official Mamba training requires CUDA; use --backend scan only for CPU contract tests")
    resumed = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if resumed and resumed["task"] != task:
        raise ValueError("Resume checkpoint task mismatch")
    if resumed:
        check_model_requests(args, resumed["model_config"])
        for key in ("queue_origin", "queue_resample", "queue_condition_gap", "queue_validation_spread"):
            saved = resumed["training_args"].get(key, False)
            if getattr(args, key, False) and not saved:
                raise ValueError(f"Resume must retain {key}; use a new checkpoint to change Queue inputs/sampling")
            setattr(args, key, saved)
        args.queue_head = resumed["model_config"].get("queue_head", "linear")
        args.queue_changes = resumed["model_config"].get("queue_changes", False)
        if args.state_regime_balance and not resumed["training_args"].get("state_regime_balance", False):
            raise ValueError("Resume must retain the loss; use a new checkpoint for regime balancing")
        # Restore settings that determine the data/loss; only the requested epoch budget and paths may change.
        for key in ("seed", "bidirectional", "windows_per_day", "normal_windows", "window_gap_minutes", "validation_windows",
                    "history_coverage", "future_coverage", "label_temperature", "uncertainty_margin", "normal_weight",
                    "soft_iou_weight", "speed_weight", "lambda_n", "lambda_dn", "lambda_fd", "max_days", "precision", "bf16",
                    "batch_size", "panel_batch", "grad_clip"):
            setattr(args, key, resumed["training_args"][key])
        args.use_bf16 = resumed["training_args"]["use_bf16"]
        # Older v2 checkpoints keep their original per-microbatch updates.
        args.update_windows = resumed["training_args"].get("update_windows")
        args.state_regime_balance = resumed["training_args"].get("state_regime_balance", False)
    context = contexts(args, task, resumed)
    if resumed:
        if set(context) != set(resumed["trained_panels"]):
            raise ValueError("Resume must retain the trained panel set")
        for name, item in context.items():
            if signature(item["fit_paths"]) != resumed["fit_signature"][name]:
                raise ValueError(f"Fit dataset changed since the checkpoint: {name}")
            if args.queue_origin and signature(item["origin_fit_paths"]) != resumed["origin_fit_signature"][name]:
                raise ValueError(f"Released Queue origin dataset changed since the checkpoint: {name}")
        vocabulary = resumed["node_vocab"]
        args.model_config = resumed["model_config"]
        model = model_from_checkpoint(resumed, device)
    else:
        vocabulary = {f"{name}|{link}": i + 1 for i, (name, link) in enumerate((name, link) for name in context for link in context[name]["panel"].links)}
        args.model_config = {"features": len(FEATURE_NAMES), "node_count": len(vocabulary), "width": args.width,
                             "blocks": args.blocks, "heads": args.heads, "dropout": args.dropout,
                             "backend": args.backend, "graph": args.graph,
                             "missing_aware": args.missing_aware, "graph_mode": args.graph_mode,
                             "station_decoder": args.station_decoder, "adaptive_graph_k": args.adaptive_graph_k,
                             "wavelet_levels": args.wavelet_levels, "wavelet_input": args.wavelet_input or "signals"}
        queue_config = {"queue_head": args.queue_head, "queue_changes": args.queue_changes,
                        "features": len(queue_feature_names(args.queue_changes))} if task == "queue" else {}
        args.model_config.update(queue_config)
        transfer = torch.load(args.init_state, map_location="cpu", weights_only=True) if task == "queue" and args.init_state else None
        if transfer:
            if transfer.get("format_version") != 2 or transfer["task"] != "state":
                raise ValueError("Transfer needs a v2 State checkpoint")
            check_model_requests(args, transfer["model_config"], queue=False)
            if any(name not in transfer["profiles"] for name in context):
                raise ValueError("The State checkpoint must cover the Queue panels")
            # Transfer fit-only statistics and identity mapping with the encoder.
            vocabulary = transfer["node_vocab"]
            args.model_config = {**transfer["model_config"], **queue_config}
            for name, item in context.items():
                if transfer["links"][name] != item["panel"].links:
                    raise ValueError("Transfer road layout mismatch")
                if transfer["fit_signature"][name] != signature(item["fit_paths"]):
                    raise ValueError("Transfer must use the same fit dates and data")
                item["profile"] = transfer["profiles"][name].numpy()
        model = (TrafficStateModel if task == "state" else TrafficQueueModel)(**args.model_config).to(device)
        if transfer:
            weights = {key.removeprefix("encoder."): value for key, value in transfer["weights"].items() if key.startswith("encoder.")}
            if args.queue_changes:
                expanded = torch.zeros_like(model.encoder.input.weight, device="cpu")
                expanded[:, :len(FEATURE_NAMES)] = weights["input.weight"]
                weights["input.weight"] = expanded
            model.encoder.load_state_dict(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=device == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=args.scheduler_patience, factor=0.5)
    start_epoch, best, bad_epochs, step = 0, -float("inf"), 0, 0
    if resumed:
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        start_epoch, best, bad_epochs, step = resumed["epoch"], resumed["best_metric"], resumed["bad_epochs"], resumed["global_step"]
        torch.set_rng_state(resumed["rng_cpu"])
        if device == "cuda" and resumed["rng_cuda"]:
            torch.cuda.set_rng_state_all(resumed["rng_cuda"])
    if args.epochs <= start_epoch:
        raise ValueError(f"--epochs is the total budget and must exceed completed epoch {start_epoch}")
    log = args.checkpoint.with_suffix(".jsonl")
    event(log, event="start", task=task, device=device, model=args.model_config, bf16=args.use_bf16,
          panels=list(context), epochs=args.epochs, start_epoch=start_epoch, parameters=sum(p.numel() for p in model.parameters()),
          update_windows=args.update_windows,
          state_regime_weights={name: item.get("regime_weights", {}) for name, item in context.items()},
          effective_update_windows={name: update_size(args, name, len(item["panel"].links), task) for name, item in context.items()})
    for epoch in range(start_epoch, args.epochs):
        fit, step = run_period(model, context, vocabulary, args, task, "fit", epoch, device, optimizer, log, step)
        select, _ = run_period(model, context, vocabulary, args, task, "select", epoch, device, log=log, global_step=step)
        metric = select["State"] if task == "state" else select["proxy_IoU"]
        improved = metric > best
        best = max(best, metric)
        bad_epochs = 0 if improved else bad_epochs + 1
        scheduler.step(metric)
        event(log, event="epoch", epoch=epoch + 1, fit_loss=fit["loss"], select_metric=metric, best_metric=best,
              learning_rate=optimizer.param_groups[0]["lr"], fit_seconds=fit["elapsed_seconds"],
              fit_windows_per_second=fit["windows_per_second"], improved=improved,
              fit_windows=fit["windows"], optimizer_updates=fit["optimizer_updates"], updates_by_panel=fit["updates_by_panel"])
        write_json(args.checkpoint.with_suffix(f".epoch{epoch + 1}.json"), select)
        save_checkpoint(args.checkpoint.with_name(args.checkpoint.stem + ".last.pt"), model, optimizer, scheduler, context, vocabulary, args, task, epoch, best, bad_epochs, step, select)
        if improved:
            save_checkpoint(args.checkpoint, model, optimizer, scheduler, context, vocabulary, args, task, epoch, best, bad_epochs, step, select)
        if args.patience and bad_epochs >= args.patience:
            event(log, event="early_stop", epoch=epoch + 1, patience=args.patience)
            break
    print(f"Best checkpoint: {args.checkpoint}; latest/resume: {args.checkpoint.stem}.last.pt", flush=True)


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("evaluate needs a v2 checkpoint; retain the previous audit for legacy scores")
    for key, value in checkpoint["training_args"].items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    if args.validation_windows is None:
        args.validation_windows = checkpoint["training_args"]["validation_windows"]
    if args.validation_windows < 1:
        raise ValueError("--validation-windows must be positive")
    args.queue_validation_spread = bool(getattr(args, "queue_validation_spread", False))
    args.seed = checkpoint["training_args"]["seed"]
    args.fixed_threshold = checkpoint["threshold"]
    args.bidirectional = checkpoint["bidirectional"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.cpu_threads)
    args.use_bf16 = device == "cuda" and checkpoint["training_args"]["use_bf16"]
    context = contexts(args, checkpoint["task"], checkpoint)
    model = model_from_checkpoint(checkpoint, device).eval()
    report, _ = run_period(model, context, checkpoint["node_vocab"], args, checkpoint["task"], args.period, 0, device)
    if args.period == "select" and checkpoint["task"] == "queue":
        # Evaluation of a saved checkpoint uses its selected threshold.
        report.update(queue_metrics(report["daily_records"], checkpoint["threshold"]))
        report["threshold"] = checkpoint["threshold"]
    if args.bootstrap:
        rows = report["daily_records"]
        dates = sorted({row["date"] for row in rows})
        by_date = {date: [row for row in rows if row["date"] == date] for date in dates}
        rng = np.random.default_rng(args.seed)
        differences = []
        for _ in range(args.bootstrap):
            sample = [row for date in rng.choice(dates, len(dates), replace=True) for row in by_date[date]]
            if checkpoint["task"] == "state":
                differences.append(state_metrics(sample)["State"] - state_metrics(sample, "interpolation")["State"])
            else:
                t = checkpoint["threshold"]
                differences.append(queue_metrics(sample, t)["proxy_IoU"] - queue_metrics(sample, t, True)["proxy_IoU"])
        report["paired_day_bootstrap_delta_95pct"] = np.percentile(differences, [2.5, 97.5]).tolist()
    report["checkpoint_epoch"] = checkpoint["epoch"]
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "daily_records"}, ensure_ascii=False, indent=2))


def inference_context(checkpoint, release, name, device):
    panel = PanelInfo.load(release, name)
    if name not in checkpoint["profiles"] or checkpoint["links"][name] != panel.links:
        raise ValueError(f"Checkpoint does not cover this panel/layout: {name}")
    return panel, checkpoint["profiles"][name].numpy(), torch.tensor(
        [checkpoint["node_vocab"].get(f"{name}|{link}", 0) for link in panel.links], device=device), torch.as_tensor(panel.edges, device=device)


def predict_state_v2(model, checkpoint, release, split, output, selected=None):
    device = next(model.parameters()).device.type
    parts = []
    for name in selected or checkpoint["trained_panels"]:
        panel, profile, node_ids, edges = inference_context(checkpoint, release, name, device)
        template = pd.read_csv(release / "task1" / name / split / "sample_submission_state.csv", dtype=str)
        keys = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
        if template[keys].duplicated().any():
            raise ValueError(f"Duplicate State template keys: {name}")
        by_date = {date: frame for date, frame in template.groupby(template.timestamp.str[:10])}
        root = release / "corridors" / name / split / "mainline_states_masked"
        for path in sorted(root.glob("**/*.parquet")):
            day = StationDay.load(path, panel)
            date = day.timestamps[0][:10]
            target = by_date.pop(date, None)
            if target is None:
                continue
            if len(target.mask_regime.unique()) != 1 or path.parent.name != f"mask_regime={target.mask_regime.iloc[0]}":
                raise ValueError(f"State regime mismatch: {path}")
            ramp = ramp_features(release, panel, split, day.timestamps)
            windows = list(center_windows(len(day.timestamps)))
            prediction = np.zeros((*day.speed.shape, 2), np.float32)
            size = 8 if len(panel.links) <= 128 else 2
            mapping = torch.as_tensor(day.station_to_link, device=device)
            for start in range(0, len(windows), size):
                chunk = windows[start:start + size]
                xs, cs = zip(*(state_features_v2(day, panel, index, profile, ramp,
                              rich=checkpoint.get("model_config", {}).get("station_decoder", "linear") != "linear") for index, _ in chunk))
                batch = tensor_batch([{"x": x, "context": c} for x, c in zip(xs, cs)], ("x", "context"), device)
                with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                    enabled=device == "cuda" and checkpoint["training_args"]["use_bf16"]):
                    result = model(batch["x"], batch["context"], mapping, node_ids, panel.segments, edges, checkpoint["bidirectional"])[:, 30:42].float().cpu().numpy()
                for values, (_, absolute) in zip(result, chunk):
                    prediction[absolute] = values[:absolute.stop - absolute.start]
            if not np.isfinite(prediction).all():
                raise RuntimeError("Non-finite State prediction")
            times = {time: i for i, time in enumerate(day.timestamps)}
            stations = {station: i for i, station in enumerate(day.station_ids)}
            ti, si = target.timestamp.map(times), target.station_id.map(stations)
            if ti.isna().any() or si.isna().any():
                raise ValueError("Template key absent from State input")
            ti, si = ti.to_numpy(int), si.to_numpy(int)
            if not np.array_equal(np.array(day.station_links)[si], target.link_id.to_numpy()):
                raise ValueError("Template station/link mapping mismatch")
            result = target.copy()
            result["speed_kmh"] = prediction[ti, si, 0] * 25
            result["flow_vph"] = prediction[ti, si, 1] * 600 * day.lanes[si]
            parts.append(result[keys + ["speed_kmh", "flow_vph"]])
        if by_date:
            raise ValueError(f"Missing masked days: {sorted(by_date)}")
    if not parts:
        raise ValueError("No State predictions")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(output, index=False)


def history_day(frame, panel):
    times = sorted(frame.timestamp.unique().tolist())
    if len(times) != 12 or frame.duplicated(["timestamp", "link_id"]).any():
        raise ValueError("Task 2 history must have twelve unique time/link cells")
    if not set(frame.link_id).issubset(panel.link_index):
        raise ValueError("Task 2 history has unknown links")
    index = pd.MultiIndex.from_product([times, panel.links], names=["timestamp", "link_id"])
    aligned = frame.set_index(["timestamp", "link_id"]).reindex(index)
    shape = len(times), len(panel.links)
    speed = pd.to_numeric(aligned.speed_kmh, errors="coerce").to_numpy(np.float32).reshape(shape) / 25
    flow = pd.to_numeric(aligned.flow_vph, errors="coerce").to_numpy(np.float32).reshape(shape) / panel.lanes / 600
    occupancy = pd.to_numeric(aligned.occupancy, errors="coerce").to_numpy(np.float32).reshape(shape)
    quality = pd.to_numeric(aligned.pct_observed, errors="coerce").fillna(0).to_numpy(np.float32).reshape(shape) / 100
    eligible = aligned.is_score_eligible.fillna(False).astype(bool).to_numpy().reshape(shape)
    return StationDay(times, panel.links, panel.links, np.arange(len(panel.links)), panel.lanes,
                      speed, flow, eligible & np.isfinite(speed) & np.isfinite(flow), occupancy, quality)


def predict_queue_v2(model, checkpoint, release, split, output, selected=None, probabilities=False):
    device = next(model.parameters()).device.type
    parts = []
    for name in selected or checkpoint["trained_panels"]:
        root = release / "task2" / name / split
        if not (root / "sample_submission_queue.csv").exists():
            continue
        panel, profile, node_ids, edges = inference_context(checkpoint, release, name, device)
        include_origin = checkpoint["training_args"].get("queue_origin", False)
        changes = checkpoint.get("model_config", {}).get("queue_changes", False)
        origin_context = {"panel": panel, "profile": profile}
        if include_origin:
            origin_context["origin_paths"] = masked_file_index(release, name, split)
        origin_cache = {}
        template = pd.read_csv(root / "sample_submission_queue.csv", dtype=str)
        index = pd.read_csv(root / "window_index.csv", dtype=str).set_index("window_id")
        history = pd.read_parquet(root / "window_history.parquet")
        for col in ("window_id", "timestamp", "link_id"):
            history[col] = history[col].astype(str)
        histories = {key: frame for key, frame in history.groupby("window_id")}
        inputs, targets = [], []
        for window_id, target in template.groupby("window_id", sort=False):
            if window_id not in histories or window_id not in index.index:
                raise ValueError(f"No history/index for {window_id}")
            day = history_day(histories[window_id], panel)
            origin = pd.Timestamp(index.loc[window_id, "forecast_origin"])
            expected = pd.date_range(origin - pd.Timedelta(minutes=60), periods=12, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            future = pd.date_range(origin + pd.Timedelta(minutes=5), periods=6, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            if day.timestamps != expected or sorted(target.timestamp.unique()) != future:
                raise ValueError(f"Unexpected Task 2 time boundary: {window_id}")
            if not set(target.link_id).issubset(panel.link_index):
                raise ValueError("Unknown Queue target link")
            timestamp = origin.strftime("%Y-%m-%dT%H:%M:%SZ")
            ramp_times = day.timestamps + ([timestamp] if include_origin else [])
            ramp = ramp_features(release, panel, split, ramp_times)
            released = released_origin_features(origin_context, timestamp, ramp[-1:], origin_cache) if include_origin else None
            inputs.append({"x": queue_features(day, panel, profile, ramp[:12], released, changes)})
            targets.append((target, future))
        size = 48 if len(panel.links) <= 128 else 12
        for start in range(0, len(inputs), size):
            batch = tensor_batch(inputs[start:start + size], ("x",), device)
            with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16,
                enabled=device == "cuda" and checkpoint["training_args"]["use_bf16"]):
                logits, _ = model(batch["x"], node_ids, panel.segments, edges, checkpoint["bidirectional"])
                values = logits.float().sigmoid().cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("Non-finite Queue prediction")
            for probability, (target, future) in zip(values, targets[start:start + size]):
                result = target.copy()
                ti = result.timestamp.map({time: i for i, time in enumerate(future)}).to_numpy(int)
                li = result.link_id.map(panel.link_index).to_numpy(int)
                result["queue_pred"] = probability[ti, li] if probabilities else (probability[ti, li] >= checkpoint["threshold"]).astype(int)
                parts.append(result[["window_id", "timestamp", "link_id", "queue_pred"]])
    if not parts:
        raise ValueError("No Queue predictions")
    result = pd.concat(parts, ignore_index=True)
    if result[["window_id", "timestamp", "link_id"]].duplicated().any():
        raise ValueError("Duplicate Queue keys")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
