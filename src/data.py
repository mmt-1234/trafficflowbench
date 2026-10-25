"""Released panel data to tensors, with masks applied before aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import heapq

import numpy as np
import pandas as pd


def scan_layout(link_ids: list[str], topology: pd.DataFrame) -> tuple[list[str], tuple[tuple[int, int], ...]]:
    """Topological order within each weakly connected component; cycles are stable."""
    links = set(link_ids)
    outgoing = {link: set() for link in links}
    neighbors = {link: set() for link in links}
    for row in topology.itertuples(index=False):
        target = str(row.link_id)
        if target not in links:
            continue
        for source in str(getattr(row, "incoming_link_ids", "")).split(";"):
            source = source.strip()
            if source in links and source != target:
                outgoing[source].add(target)
                neighbors[source].add(target)
                neighbors[target].add(source)
        for dest in str(getattr(row, "outgoing_link_ids", "")).split(";"):
            dest = dest.strip()
            if dest in links and dest != target:
                outgoing[target].add(dest)
                neighbors[target].add(dest)
                neighbors[dest].add(target)
    seen: set[str] = set()
    ordered: list[str] = []
    segments: list[tuple[int, int]] = []
    for root in sorted(links):
        if root in seen:
            continue
        component = {root}
        pending = [root]
        seen.add(root)
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    pending.append(neighbor)
        indegree = {link: 0 for link in component}
        for source in component:
            for target in outgoing[source] & component:
                indegree[target] += 1
        ready = [link for link in component if indegree[link] == 0]
        heapq.heapify(ready)
        part = []
        while ready:
            source = heapq.heappop(ready)
            part.append(source)
            for target in outgoing[source] & component:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)
        part.extend(sorted(component - set(part)))
        start = len(ordered)
        ordered.extend(part)
        segments.append((start, len(ordered)))
    return ordered, tuple(segments)


@dataclass
class PanelInfo:
    name: str
    links: list[str]
    segments: tuple[tuple[int, int], ...]
    link_index: dict[str, int]
    free_speed: np.ndarray
    lanes: np.ndarray
    lengths: np.ndarray
    cutoffs: np.ndarray
    station_lanes: dict[str, float]
    capacity: np.ndarray | None = None
    jam_density: np.ndarray | None = None
    position: np.ndarray | None = None
    edges: np.ndarray = field(default_factory=lambda: np.empty((2, 0), dtype=np.int64))
    ramp_map: dict[str, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, release: Path, panel: str) -> "PanelInfo":
        panel_dir = release / "corridors" / panel
        candidates = sorted((panel_dir / "train" / "mainline_states").glob("**/*.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No original train observations under {panel_dir}")
        observed = pd.read_parquet(candidates[0], columns=["link_id"]).link_id.astype(str).unique().tolist()
        network = panel_dir / "network"
        topology = pd.read_csv(network / "lwr_mainline_topology.csv", dtype=str)
        # Latent mainline connectors carry no detector value, but preserve the
        # road between observed links during the spatial scan.
        links, segments = scan_layout(sorted(set(observed) | set(topology.link_id.dropna().astype(str))), topology)
        frame = pd.read_csv(network / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
        fd_path = network / "fd_parameters.csv"
        fd = pd.read_csv(fd_path, dtype={"link_id": str, "station_id": str}) if fd_path.exists() else pd.DataFrame()
        grouped = fd.groupby("link_id").mean(numeric_only=True) if not fd.empty else pd.DataFrame()

        def parameter(column: str, default: float) -> np.ndarray:
            raw = pd.to_numeric(frame[column], errors="coerce").reindex(links) if column in frame else pd.Series(default, index=links)
            if column in grouped:
                raw = pd.to_numeric(grouped[column], errors="coerce").reindex(links).fillna(raw)
            return raw.fillna(default).to_numpy(dtype=np.float32)

        free_speed = parameter("free_speed_kmh", 105.0)
        lanes = np.clip(parameter("lanes", 1.0), 1, 16)
        lengths = parameter("length_km", 1.0)
        cutoffs = 0.6 * free_speed
        if "v_cut" in grouped:
            cutoffs = pd.to_numeric(grouped.v_cut, errors="coerce").reindex(links).fillna(pd.Series(cutoffs, index=links)).to_numpy(dtype=np.float32)
        station_lanes: dict[str, float] = {}
        if "station_id" in fd and "lanes" in fd:
            for row in fd.dropna(subset=["station_id", "lanes"]).itertuples(index=False):
                station_lanes[str(row.station_id)] = float(row.lanes)
        lookup = {link: i for i, link in enumerate(links)}
        pairs = set()
        for row in topology.itertuples(index=False):
            target = str(row.link_id)
            for source in str(getattr(row, "incoming_link_ids", "")).split(";"):
                if source in lookup and target in lookup and source != target:
                    pairs.add((lookup[source], lookup[target]))
            for dest in str(getattr(row, "outgoing_link_ids", "")).split(";"):
                if target in lookup and dest in lookup and dest != target:
                    pairs.add((lookup[target], lookup[dest]))
        positions = topology.drop_duplicates("link_id").set_index("link_id")
        position = pd.to_numeric(positions.get("order_index", pd.Series(dtype=float)), errors="coerce").reindex(links)
        position = position.fillna(pd.Series(np.arange(len(links)), index=links)).to_numpy(np.float32)
        position = (position - position.min()) / max(float(np.ptp(position)), 1.0)
        ramp_map = {}
        ramp_path = network / "ramp_attachment_map.csv"
        if ramp_path.exists():
            for row in pd.read_csv(ramp_path, dtype=str).itertuples(index=False):
                if row.nearest_mainline_link_id in lookup:
                    if row.ramp_type not in {"OR", "FR", "on", "off"}:
                        raise ValueError(f"Unknown ramp type: {row.ramp_type}")
                    ramp_map[row.ramp_link_id] = (lookup[row.nearest_mainline_link_id], int(row.ramp_type in {"FR", "off"}))
        return cls(panel, links, segments, lookup, free_speed, lanes, lengths, cutoffs, station_lanes,
                   parameter("capacity_vph", 6000.0), parameter("k_jam", 400.0), position,
                   np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2).T, ramp_map)


@dataclass
class StationDay:
    timestamps: list[str]
    station_ids: list[str]
    station_links: list[str]
    station_to_link: np.ndarray
    lanes: np.ndarray
    speed: np.ndarray
    flow_lane: np.ndarray
    eligible: np.ndarray
    occupancy: np.ndarray | None = None
    quality: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path, panel: PanelInfo) -> "StationDay":
        columns = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]
        # Optional published features also exist in Task 2 histories.
        import pyarrow.parquet as pq
        available = set(pq.read_schema(path).names) if path.stat().st_size else set(columns)
        columns += [c for c in ("occupancy", "pct_observed") if c in available]
        frame = pd.read_parquet(path, columns=columns)
        for column in ("timestamp", "station_id", "link_id"):
            frame[column] = frame[column].astype(str)
        if frame.duplicated(["timestamp", "station_id", "link_id"]).any():
            raise ValueError(f"Duplicate station/time rows in {path}")
        stations = frame[["station_id", "link_id"]].drop_duplicates()
        if stations.station_id.duplicated().any():
            raise ValueError(f"Station maps to multiple links in {path}")
        stations = stations.sort_values(["link_id", "station_id"])
        station_ids = stations.station_id.tolist()
        station_links = stations.link_id.tolist()
        unknown = set(station_links) - set(panel.link_index)
        if unknown:
            raise ValueError(f"Links absent from panel layout: {sorted(unknown)[:4]}")
        timestamps = sorted(frame.timestamp.unique().tolist())
        index = pd.MultiIndex.from_product([timestamps, station_ids], names=["timestamp", "station_id"])
        aligned = frame.set_index(["timestamp", "station_id"]).reindex(index)
        speed = pd.to_numeric(aligned.speed_kmh, errors="coerce").to_numpy(dtype=np.float32).reshape(len(timestamps), -1)
        flow = pd.to_numeric(aligned.flow_vph, errors="coerce").to_numpy(dtype=np.float32).reshape(len(timestamps), -1)
        lanes = np.array([panel.station_lanes.get(s, panel.lanes[panel.link_index[l]]) for s, l in zip(station_ids, station_links)], dtype=np.float32)
        lanes = np.clip(lanes, 1, 16)
        eligible = aligned.is_score_eligible.fillna(False).astype(bool).to_numpy(copy=True).reshape(len(timestamps), -1)
        eligible &= np.isfinite(speed) & np.isfinite(flow)
        occupancy = pd.to_numeric(aligned.occupancy, errors="coerce").to_numpy(np.float32).reshape(speed.shape) if "occupancy" in aligned else np.full_like(speed, np.nan)
        quality = pd.to_numeric(aligned.pct_observed, errors="coerce").fillna(0).to_numpy(np.float32).reshape(speed.shape) / 100 if "pct_observed" in aligned else eligible.astype(np.float32)
        return cls(timestamps, station_ids, station_links,
                   np.array([panel.link_index[l] for l in station_links], dtype=np.int64),
                   lanes, speed / 25.0, flow / lanes[None, :] / 600.0, eligible, occupancy, quality)


def station_features(day: StationDay, panel: PanelInfo, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate each station after masking; never aggregate a hidden answer."""
    steps, stations = day.speed.shape
    context = np.zeros((steps, stations, 3), dtype=np.float32)
    position = np.arange(steps)
    for station in range(stations):
        good = np.flatnonzero(visible[:, station])
        if len(good):
            context[:, station, 0] = np.interp(position, good, day.speed[good, station])
            context[:, station, 1] = np.interp(position, good, day.flow_lane[good, station])
    context[:, :, 2] = day.lanes[None, :] / 5.0
    inputs = link_features(day.timestamps, day.speed, day.flow_lane, visible, day.station_to_link, panel)
    return inputs, context


def station_window_features(day: StationDay, panel: PanelInfo, visible: np.ndarray,
                            index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = StationDay([day.timestamps[i] for i in index], day.station_ids, day.station_links,
                        day.station_to_link, day.lanes, day.speed[index], day.flow_lane[index], day.eligible[index])
    return station_features(window, panel, visible[index])


def link_features(timestamps: list[str], speed: np.ndarray, flow: np.ndarray, visible: np.ndarray,
                  station_to_link: np.ndarray, panel: PanelInfo) -> np.ndarray:
    steps = len(timestamps)
    nodes = len(panel.links)
    values = np.zeros((steps, nodes, 9), dtype=np.float32)
    counts = np.zeros((steps, nodes), dtype=np.float32)
    for station, link in enumerate(station_to_link):
        good = visible[:, station]
        values[good, link, 0] += speed[good, station]
        values[good, link, 1] += flow[good, station]
        counts[good, link] += 1
    values[:, :, 0:2] /= np.maximum(counts[:, :, None], 1)
    values[:, :, 2] = (counts > 0).astype(np.float32)
    values[:, :, 3] = panel.free_speed[None, :] / 125.0
    values[:, :, 4] = panel.lanes[None, :] / 5.0
    values[:, :, 5] = panel.lengths[None, :] / 5.0
    dates = pd.to_datetime(timestamps, utc=True, format="ISO8601")
    slot = np.asarray(dates.hour * 12 + dates.minute // 5)
    values[:, :, 6] = np.sin(2 * np.pi * slot / 288)[:, None]
    values[:, :, 7] = np.cos(2 * np.pi * slot / 288)[:, None]
    values[:, :, 8] = dates.dayofweek.to_numpy()[:, None] / 6.0
    return values


# The first nine channels retain the legacy order; v2 checkpoints store this list.
FEATURE_NAMES = ("speed", "flow_lane", "both_observed", "free_speed", "lanes", "length",
                 "time_sin", "time_cos", "weekday", "occupancy", "speed_observed", "flow_observed",
                 "occupancy_observed", "pct_observed", "speed_age", "flow_age", "occupancy_age",
                 "capacity", "road_position", "on_ramp_attached", "off_ramp_attached", "on_ramp_flow",
                 "off_ramp_flow", "on_ramp_valid", "off_ramp_valid", "prior_speed", "prior_flow_lane",
                 "prior_occupancy", "time_slot", "day_of_week", "relative_time")

QUEUE_CHANGE_NAMES = ("speed_margin",) + tuple(
    f"{channel}_{field}_{minutes}m" for minutes in (5, 15)
    for channel in ("speed", "flow_lane", "occupancy") for field in ("delta", "valid")
)


def queue_feature_names(changes=False):
    return FEATURE_NAMES + QUEUE_CHANGE_NAMES if changes else FEATURE_NAMES

STATE_CONTEXT_NAMES = ("speed", "flow_lane", "lanes") + tuple(
    f"{channel}_{field}" for channel in ("speed", "flow")
    for field in ("left", "right", "left_age", "right_age", "left_valid", "right_valid", "observed")
) + ("pct_observed", "prior_speed", "prior_flow_lane")


def aggregate_channels(day: StationDay, panel: PanelInfo, visible: np.ndarray | None = None):
    values = np.stack((day.speed, day.flow_lane,
                       day.occupancy if day.occupancy is not None else np.full_like(day.speed, np.nan)), -1)
    good = np.isfinite(values)
    if visible is not None:
        good &= visible[..., None] if visible.ndim == 2 else visible
    sums = np.zeros((len(day.timestamps), len(panel.links), 3), np.float32)
    counts = np.zeros_like(sums)
    quality = np.zeros(sums.shape[:2], np.float32)
    qcount = np.zeros_like(quality)
    for station, link in enumerate(day.station_to_link):
        sums[:, link] += np.where(good[:, station], values[:, station], 0)
        counts[:, link] += good[:, station]
        present = good[:, station].any(-1)
        q = day.quality[:, station] if day.quality is not None else present.astype(np.float32)
        # Observation quality remains public even when measured values are masked.
        quality[:, link] += np.where(np.isfinite(q), q, 0)
        qcount[:, link] += np.isfinite(q)
    return sums / np.maximum(counts, 1), counts > 0, quality / np.maximum(qcount, 1)


def features_v2(day: StationDay, panel: PanelInfo, profile: np.ndarray,
                ramp: np.ndarray | None = None, visible: np.ndarray | None = None):
    values, good, quality = aggregate_channels(day, panel, visible)
    steps, nodes = values.shape[:2]
    out = np.zeros((steps, nodes, len(FEATURE_NAMES)), np.float32)
    dates = pd.to_datetime(day.timestamps, utc=True, format="ISO8601")
    slots = np.asarray(dates.hour * 12 + dates.minute // 5)
    dow = dates.dayofweek.to_numpy()
    out[..., :2] = values[..., :2]
    out[..., 2] = good[..., :2].all(-1)
    out[..., 3] = panel.free_speed / 125
    out[..., 4] = panel.lanes / 5
    out[..., 5] = panel.lengths / 5
    out[..., 6] = np.sin(2 * np.pi * slots / 288)[:, None]
    out[..., 7] = np.cos(2 * np.pi * slots / 288)[:, None]
    out[..., 8] = dow[:, None] / 6
    out[..., 9] = values[..., 2]
    out[..., 10:13] = good
    out[..., 13] = quality
    elapsed = np.asarray((dates - dates[0]).total_seconds() / 300, dtype=np.float32)
    for channel in range(3):
        last = np.maximum.accumulate(np.where(good[..., channel], elapsed[:, None], -72), axis=0)
        out[..., 14 + channel] = np.minimum(elapsed[:, None] - last, 72) / 72
    out[..., 17] = (panel.capacity if panel.capacity is not None else np.full(nodes, 6000)) / 10000
    out[..., 18] = panel.position if panel.position is not None else np.arange(nodes) / max(nodes - 1, 1)
    for link, kind in panel.ramp_map.values():
        out[:, link, 19 + kind] = 1
    if ramp is not None:
        out[..., 21:23] = ramp[..., :2] / 600
        out[..., 23:25] = ramp[..., 2:]
    out[..., 25:28] = profile[dow * 288 + slots]
    out[..., 28] = slots[:, None] / 287
    out[..., 29] = dow[:, None] / 6
    out[..., 30] = np.arange(steps)[:, None] / max(steps - 1, 1)
    return out


def queue_features(day, panel, profile, ramp=None, origin=None, changes=False):
    """Build Queue inputs from the supplied past and optional released origin only."""
    x = features_v2(day, panel, profile, ramp)
    if origin is None and not changes:
        return x
    times = pd.to_datetime(day.timestamps, utc=True, format="ISO8601")
    if origin is not None:
        if origin.shape != (1, len(panel.links), len(FEATURE_NAMES)):
            raise ValueError("Unexpected released Queue origin shape")
        x = np.concatenate((x, origin), axis=0)
        times = times.append(pd.DatetimeIndex([times[-1] + pd.Timedelta(minutes=5)]))
        elapsed = np.asarray((times - times[0]).total_seconds() / 300, dtype=np.float32)
        for channel in range(3):
            last = np.maximum.accumulate(np.where(x[..., 10 + channel] > 0, elapsed[:, None], -72), axis=0)
            x[..., 14 + channel] = np.minimum(elapsed[:, None] - last, 72) / 72
        x[..., 30] = np.arange(len(x))[:, None] / (len(x) - 1)
    if not changes:
        return x
    extra = np.zeros((*x.shape[:2], len(QUEUE_CHANGE_NAMES)), np.float32)
    extra[..., 0] = np.where(x[..., 10] > 0, x[..., 0] - panel.cutoffs[None] / 25, 0)
    column = 1
    for lag in (1, 3):
        consecutive = np.asarray(times[lag:] - times[:-lag] == pd.Timedelta(minutes=5 * lag))
        for channel, position in enumerate((0, 1, 9)):
            good = (x[lag:, :, 10 + channel] > 0) & (x[:-lag, :, 10 + channel] > 0) & consecutive[:, None]
            extra[lag:, :, column] = np.where(good, x[lag:, :, position] - x[:-lag, :, position], 0)
            extra[lag:, :, column + 1] = good
            column += 2
    return np.concatenate((x, extra), axis=-1)


def slice_day(day: StationDay, index: np.ndarray) -> StationDay:
    return StationDay([day.timestamps[i] for i in index], day.station_ids, day.station_links,
                      day.station_to_link, day.lanes, day.speed[index], day.flow_lane[index], day.eligible[index],
                      None if day.occupancy is None else day.occupancy[index],
                      None if day.quality is None else day.quality[index])


def state_features_v2(day: StationDay, panel: PanelInfo, index: np.ndarray,
                      profile: np.ndarray, ramp: np.ndarray | None = None, rich=False):
    window = slice_day(day, index)
    x = features_v2(window, panel, profile, None if ramp is None else ramp[index])
    context = np.zeros((len(index), len(day.station_ids), len(STATE_CONTEXT_NAMES) if rich else 3), np.float32)
    dates = pd.to_datetime(window.timestamps, utc=True, format="ISO8601")
    slots = np.asarray(dates.dayofweek * 288 + dates.hour * 12 + dates.minute // 5)
    elapsed = np.asarray((dates - dates[0]).total_seconds() / 300, dtype=np.float32)
    position = np.arange(len(index))[:, None]
    for channel, values in enumerate((window.speed, window.flow_lane)):
        good = np.isfinite(values)
        prior = profile[slots[:, None], day.station_to_link[None], channel]
        # Search all stations together, within the masked window only.
        left = np.maximum.accumulate(np.where(good, position, -1), axis=0)
        right = np.minimum.accumulate(np.where(good, position, len(index))[::-1], axis=0)[::-1]
        left_valid, right_valid = left >= 0, right < len(index)
        left_safe, right_safe = np.clip(left, 0, len(index) - 1), np.clip(right, 0, len(index) - 1)
        left_value = np.take_along_axis(values, left_safe, axis=0)
        right_value = np.take_along_axis(values, right_safe, axis=0)
        first = np.where(left_valid, left_value, np.where(right_valid, right_value, prior)).astype(np.float64)
        last = np.where(right_valid, right_value, first).astype(np.float64)
        fraction = np.clip((position - left) / np.maximum(right - left, 1), 0, 1)
        # Match np.interp's index-based interpolation and constant endpoint extension.
        context[..., channel] = first + fraction * (last - first)
        if rich:
            offset = 3 + 7 * channel
            for side, safe, valid, endpoint in ((0, left_safe, left_valid, left_value),
                                               (1, right_safe, right_valid, right_value)):
                context[..., offset + side] = np.where(valid, endpoint, prior)
                context[..., offset + 2 + side] = np.where(valid, np.minimum(np.abs(elapsed[:, None] - elapsed[safe]), 72) / 72, 1)
                context[..., offset + 4 + side] = valid
            context[..., offset + 6] = good
            context[..., 18 + channel] = prior
    context[..., 2] = day.lanes / 5
    if rich:
        context[..., 17] = (np.clip(np.nan_to_num(window.quality), 0, 1)
                            if window.quality is not None else np.isfinite(window.speed).astype(np.float32))
    return x, context


def ramp_features(release: Path, panel: PanelInfo, split: str, timestamps: list[str]) -> np.ndarray:
    out = np.zeros((len(timestamps), len(panel.links), 4), np.float32)
    out[..., 2:] = 1  # A road without an attached ramp has known zero ramp flow.
    expected = np.zeros((len(panel.links), 2), np.float32)
    for link, kind in panel.ramp_map.values():
        expected[link, kind] += 1
    counts = np.zeros_like(out[..., :2])
    date = timestamps[0][:10]
    root = release / "corridors" / panel.name / split / "ramp_states"
    paths = sorted(set(root.glob(f"**/*{date}*.parquet")) | set(root.glob(f"**/*{date.replace('-', '_')}*.parquet")))
    time_index = {value: i for i, value in enumerate(timestamps)}
    for path in paths:
        frame = pd.read_parquet(path, columns=["timestamp", "ramp_link_id", "flow_vph", "pct_observed"])
        frame = frame[frame.timestamp.isin(time_index)].drop_duplicates(["timestamp", "ramp_link_id"])
        for row in frame.itertuples(index=False):
            if row.ramp_link_id in panel.ramp_map and pd.notna(row.flow_vph) and row.pct_observed >= 75:
                link, kind = panel.ramp_map[row.ramp_link_id]
                index = time_index[row.timestamp]
                out[index, link, kind] += float(row.flow_vph)
                counts[index, link, kind] += 1
    out[..., 2:] = counts >= expected[None]
    out[..., :2] *= out[..., 2:]
    return out


class TargetIndex:
    """Compact, cached exact template targets; never infer targets from nulls."""
    def __init__(self, release: Path, panel: str, station_ids: list[str]):
        source = release / "task1" / panel / "train" / "sample_submission_state.csv"
        signature = hashlib.sha256(json.dumps([source.stat().st_size, source.stat().st_mtime_ns, station_ids]).encode()).hexdigest()[:16]
        folder = release.parent / "processed_v2"
        folder.mkdir(parents=True, exist_ok=True)
        cache = folder / f"{panel}-targets-{signature}.npz"
        if not cache.exists():
            print(f"Preparing exact Task 1 targets: {panel}", flush=True)
            lookup = {station: i for i, station in enumerate(station_ids)}
            by_date = {}
            regimes = {}
            for frame in pd.read_csv(source, usecols=["timestamp", "station_id", "mask_regime"], dtype=str, chunksize=200000):
                positions = frame.station_id.map(lookup)
                if positions.isna().any():
                    raise ValueError(f"Template has unknown stations: {panel}")
                slot = frame.timestamp.str[11:13].astype(np.int64) * 12 + frame.timestamp.str[14:16].astype(np.int64) // 5
                frame = frame.assign(_date=frame.timestamp.str[:10], _cell=(slot * len(station_ids) + positions).astype(np.uint32))
                for date, part in frame.groupby("_date", sort=False):
                    unique = part.mask_regime.unique()
                    if len(unique) != 1 or date in regimes and regimes[date] != unique[0]:
                        raise ValueError(f"Multiple regimes in {panel} {date}")
                    regimes[date] = unique[0]
                    by_date.setdefault(date, []).append(part._cell.to_numpy())
            arrays = {date: np.concatenate(parts) for date, parts in by_date.items()}
            if any(len(np.unique(cells)) != len(cells) for cells in arrays.values()):
                raise ValueError(f"Duplicate Task 1 template targets: {panel}")
            arrays["dates"] = np.array(list(regimes))
            arrays["regimes"] = np.array(list(regimes.values()))
            temporary = cache.with_suffix(".pending.npz")
            np.savez_compressed(temporary, **arrays)
            temporary.replace(cache)
        self.archive = np.load(cache, allow_pickle=False)
        self.regimes = dict(zip(self.archive["dates"].tolist(), self.archive["regimes"].tolist()))
        self.stations = station_ids

    def get(self, day: StationDay):
        date = day.timestamps[0][:10]
        if day.station_ids != self.stations or date not in self.regimes:
            raise ValueError(f"Target mapping changed or absent: {date}")
        mask = np.zeros(day.speed.shape, bool)
        slots = {int(value[11:13]) * 12 + int(value[14:16]) // 5: i for i, value in enumerate(day.timestamps)}
        cells = self.archive[date]
        mask[[slots[int(slot)] for slot in cells // len(self.stations)], cells % len(self.stations)] = True
        if (mask & ~day.eligible).any():
            raise ValueError(f"Template target is ineligible: {date}")
        return self.regimes[date], mask
