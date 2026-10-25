"""Local MagiNet Task 1/3 runner, adapted from task13_maginet_colab.ipynb.

Legacy presets retain the notebook architecture. Optional context/channel graph
paths and State-based selection are documented with local CLI commands in README.md.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from functools import lru_cache, partial
import hashlib, heapq, json, random, re, shutil, tempfile, time, zipfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / 'data' / 'kaggle_public'
INPUT_NORMALIZATION = 'standard'  # 'fixed': 기존 /25·/600, 'standard': fit 관측으로 입력 표준화
EXPERIMENT = 'temporal_direct'
# 한 번에 한 요소만 비교합니다. 검증된 조합은 이 표에 별도 이름으로 추가하세요.
EXPERIMENTS = {
    'baseline':        (0.10, False, False, False),
    'dn_003':          (0.03, False, False, False),
    'dn_zero':         (0.00, False, False, False),
    'temporal_direct': (0.10, True,  False, False),
    'mask_resample':   (0.10, False, True,  False),
    'learned_graph':   (0.10, False, False, True),
    'temporal_direct_dn003': (0.03, True, False, False),
    'temporal_direct_dnzero': (0.00, True, False, False),
    'temporal_direct_learned_graph': (0.10, True, False, True),
    'temporal_direct_dn003_learned_graph': (0.03, True, False, True),
    'temporal_direct_dn003_context_graph': (0.03, True, False, True),
    'temporal_direct_dn003_channel_graph': (0.03, True, False, True),
}
SPATIAL_EXPERIMENTS = {
    'temporal_direct_dn003_context_graph': 'context',
    'temporal_direct_dn003_channel_graph': 'channel',
}
SPATIAL_MODE = 'legacy'
if EXPERIMENT not in EXPERIMENTS:
    raise ValueError(f'알 수 없는 실험: {EXPERIMENT}')
DN_WEIGHT, TEMPORAL_DIRECT, MASK_RESAMPLE, LEARNED_GRAPH = EXPERIMENTS[EXPERIMENT]
REPORT_BASE = PROJECT_ROOT / 'reports' / 'maginet'
REPORT_DIR = REPORT_BASE / EXPERIMENT
CACHE_STORE_DIR = PROJECT_ROOT / 'data' / 'maginet_cache'
CACHE_METADATA_DIR = CACHE_STORE_DIR / 'metadata'
SAVED_CACHE_ZIP = None
CACHE_DIR = CACHE_STORE_DIR  # ZIP을 한 번 풀 로컬 폴더
CACHE_MODE = 'build'             # 'reuse': 누락 시 중단, 'build': 없는 날짜만 원본에서 생성
CACHE_PROFILE_DIR = CACHE_STORE_DIR / 'legacy_profiles'  # 이전 config.json·fit_profile-*.npy
PANELS = None                    # None: 전체 패널 공동 학습
MAX_TRAIN_DAYS = None            # 실행 점검 시 12 이상으로 줄일 수 있습니다.
DAYS_PER_REGIME = None          # 점검 시 각 기간·regime의 날짜 수를 제한합니다.
DATE_OVERRIDES = {}
SEED = 42
WINDOW, STRIDE = 72, 36
WIDTH, BLOCKS, HEADS, CHEB_K = 64, 3, 4, 3
GRAPH_DIM = 128
DROPOUT = 0.05
MGL_THRESHOLD = 0.5              # 공식 구현의 missing graph 판정 시작값
BATCH_SIZE = 2
BATCH_POLICY = 'fixed'           # adaptive: 관측소 128개 이하 8, 초과 4
PANEL_BATCH = {}                  # 명시한 패널의 배치가 policy보다 우선합니다.
UPDATES_PER_GROUP = 24           # epoch당 패널×regime 업데이트
EPOCHS, PATIENCE = 50, 8
LR, WEIGHT_DECAY = 1e-3, 1e-4
WARMUP_EPOCHS, PHYSICS_RAMP_EPOCHS = 5, 5
LOSS_WEIGHTS = dict(N=0.05, dN=DN_WEIGHT, FD=0.0)
# 공식 Task 1·3 비중을 참고한 대리 선택식입니다. 공식 Physics 점수가 아닙니다.
SELECT_WEIGHTS = dict(State=0.70, FD_proxy=0.10, LWR_proxy=0.20)
CHECKPOINT_METRIC = 'selection_proxy'
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_MODEL = True              # False: 현재 실험의 저장된 best로 평가·오차 분해만 실행
RESUME = False                   # last_maginet.pt에서 재개; EPOCHS를 마지막 epoch보다 크게 설정
LWR_MODES = {}                   # 공개 기본값 A; 패널별 B/C 실험 시 명시

REGIMES = ('R1', 'R2', 'R3')
KEYS = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
SCALE = np.array([25., 600.], dtype=np.float32)
PERIODS = dict(fit=('2030-06-01', '2030-12-31'), early=('2031-01-01', '2031-01-15'),
               blend=('2031-01-16', '2031-01-31'), february=('2031-02-01', '2031-02-28'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = DEVICE.type == 'cuda'
AMP_DTYPE = torch.bfloat16 if USE_AMP and torch.cuda.is_bf16_supported() else torch.float16
assert tuple(map(int, re.match(r'(\d+)\.(\d+)', torch.__version__).groups())) >= (2, 3)
assert 12 <= WINDOW <= 288 and 1 <= STRIDE <= WINDOW
assert WIDTH % HEADS == 0 and WIDTH % 2 == 0 and CHEB_K >= 1 and BLOCKS >= 1
assert BATCH_SIZE >= 1 and UPDATES_PER_GROUP >= 1 and EPOCHS >= 1
assert INPUT_NORMALIZATION in ('fixed', 'standard') and CACHE_MODE in ('reuse', 'build')
assert all(v >= 0 for v in LOSS_WEIGHTS.values()) and sum(SELECT_WEIGHTS.values()) > 0
assert all(v >= 0 for v in SELECT_WEIGHTS.values())
assert WARMUP_EPOCHS >= 0 and PHYSICS_RAMP_EPOCHS >= 0
assert len(set(ALPHAS)) == len(ALPHAS) and all(0 <= a <= 1 for a in ALPHAS) and 0.0 in ALPHAS and 1.0 in ALPHAS

def reset_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_layout(link_ids: list[str], topology: pd.DataFrame) -> list[str]:
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
        ordered.extend(part)
    return ordered

@dataclass
class PanelInfo:
    name: str
    links: list[str]
    link_index: dict[str, int]
    free_speed: np.ndarray
    lanes: np.ndarray
    lengths: np.ndarray
    station_lanes: dict[str, float]
    capacity: np.ndarray | None = None
    jam_density: np.ndarray | None = None
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
        links = scan_layout(sorted(set(observed) | set(topology.link_id.dropna().astype(str))), topology)
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
        ramp_map = {}
        ramp_path = network / "ramp_attachment_map.csv"
        if ramp_path.exists():
            for row in pd.read_csv(ramp_path, dtype=str).itertuples(index=False):
                if row.nearest_mainline_link_id in lookup:
                    if row.ramp_type not in {"OR", "FR", "on", "off"}:
                        raise ValueError(f"Unknown ramp type: {row.ramp_type}")
                    ramp_map[row.ramp_link_id] = (lookup[row.nearest_mainline_link_id], int(row.ramp_type in {"FR", "off"}))
        return cls(panel, links, lookup, free_speed, lanes, lengths, station_lanes,
                   parameter("capacity_vph", 6000.0), np.maximum(parameter("k_jam", np.nan), 1.05 * parameter("capacity_vph", 6000.0) / np.maximum(free_speed, 1)),
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

def ramp_files(release, panel, split, date):
    root = release / 'corridors' / panel.name / split / 'ramp_states'
    return sorted(set(root.glob(f"**/*{date}*.parquet"))
                  | set(root.glob(f"**/*{date.replace('-', '_')}*.parquet")))


def ramp_features(release: Path, panel: PanelInfo, split: str, timestamps: list[str]) -> np.ndarray:
    out = np.zeros((len(timestamps), len(panel.links), 4), np.float32)
    out[..., 2:] = 1  # A road without an attached ramp has known zero ramp flow.
    expected = np.zeros((len(panel.links), 2), np.float32)
    for link, kind in panel.ramp_map.values():
        expected[link, kind] += 1
    counts = np.zeros_like(out[..., :2])
    time_index = {value: i for i, value in enumerate(timestamps)}
    for path in ramp_files(release, panel, split, timestamps[0][:10]):
        frame = pd.read_parquet(path, columns=["timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
        frame = frame[frame.timestamp.isin(time_index)].drop_duplicates(["timestamp", "ramp_link_id"])
        for row in frame.itertuples(index=False):
            if row.ramp_link_id in panel.ramp_map and np.isfinite(row.flow_vph) and row.pct_observed >= 75 and bool(row.is_score_eligible):
                link, kind = panel.ramp_map[row.ramp_link_id]
                index = time_index[row.timestamp]
                out[index, link, kind] += float(row.flow_vph)
                counts[index, link, kind] += 1
    out[..., 2:] = counts >= expected[None]
    out[..., :2] *= out[..., 2:]
    return out

def dated_files(root):
    result = {}
    for path in sorted(Path(root).rglob("*.parquet")):
        match = re.search(r"\d{4}[-_]\d{2}[-_]\d{2}", path.name)
        if match is None:
            raise ValueError(f"Date missing from filename: {path}")
        date = match.group().replace("_", "-")
        if date in result:
            raise ValueError(f"Duplicate date: {date}")
        result[date] = path
    if not result:
        raise FileNotFoundError(f"No parquet files: {root}")
    return result

def fit_prior(panel, paths):
    """Only fit dates contribute to weekday/slot means and fallbacks."""
    sums = np.zeros((2016, len(panel.links), 3), np.float64)
    counts = np.zeros_like(sums)
    for number, path in enumerate(paths, 1):
        if number == 1 or number % 30 == 0 or number == len(paths):
            print(f"과거 평균 계산 {panel.name}: {number}/{len(paths)} days", flush=True)
        day = StationDay.load(path, panel)
        values, valid, _ = aggregate_channels(day, panel, day.eligible)
        dates = pd.to_datetime(day.timestamps, utc=True)
        slot = np.asarray(dates.dayofweek * 288 + dates.hour * 12 + dates.minute // 5)
        np.add.at(sums, slot, values * valid)
        np.add.at(counts, slot, valid)
    mean = sums.sum((0, 1)) / np.maximum(counts.sum((0, 1)), 1)
    link_mean = np.divide(sums.sum(0), counts.sum(0),
                          out=np.broadcast_to(mean, sums.shape[1:]).copy(),
                          where=counts.sum(0) > 0)
    return np.divide(sums, counts, out=np.broadcast_to(link_mean, sums.shape).copy(),
                     where=counts > 0).astype(np.float32)

class TargetIndex:
    def __init__(self, release, name, day, split='train'):
        path = release / 'task1' / name / split / 'sample_submission_state.csv'
        self.stations, self.links = day.station_ids, day.station_links
        self.archive, self.regimes = {}, {}
        station_lookup = {s: i for i, s in enumerate(self.stations)}
        parts = {}
        for frame in pd.read_csv(path, usecols=KEYS, dtype=str, chunksize=200_000):
            positions = frame.station_id.map(station_lookup)
            if positions.isna().any() or set(frame.panel) != {name}:
                raise ValueError(f'템플릿의 panel/station이 잘못되었습니다: {path}')
            if not np.array_equal(np.asarray(self.links)[positions.to_numpy(int)], frame.link_id.to_numpy()):
                raise ValueError(f'템플릿 station/link 연결이 다릅니다: {path}')
            times = pd.to_datetime(frame.timestamp, utc=True, format='ISO8601')
            if ((times.dt.minute % 5 != 0) | (times.dt.second != 0)).any():
                raise ValueError('템플릿은 5분 시각이어야 합니다.')
            slots = times.dt.hour * 12 + times.dt.minute // 5
            frame = frame.assign(_date=frame.timestamp.str[:10],
                                 _cell=(slots * len(self.stations) + positions).astype(np.uint32))
            for date, part in frame.groupby('_date', sort=False):
                regimes = part.mask_regime.unique()
                if len(regimes) != 1 or regimes[0] not in REGIMES:
                    raise ValueError(f'날짜별 regime이 잘못되었습니다: {date}')
                if date in self.regimes and self.regimes[date] != regimes[0]:
                    raise ValueError(f'날짜별 regime이 바뀝니다: {date}')
                self.regimes[date] = regimes[0]
                parts.setdefault(date, []).append(part._cell.to_numpy())
        if not parts:
            raise ValueError(f'빈 템플릿: {path}')
        self.archive = {d: np.concatenate(p) for d, p in parts.items()}
        if any(len(np.unique(a)) != len(a) for a in self.archive.values()):
            raise ValueError(f'중복 대상 셀: {path}')

    def get(self, day, require_eligible=True):
        date = day.timestamps[0][:10]
        if (day.station_ids, day.station_links) != (self.stations, self.links) or date not in self.archive:
            raise ValueError(f'대상 관측소 구성이 달라졌습니다: {date}')
        if any(t[:10] != date for t in day.timestamps):
            raise ValueError('하루 파일에 여러 날짜가 있습니다.')
        slots = {int(t[11:13]) * 12 + int(t[14:16]) // 5: i for i, t in enumerate(day.timestamps)}
        mask = np.zeros(day.speed.shape, bool)
        cells = self.archive[date]
        mask[[slots[int(s)] for s in cells // len(self.stations)], cells % len(self.stations)] = True
        if require_eligible and (mask & ~day.eligible).any():
            raise ValueError(f'정답이 없는 대상 셀: {date}')
        return self.regimes[date], mask

def available_panels(release):
    release = Path(release).expanduser().resolve()
    if not (release / 'corridors').is_dir() or not (release / 'task1').is_dir():
        raise FileNotFoundError(f'RELEASE_ROOT에 corridors/와 task1/이 필요합니다: {release}')
    available = sorted(p.name for p in (release / 'corridors').iterdir() if p.is_dir())
    names = available if PANELS is None else sorted(set(PANELS))
    if not names or set(names) - set(available):
        raise ValueError(f'잘못된 PANELS: {names}')
    return names

def station_graph(panel, day):
    # 무관측 connector를 따라 실제 도로에서 가장 가까운 관측 링크를 찾습니다.
    observed = set(day.station_to_link.tolist())
    neighbours = [set() for _ in panel.links]
    for a, b in panel.edges.T:
        neighbours[a].add(int(b))
        neighbours[b].add(int(a))
    links = np.eye(len(panel.links), dtype=bool)
    for origin in observed:
        pending, seen = list(neighbours[origin]), {origin}
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            if node in observed:
                links[origin, node] = True
            else:
                pending.extend(neighbours[node] - seen)
    return links[day.station_to_link[:, None], day.station_to_link[None, :]]

def prepare_panel(release, name, code_id, station_offset):
    started = time.perf_counter()
    panel = PanelInfo.load(release, name)
    files = dated_files(release / 'corridors' / name / 'train' / 'mainline_states')
    masked = dated_files(release / 'corridors' / name / 'train' / 'mainline_states_masked')
    dates = {p: [d for d in files if lo <= d <= hi] for p, (lo, hi) in PERIODS.items()}
    if name in DATE_OVERRIDES:
        saved_dates = DATE_OVERRIDES[name]
        if any(not set(saved_dates[p]) <= set(dates[p]) for p in PERIODS):
            raise ValueError(f'저장된 날짜가 로컬 데이터 기간과 다릅니다: {name}')
        dates = {p: list(saved_dates[p]) for p in PERIODS}
    if MAX_TRAIN_DAYS is not None:
        if MAX_TRAIN_DAYS < 1:
            raise ValueError('MAX_TRAIN_DAYS는 양수 또는 None이어야 합니다.')
        indices = np.linspace(0, len(dates['fit']) - 1, min(MAX_TRAIN_DAYS, len(dates['fit'])), dtype=int)
        dates['fit'] = [dates['fit'][i] for i in indices]
    if any(not dates[p] for p in ('fit', 'early', 'blend')):
        raise ValueError(f'학습·조기 종료·선택 구간이 모두 필요합니다: {name}')
    first = StationDay.load(files[dates['fit'][0]], panel)
    observed_links = np.unique(first.station_to_link)
    for attribute, fallback in (('free_speed', 100.), ('capacity', 1800. * panel.lanes), ('lengths', 1.)):
        values = getattr(panel, attribute)
        invalid = ~np.isfinite(values) | ((values < 0) if attribute == 'lengths' else (values <= 0))
        if invalid[observed_links].any():
            raise ValueError(f'관측 링크의 {attribute}가 잘못되었습니다: {name}')
        # 무관측 connector의 placeholder 값이 손실 연산에 NaN을 전파하지 않도록 합니다.
        setattr(panel, attribute, np.where(invalid, fallback, values).astype(np.float32))
    panel.jam_density = np.where(np.isfinite(panel.jam_density), panel.jam_density, 100 * panel.lanes)
    panel.jam_density = np.maximum(panel.jam_density, 1.05 * panel.capacity / panel.free_speed)
    target_index = TargetIndex(release, name, first)
    if DAYS_PER_REGIME is not None:
        for period, candidates in dates.items():
            chosen = []
            for regime in REGIMES:
                pool = [d for d in candidates if target_index.regimes.get(d) == regime]
                indices = np.linspace(0, len(pool) - 1, min(DAYS_PER_REGIME, len(pool)), dtype=int)
                chosen.extend(pool[i] for i in indices)
            dates[period] = sorted(chosen)
    for p in ('fit', 'early', 'blend'):
        if {target_index.regimes.get(d) for d in dates[p]} != set(REGIMES):
            raise ValueError(f'{name}/{p}의 R1·R2·R3가 부족합니다. 날짜 수를 늘려 주세요.')
        if not set(dates[p]) <= set(masked):
            raise ValueError(f'masked 날짜가 누락되었습니다: {name}/{p}')
    mode = LWR_MODES.get(name, 'A').upper()
    if mode not in ('A', 'B', 'C'):
        raise ValueError(f'지원하지 않는 LWR mode: {mode}')
    out = dict(release=release, panel=panel, files=files, masked=masked, dates=dates,
               targets=target_index, profile=load_fit_profile(panel, target_index, dates['fit'], files),
               panel_code=code_id, station_codes=np.arange(station_offset, station_offset + len(first.station_ids)),
               mapping=first.station_to_link, lanes=first.lanes, station_graph=station_graph(panel, first),
               mode=mode)
    topology = pd.read_csv(release / 'corridors' / name / 'network' / 'lwr_mainline_topology.csv', dtype=str)
    out['topology_nodes'] = {panel.link_index[l] for l in topology.link_id.dropna().astype(str)}
    out['preparation_seconds'] = time.perf_counter() - started
    print({'panel': name, 'stations': len(first.station_ids), 'days': {p: len(ds) for p, ds in dates.items()},
           'preparation_seconds': round(out['preparation_seconds'], 1)}, flush=True)
    return out

def link_mean_numpy(values, active, mapping, nodes):
    sums = np.zeros((len(values), nodes, values.shape[-1]), np.float64)
    counts = np.zeros((len(values), nodes), np.float64)
    np.add.at(sums.transpose(1, 0, 2), mapping, np.where(active[..., None], values, 0).transpose(1, 0, 2))
    np.add.at(counts.T, mapping, active.T)
    return sums / np.maximum(counts[..., None], 1), counts

def nearest_features(values, prior, elapsed):
    good = np.isfinite(values)
    t = np.arange(len(values))[:, None]
    left = np.maximum.accumulate(np.where(good, t, -1), axis=0)
    right = np.minimum.accumulate(np.where(good, t, len(values))[::-1], axis=0)[::-1]
    lv, rv = left >= 0, right < len(values)
    ls, rs = np.clip(left, 0, len(values) - 1), np.clip(right, 0, len(values) - 1)
    a, b = np.take_along_axis(values, ls, axis=0), np.take_along_axis(values, rs, axis=0)
    a = np.where(lv, a, np.where(rv, b, prior))
    b = np.where(rv, b, a)
    la, ra = elapsed[:, None] - elapsed[ls], elapsed[rs] - elapsed[:, None]
    fraction = np.divide(la, la + ra, out=np.zeros_like(la), where=(la + ra) > 0).clip(0, 1)
    base = a + fraction * (b - a)
    la, ra = np.where(lv, la, np.nan), np.where(rv, ra, np.nan)
    return base, la, ra

def macro_mean(panel_values):
    families = {}
    for name, value in panel_values.items():
        families.setdefault(name.rsplit('_', 1)[0], []).append(value)
    return float(np.mean([np.mean(values) for values in families.values()]))

AUX_NAMES = ('quality', 'speed_visible', 'flow_visible', 'occupancy_visible',
             'speed_left_age', 'speed_right_age', 'flow_left_age', 'flow_right_age',
             'profile_speed', 'profile_flow', 'profile_occupancy',
             'lanes', 'free_speed', 'capacity_per_lane', 'length',
             'ramp_in', 'ramp_out', 'ramp_in_valid', 'ramp_out_valid', 'on_attached', 'off_attached')


def restore_input_cache(names, archive_path, destination):
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path is not None:
        archive_path = Path(archive_path).expanduser()
        if not archive_path.is_file():
            raise FileNotFoundError(f'저장 캐시 ZIP이 없습니다: {archive_path}. 새 캐시를 만들 때는 SAVED_CACHE_ZIP=None, CACHE_MODE="build"로 설정해 주세요.')
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            for member in members:
                relative = PurePosixPath(member.filename)
                if (relative.is_absolute() or '..' in relative.parts or '\\' in member.filename or ':' in member.filename
                        or not (destination / member.filename).resolve().is_relative_to(destination.resolve())
                        or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise ValueError(f'안전하지 않은 ZIP 경로: {member.filename}')
            fingerprint = hashlib.sha256(json.dumps(
                [(m.filename, m.file_size, m.CRC) for m in members], sort_keys=True).encode()).hexdigest()
            marker = destination / '.cache_source.json'
            if marker.exists() and json.loads(marker.read_text())['fingerprint'] != fingerprint:
                raise ValueError('다른 ZIP을 푼 폴더입니다. CACHE_DIR를 새 로컬 경로로 지정해 주세요.')
            pending = [m for m in members if not m.is_dir() and
                       (not marker.exists() or not (destination / m.filename).is_file())]
            if pending:
                needed = sum(m.file_size for m in pending)
                if shutil.disk_usage(destination).free < needed + 2 * 2**30:
                    raise RuntimeError(f'캐시 ZIP 복원에 {needed / 2**30:.1f} GiB + 여유 2 GiB가 필요합니다.')
                began = time.perf_counter()
                for number, member in enumerate(pending, 1):
                    target = destination / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + '.tmp')
                    with archive.open(member) as src, temporary.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    temporary.replace(target)
                    if number == 1 or number % 100 == 0 or number == len(pending):
                        print(f'저장 ZIP 복원: {number}/{len(pending)}파일', flush=True)
                marker_temporary = marker.with_suffix('.json.tmp')
                marker_temporary.write_text(json.dumps(dict(fingerprint=fingerprint, archive=str(archive_path))))
                marker_temporary.replace(marker)
                print({'cache_restore_seconds': round(time.perf_counter() - began, 1)})
            else:
                print('같은 ZIP이 이미 복원되어 로컬 NPZ를 재사용합니다.')
    roots = {p.parent.parent.parent for p in destination.rglob('*.npz')
             if p.parent.name in PERIODS and p.parent.parent.name in names
             and re.fullmatch(r'\d{4}-\d{2}-\d{2}\.npz', p.name)}
    if len(roots) > 1:
        raise ValueError('ZIP에 캐시 루트가 여러 개입니다. 필요한 폴더를 CACHE_DIR로 지정하고 SAVED_CACHE_ZIP=None으로 설정해 주세요.')
    if not roots and CACHE_MODE == 'reuse':
        raise ValueError('캐시의 <패널>/fit|early|blend/YYYY-MM-DD.npz 구조를 찾지 못했습니다. 다른 모델의 NPZ는 사용할 수 없습니다.')
    return next(iter(roots)) if roots else destination, destination


def source_signature(paths):
    records = [(Path(p).relative_to(RELEASE_ROOT).as_posix(), file_digest(p)) for p in sorted(set(paths))]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def profile_source_signature(panel, fit_dates, files):
    network = RELEASE_ROOT / 'corridors' / panel.name / 'network'
    return source_signature([files[d] for d in fit_dates] + list(network.glob('*.csv')))


def day_source_signature(context, date):
    paths = [context['files'][date], context['masked'][date]]
    paths += ramp_files(context['release'], context['panel'], 'train', date)
    digest = hashlib.sha256(source_signature(paths).encode())
    digest.update(np.sort(context['targets'].archive[date]).astype('<u4').tobytes())
    digest.update(context['targets'].regimes[date].encode())
    return digest.hexdigest()


def cache_is_current(path, source):
    marker = path.with_suffix('.sources.json')
    return marker.is_file() and json.loads(marker.read_text(encoding='utf-8')) == source


def write_cache_source(path, source):
    marker = path.with_suffix('.sources.json')
    temporary = marker.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(source), encoding='utf-8')
    temporary.replace(marker)


def load_fit_profile(panel, targets, fit_dates, files):
    metadata = dict(format='maginet-input-v1', panel=panel.name, links=panel.links,
                    stations=targets.stations, station_links=targets.links,
                    fit_dates=fit_dates, aux_names=list(AUX_NAMES), scale=SCALE.tolist())
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    folder = CACHE_METADATA_DIR / 'profiles' / panel.name / key
    folder.mkdir(parents=True, exist_ok=True)
    profile_path, metadata_path = folder / 'fit_profile.npy', folder / 'cache_metadata.json'
    source = profile_source_signature(panel, fit_dates, files)
    legacy = CACHE_ROOT / panel.name
    legacy_metadata = legacy / 'cache_metadata.json'
    if legacy_metadata.exists() and json.loads(legacy_metadata.read_text()) != metadata:
        raise ValueError(f'이전 캐시의 학습 날짜·관측소·입력 형식이 다릅니다: {panel.name}')
    if not metadata_path.exists() and legacy_metadata.exists():
        legacy_profile = legacy / 'fit_profile.npy'
        if not legacy_profile.is_file():
            raise FileNotFoundError(f'이전 캐시 과거 평균이 누락되었습니다: {legacy_profile}')
        profile = np.load(legacy_profile, allow_pickle=False)
        if profile.shape != (2016, len(panel.links), 3) or not np.isfinite(profile).all():
            raise ValueError(f'이전 캐시 과거 평균의 형태·값이 잘못되었습니다: {panel.name}')
        temporary = profile_path.with_suffix('.npy.tmp')
        shutil.copyfile(legacy_profile, temporary)
        temporary.replace(profile_path)
        metadata_temporary = metadata_path.with_suffix('.json.tmp')
        metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        metadata_temporary.replace(metadata_path)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError(f'저장 과거 평균의 학습 날짜·관측소·입력 형식이 다릅니다: {metadata_path}')
    if profile_path.is_file() and metadata_path.is_file() and cache_is_current(profile_path, source):
        profile = np.load(profile_path, allow_pickle=False)
    else:
        if CACHE_MODE == 'reuse':
            raise ValueError(f'{panel.name}: 과거 평균의 원본이 변경됐거나 출처 기록이 없습니다. --cache-mode build로 한 번 실행해 주세요.')
        profile = fit_prior(panel, [files[d] for d in fit_dates])
        candidates = [CACHE_PROFILE_DIR / f'fit_profile-{panel.name}.npy']
        candidates += sorted(CACHE_CONTAINER.rglob(f'fit_profile-{panel.name}.npy'))
        for candidate in dict.fromkeys(candidates):
            config_path = candidate.parent / 'config.json'
            if not candidate.is_file() or not config_path.is_file():
                continue
            saved = json.loads(config_path.read_text(encoding='utf-8')).get('panels', {}).get(panel.name, {})
            legacy_profile = np.load(candidate, allow_pickle=False)
            if (saved.get('stations') != targets.stations or saved.get('links') != targets.links
                    or saved.get('dates', {}).get('fit') != fit_dates
                    or hashlib.sha256(legacy_profile.tobytes()).hexdigest() != saved.get('profile_hash')
                    or not np.array_equal(legacy_profile, profile)):
                raise ValueError(f'이전 과거 평균이 현재 원본·관측소·fit 날짜와 다릅니다: {candidate}')
            break
        if profile.shape != (2016, len(panel.links), 3) or not np.isfinite(profile).all():
            raise ValueError(f'저장 과거 평균의 형태·값이 잘못되었습니다: {panel.name}')
        if not profile_path.is_file() or not np.array_equal(np.load(profile_path, allow_pickle=False), profile):
            temporary = profile_path.with_suffix('.npy.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, profile, allow_pickle=False)
            temporary.replace(profile_path)
        metadata_temporary = metadata_path.with_suffix('.json.tmp')
        metadata_temporary.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
        metadata_temporary.replace(metadata_path)
        write_cache_source(profile_path, source)
    if profile.shape != (2016, len(panel.links), 3) or not np.isfinite(profile).all():
        raise ValueError(f'캐시 과거 평균의 형태·값이 잘못되었습니다: {panel.name}')
    return profile.astype(np.float32, copy=False)


def validate_cache_targets(data, context, date):
    required = {'values', 'visible', 'aux', 'slot', 'weekday', 'base', 'truth', 'target',
                'active', 'observed', 'timestamps', 'ti', 'si', 'regime', 'ramp_valid', 'consecutive'}
    if required - set(data):
        raise ValueError(f'이 MagiNet 노트북의 NPZ 형식이 아닙니다: {date}, 누락 {sorted(required - set(data))}')
    target, timestamps = data['target'], data['timestamps']
    stations = len(context['targets'].stations)
    if (target.dtype != np.bool_ or target.ndim != 2 or target.shape != (len(timestamps), stations)
            or len(set(timestamps.tolist())) != len(timestamps)
            or any(str(t)[:10] != date for t in timestamps)):
        raise ValueError(f'캐시 날짜·관측소 형태가 다릅니다: {date}')
    ti, si = np.nonzero(target)
    cells = np.array([int(str(t)[11:13]) * 12 + int(str(t)[14:16]) // 5
                      for t in timestamps], np.int64)[ti] * stations + si
    expected = context['targets'].archive[date]
    if (not np.array_equal(np.sort(cells), np.sort(expected))
            or not np.array_equal(ti, data['ti']) or not np.array_equal(si, data['si'])
            or int(data['regime']) != REGIMES.index(context['targets'].regimes[date])):
        raise ValueError(f'캐시 대상 셀·regime이 템플릿과 다릅니다: {date}')
    if 'profile' in context:
        slot = data['weekday'] * 288 + data['slot']
        prior = context['profile'][slot[:, None], context['mapping'][None, :]].clip(-10, 10)
        if not np.allclose(data['aux'][..., 8:11], prior, atol=1e-6, rtol=1e-6):
            raise ValueError(f'캐시의 fit 과거 평균 특징이 현재 학습 구간과 다릅니다: {date}')
    return int(data['regime']), len(ti)


def fit_input_normalization(entries, cache_folder, mode=None, legacy_folder=None):
    mode = INPUT_NORMALIZATION if mode is None else mode
    fit = sorted((e for e in entries if e['period'] == 'fit'), key=lambda e: (e['panel'], e['date']))
    if not fit or mode not in ('fixed', 'standard'):
        raise ValueError('표준화에는 fit 캐시와 fixed/standard 모드가 필요합니다.')
    provenance = []
    for entry in fit:
        # NPZ의 배열 CRC는 ZIP 복원·경로 이동·수정 시각 변경 후에도 같습니다.
        with zipfile.ZipFile(entry['path']) as archive:
            arrays = [(name, archive.getinfo(name).file_size, archive.getinfo(name).CRC)
                      for name in ('values.npy', 'visible.npy')]
        provenance.append((entry['panel'], entry['date'], arrays))
    key = hashlib.sha256(json.dumps(dict(format='fit-visible-crc-v1', mode=mode, files=provenance),
                                   sort_keys=True).encode()).hexdigest()
    path = Path(cache_folder) / f'input_normalization-{key[:16]}.json'
    sources = ([Path(legacy_folder) / path.name] if legacy_folder is not None else [])
    sources.append(REPORT_BASE / 'full' / path.name)
    for source in sources:
        if path.exists():
            break
        if not source.is_file():
            continue
        saved = json.loads(source.read_text())
        if saved.get('source_signature') != key or saved.get('mode') != mode:
            raise ValueError('이전 실행의 입력 표준화 설정이 다릅니다.')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
        temporary.replace(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('source_signature') != key or saved.get('mode') != mode:
            raise ValueError('저장된 입력 표준화 설정이 다릅니다.')
        return saved
    count, total, square = np.zeros(3, np.int64), np.zeros(3, np.float64), np.zeros(3, np.float64)
    if mode == 'standard':
        for number, entry in enumerate(fit, 1):
            with np.load(entry['path'], allow_pickle=False) as data:
                values, visible = data['values'], data['visible']
                if (values.shape != visible.shape or visible.dtype != np.bool_
                        or values.ndim != 3 or values.shape[-1] != 3):
                    raise ValueError(f'표준화 입력 형태가 다릅니다: {entry["path"]}')
                if not np.isfinite(values[visible]).all():
                    raise ValueError(f'관측된 입력에 NaN/Inf가 있습니다: {entry["path"]}')
                observed = np.where(visible, values, 0).astype(np.float64)
                count += visible.sum((0, 1))
                total += observed.sum((0, 1))
                square += np.square(observed).sum((0, 1))
            if number == 1 or number % 100 == 0 or number == len(fit):
                print(f'fit 관측 입력 표준화: {number}/{len(fit)}일', flush=True)
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(square / np.maximum(count, 1) - mean**2, 0))
    std = np.where(std > 1e-6, std, 1.)
    saved = dict(mode=mode, mean=mean.tolist(), std=std.tolist(), count=count.tolist(),
                 source_signature=key, fitted_period='fit', channels=['speed/25', 'flow_per_lane/600', 'occupancy'])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(saved, indent=2, ensure_ascii=False))
    temporary.replace(path)
    return saved


def attach_vocabulary(contexts):
    families = sorted({n.rsplit('_', 1)[0] for n in contexts})
    for name, c in contexts.items():
        family, direction = name.rsplit('_', 1)
        if direction not in ('N', 'S', 'E', 'W'):
            raise ValueError(f'알 수 없는 방향: {name}')
        c['family_code'] = families.index(family)
        c['direction_code'] = ('N', 'S', 'E', 'W').index(direction)


def chebyshev_supports(adjacency, order=None):
    order = CHEB_K if order is None else order
    adjacency = np.asarray(adjacency, np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1] or not np.allclose(adjacency, adjacency.T):
        raise ValueError('Chebyshev 그래프는 대칭 인접행렬이어야 합니다.')
    degree = adjacency.sum(-1)
    inv = 1 / np.sqrt(np.maximum(degree, 1))
    identity = np.eye(len(adjacency))
    laplacian = identity - inv[:, None] * adjacency * inv[None, :]
    largest = float(np.linalg.eigvalsh(laplacian).max())
    scaled = 2 * laplacian / largest - identity if largest > 1e-10 else -identity
    supports = [identity]
    if order > 1:
        supports.append(scaled)
    for _ in range(2, order):
        supports.append(2 * scaled @ supports[-1] - supports[-2])
    return np.stack(supports).astype(np.float32)


def input_arrays(day, context, split='train'):
    panel, mapping = context['panel'], day.station_to_link
    values = np.stack((day.speed, day.flow_lane, day.occupancy), -1)
    visible = np.isfinite(values)
    dates = pd.to_datetime(day.timestamps, utc=True, format='ISO8601')
    elapsed = np.asarray((dates - dates[0]).total_seconds() / 60, np.float32)
    if len(elapsed) < 2 or (np.diff(elapsed) <= 0).any():
        raise ValueError('하루 시각은 서로 다른 오름차순이어야 합니다.')
    slot, weekday = np.asarray(dates.hour * 12 + dates.minute // 5), np.asarray(dates.dayofweek)
    prior = context['profile'][(weekday * 288 + slot)[:, None], mapping[None, :]]
    baseline, ages = [], []
    for channel, data in enumerate((day.speed, day.flow_lane)):
        base, left_age, right_age = nearest_features(data, prior[..., channel], elapsed)
        baseline.append(base)
        ages += [np.nan_to_num(age / 360, nan=4.).clip(0, 4) for age in (left_age, right_age)]
    ramps = ramp_features(context['release'], panel, split, day.timestamps)
    r = ramps[:, mapping]
    attached = np.zeros((len(panel.links), 2), np.float32)
    for link, kind in panel.ramp_map.values():
        attached[link, kind] = 1
    spread = lambda v: np.broadcast_to(v, day.speed.shape)
    aux = np.stack((day.quality, visible[..., 0], visible[..., 1], visible[..., 2], *ages,
                    prior[..., 0], prior[..., 1], prior[..., 2], spread(day.lanes / 5),
                    spread(panel.free_speed[mapping] / 25), spread(panel.capacity[mapping] / panel.lanes[mapping] / 600),
                    spread(panel.lengths[mapping]), r[..., 0] / 600, r[..., 1] / 600, r[..., 2], r[..., 3],
                    spread(attached[mapping, 0]), spread(attached[mapping, 1])), -1)
    if aux.shape[-1] != len(AUX_NAMES) or np.isinf(values).any():
        raise ValueError('입력 배열이 잘못되었습니다.')
    # 수치상 placeholder 0은 mask로 제외되며 관측값 임베딩으로 사용되지 않습니다.
    values = np.where(visible, values, 0).astype(np.float32)
    aux = np.nan_to_num(aux, nan=0., posinf=10., neginf=-10.).clip(-10, 10).astype(np.float32)
    base = np.stack(baseline, -1).astype(np.float32)
    if not np.isfinite(base).all():
        raise ValueError('기준 보간이 유한하지 않습니다.')
    return dict(values=values, visible=visible, aux=aux, slot=slot.astype(np.int64),
                weekday=weekday.astype(np.int64), base=base,
                ramp_valid=(ramps[..., 2] > 0) & (ramps[..., 3] > 0))


def consecutive_five_minutes(timestamps):
    times = pd.to_datetime(timestamps, utc=True, format='ISO8601')
    return np.asarray((times[1:] - times[:-1]) == pd.Timedelta(minutes=5))


def make_day(context, date, split='train', target_index=None, input_files=None):
    panel = context['panel']
    path = context['masked'][date] if split == 'train' else input_files[date]
    day = StationDay.load(path, panel)
    if split == 'train':
        truth = StationDay.load(context['files'][date], panel)
        if (day.timestamps, day.station_ids, day.station_links) != (truth.timestamps, truth.station_ids, truth.station_links):
            raise ValueError(f'정답과 masked 입력이 다릅니다: {date}')
        regime, target = context['targets'].get(truth)
    else:
        truth = None
        regime, target = target_index.get(day, require_eligible=False)
    if (day.station_ids, day.station_links) != (context['targets'].stations, context['targets'].links):
        raise ValueError(f'관측소 구성이 다릅니다: {date}')
    path_regimes = [p.name.split('=', 1)[1] for p in path.parents if p.name.startswith('mask_regime=')]
    if path_regimes and path_regimes[0] != regime:
        raise ValueError(f'입력과 템플릿의 regime이 다릅니다: {date}')
    if np.isfinite(day.speed[target]).any() or np.isfinite(day.flow_lane[target]).any():
        raise ValueError(f'대상 정답이 입력에 보입니다: {date}')
    d = input_arrays(day, context, split)
    observed_values = np.stack((day.speed * 25, day.flow_lane * 600 * day.lanes), -1)
    observed = np.isfinite(observed_values).all(-1)
    active = (observed & (truth.eligible if truth is not None else day.eligible)) | target
    labels = np.stack((truth.speed, truth.flow_lane), -1) if truth is not None else np.zeros((*day.speed.shape, 2), np.float32)
    ti, si = np.nonzero(target)
    d.update(truth=np.nan_to_num(labels), target=target, active=active,
             observed=np.nan_to_num(observed_values), timestamps=np.asarray(day.timestamps), ti=ti, si=si,
             regime=np.array(REGIMES.index(regime), np.int32),
             consecutive=consecutive_five_minutes(day.timestamps))
    return d


def validate_resampling_occupancy(data):
    if not data['visible'][data['target'], 2].all():
        raise ValueError('mask_resample 중단: 기존 대상의 occupancy 원본이 NPZ에 없어 결측을 복원할 수 없습니다. 다른 EXPERIMENT를 사용하거나 원본 fit occupancy를 보관하는 전처리를 별도로 구현해 주세요.')


def cache_context(context, cache_root, date_overrides=None, periods=('fit', 'early', 'blend')):
    started, entries = time.perf_counter(), []
    for period in periods:
        dates = context['dates'][period] if date_overrides is None else date_overrides[period]
        folder = cache_root / context['panel'].name / period
        folder.mkdir(parents=True, exist_ok=True)
        for number, date in enumerate(dates, 1):
            path = folder / f'{date}.npz'
            source = day_source_signature(context, date)
            if not path.exists() or not cache_is_current(path, source):
                if CACHE_MODE == 'reuse':
                    if path.exists():
                        raise ValueError(f'캐시 원본이 변경됐거나 출처 기록이 없습니다: {path}. --cache-mode build로 실행해 주세요.')
                    raise FileNotFoundError(f'저장 캐시가 누락되었습니다: {path}. 원본에서 생성하려면 CACHE_MODE="build"로 변경해 주세요.')
                if shutil.disk_usage(cache_root).free < 2 * 2**30:
                    raise RuntimeError('임시 캐시 디스크 여유 공간이 2 GiB 미만입니다.')
                d = make_day(context, date)
                if TRAIN_MODEL and MASK_RESAMPLE and period == 'fit':
                    validate_resampling_occupancy(d)
                regime, targets = validate_cache_targets(d, context, date)
                same = False
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        same = set(saved.files) == set(d) and all(np.array_equal(saved[k], v) for k, v in d.items())
                if not same:
                    temporary = path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, **d)
                    temporary.replace(path)
                write_cache_source(path, source)
                load_day.cache_clear()
            else:
                with np.load(path, allow_pickle=False) as d:
                    regime, targets = validate_cache_targets(d, context, date)
                    if TRAIN_MODEL and MASK_RESAMPLE and period == 'fit':
                        validate_resampling_occupancy(d)
            entries.append(dict(panel=context['panel'].name, date=date, period=period, regime=regime,
                                targets=targets, path=str(path)))
            if number == 1 or number % 20 == 0 or number == len(dates):
                print(f"입력 캐시 {context['panel'].name}/{period}: {number}/{len(dates)}일", flush=True)
    context['cache_seconds'] = context.get('cache_seconds', 0.) + time.perf_counter() - started
    return entries


@lru_cache(maxsize=2)
def load_day(path):
    with np.load(path, allow_pickle=False) as d:
        result = {k: d[k] for k in d.files}
    values, visible, base, aux = (result[k] for k in ('values', 'visible', 'base', 'aux'))
    if (values.ndim != 3 or values.shape[-1] != 3 or visible.dtype != np.bool_
            or visible.shape != values.shape or base.shape != values.shape[:2] + (2,)
            or aux.shape != values.shape[:2] + (len(AUX_NAMES),)
            or not np.isfinite(values[visible]).all() or not np.isfinite(base).all()
            or not np.isfinite(aux).all() or (base < 0).any()
            or visible[result['target'], :2].any()):
        raise ValueError(f'캐시 배열 형태·관측 마스크·보간값이 잘못되었습니다: {path}')
    return result


def resample_fit_mask(d, context, rng, period):
    if period != 'fit':
        raise ValueError('결측 마스크 재표본화는 fit 학습에서만 사용할 수 있습니다.')
    validate_resampling_occupancy(d)
    eligible = d['active'].astype(bool)
    if not eligible.any() or d['truth'].shape != (*eligible.shape, 2) or not np.isfinite(d['truth'][eligible]).all():
        raise ValueError('마스크 재표본화에 필요한 fit 정답이 잘못되었습니다.')
    # 같은 링크의 관측소를 함께 가리고, 기존 자연 결측·Task 2 문맥 결측은 유지합니다.
    # shortcut: NPZ에 Task 2 history 제외 플래그가 없어 fit history 관측도 재마스킹합니다; 정확한 분포 비교 시 플래그를 캐시에 추가하세요.
    rate = (0.20, 0.30, 0.50)[int(d['regime'])]
    draws = rng.random((len(eligible), len(context['panel'].links)))
    target = (draws[:, context['mapping']] < rate) & eligible
    if not target.any():
        first_time, first_station = (indices[0] for indices in np.nonzero(eligible))
        target[first_time] = (context['mapping'] == context['mapping'][first_station]) & eligible[first_time]
    values, visible, aux = (d[key].copy() for key in ('values', 'visible', 'aux'))
    values[eligible, :2] = d['truth'][eligible]
    visible[eligible, :2] = True
    values[target, :2], visible[target, :2] = 0., False
    aux[..., 1:4] = visible
    dates = pd.to_datetime(d['timestamps'], utc=True, format='ISO8601')
    elapsed = np.asarray((dates - dates[0]).total_seconds() / 60, np.float32)
    baseline = []
    for channel in range(2):
        observed = np.where(visible[..., channel], values[..., channel], np.nan)
        base, left_age, right_age = nearest_features(observed, aux[..., 8+channel], elapsed)
        baseline.append(base)
        for offset, age in enumerate((left_age, right_age)):
            aux[..., 4 + channel * 2 + offset] = np.nan_to_num(age / 360, nan=4.).clip(0, 4)
    physical = d['observed'].copy()
    units = np.stack((np.full_like(context['lanes'], 25.), 600 * context['lanes']), -1)
    physical[eligible] = (d['truth'] * units)[eligible]
    physical[target] = 0.
    ti, si = np.nonzero(target)
    result = {key: value for key, value in d.items() if key != '_transition'}
    result.update(values=values, visible=visible, aux=aux, base=np.stack(baseline, -1).astype(np.float32),
                  target=target, observed=physical, ti=ti, si=si)
    if visible[target, :2].any() or not np.isfinite(result['base']).all():
        raise ValueError('재표본화한 마스크에 정답 노출 또는 잘못된 보간값이 있습니다.')
    return result


def window_starts(length):
    if length <= WINDOW:
        return [0]
    return sorted(set(range(0, length - WINDOW + 1, STRIDE)) | {length - WINDOW})

def masked_softmax(scores, valid):
    valid = valid.to(dtype=torch.bool)
    weights = scores.float().masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
    weights = torch.where(valid, weights, 0.)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class AMSTEncoder(nn.Module):
    def __init__(self, station_count, panel_count, family_count):
        super().__init__()
        self.value_weight = nn.Parameter(torch.empty(3, WIDTH))
        self.value_bias = nn.Parameter(torch.zeros(3, WIDTH))
        self.missing_token = nn.Parameter(torch.empty(3, WIDTH))
        self.slot = nn.Embedding(288, WIDTH)
        self.weekday = nn.Embedding(7, WIDTH)
        self.relative = nn.Embedding(288, WIDTH)
        self.station = nn.Embedding(station_count, WIDTH)
        self.panel = nn.Embedding(panel_count, WIDTH)
        self.family = nn.Embedding(family_count, WIDTH)
        self.direction = nn.Embedding(4, WIDTH)
        self.regime = nn.Embedding(3, WIDTH)
        self.aux = nn.Linear(len(AUX_NAMES), WIDTH)
        self.base_embedding = nn.Linear(2, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(DROPOUT)
        nn.init.xavier_uniform_(self.value_weight)
        nn.init.normal_(self.missing_token, std=0.02)

    def forward(self, values, visible, aux, base, slot, weekday, regime, ct):
        # values/visible: B×T×station×3; 결측 placeholder는 여기서 제외합니다.
        tokens = values[..., None] * self.value_weight + self.value_bias
        tokens = torch.where(visible[..., None], tokens, self.missing_token)
        h = tokens.sum(-2) / np.sqrt(3.)
        t = len(values[0])
        time_embedding = self.slot(slot) + self.weekday(weekday) + self.relative(torch.arange(t, device=values.device))[None]
        node_embedding = (self.station(ct['station_codes']) + self.panel(ct['panel_code'])
                          + self.family(ct['family_code']) + self.direction(ct['direction_code']))
        h = h + self.aux(aux) + self.base_embedding(base) + time_embedding[:, :, None] + node_embedding[None, None] + self.regime(regime)[:, None, None]
        return self.dropout(self.norm(h)).permute(0, 2, 1, 3), node_embedding


class MaskTemporalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.project = nn.Linear(WIDTH, WIDTH)
        self.norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, h, visible, residual_attention):
        b, s, t, d = h.shape
        q, k, v = self.qkv(h).reshape(b, s, t, 3, HEADS, d // HEADS).permute(3, 0, 1, 4, 2, 5).unbind(0)
        scores = (q.float() @ k.float().transpose(-1, -2)) / np.sqrt(d // HEADS)
        if residual_attention is not None:
            scores = scores + residual_attention
        key_valid = visible.any(-1).permute(0, 2, 1)[:, :, None, None, :]
        weights = masked_softmax(scores, key_valid)
        attended = (weights.to(v.dtype) @ v).transpose(2, 3).reshape(b, s, t, d)
        return self.norm(h + self.dropout(self.project(attended))), scores


class MaskSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = nn.Linear(2 * WIDTH, GRAPH_DIM)
        self.position = nn.Linear(WIDTH, GRAPH_DIM)
        self.qk = nn.Linear(GRAPH_DIM, 2 * CHEB_K * 16, bias=False)
        self.mask_pair = nn.Linear(GRAPH_DIM, 2 * CHEB_K * 8, bias=False)
        self.norm = nn.LayerNorm(GRAPH_DIM)

    def forward(self, h, visible, node_embedding):
        b, s, t, d = h.shape
        valid = visible.any(-1).permute(0, 2, 1).float()
        observed_summary = (h * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]
        summary = self.summary(torch.cat((observed_summary, h.mean(2)), -1))
        # MagiNet의 관측 문맥 기반 missing spatial graph를 유지합니다.
        similarity = F.relu(summary.float() @ summary.float().transpose(-1, -2))
        missing_graph = (similarity.softmax(-1) > MGL_THRESHOLD).float()
        positioned = self.norm(summary + self.position(node_embedding)[None])
        q, k = self.qk(positioned).reshape(b, s, 2, CHEB_K, 16).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q.float() @ k.float().transpose(-1, -2)) / 4.
        a, z = self.mask_pair(positioned).reshape(b, s, 2, CHEB_K, 8).permute(2, 0, 3, 1, 4).unbind(0)
        pair_mask = torch.tanh((a.float() @ z.float().transpose(-1, -2)) / np.sqrt(8.))
        reliability = 0.25 + 0.75 * valid.mean(2)
        scores = scores + missing_graph[:, None] * pair_mask + reliability.log()[:, None, None, :]
        return scores.softmax(-1)


class MaskAwareChebConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(CHEB_K, WIDTH, WIDTH))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, h, spatial_attention, supports):
        weights = spatial_attention * supports[None]
        output = torch.einsum('bkij,bjtd,kdf->bitf', weights.to(h.dtype), h, self.theta.to(h.dtype))
        return F.relu(output)


def observation_context(visible, aux, neighbors):
    """Public observations only; physical neighbors exclude the station itself."""
    with torch.autocast(device_type=visible.device.type, enabled=False):
        joint = visible[..., :2].all(-1).float()
        counts = torch.einsum('ij,btj->bti', neighbors.float(), joint)
        degree = neighbors.float().sum(-1)[None, None]
        features = torch.cat((visible.float(), (counts / (1 + counts))[..., None],
                              (counts / degree.clamp_min(1))[..., None],
                              (degree > 0).expand_as(counts)[..., None].float(),
                              aux[..., 4:8].float().clamp(0, 4) / 4), -1)
    return features.permute(0, 2, 1, 3)


def gather_visible_spatial(h, spatial_attention, visible):
    with torch.autocast(device_type=h.device.type, enabled=False):
        weights = spatial_attention.float().mean(1)
        valid = visible.any(-1).permute(0, 2, 1).float()
        numerator = torch.einsum('bij,bjtd->bitd', weights, h.float() * valid[..., None])
        denominator = torch.einsum('bij,bjt->bit', weights, valid)
        return (numerator / denominator.clamp_min(1e-8)[..., None]).to(h.dtype)


class LearnedSpatialConv(nn.Module):
    def __init__(self, contextual=False):
        super().__init__()
        self.project = nn.Linear(WIDTH, WIDTH, bias=False)
        self.gate = nn.Parameter(torch.tensor(-3.))
        self.context_gate = nn.Linear(10, 1, bias=False) if contextual else None
        if self.context_gate is not None:
            nn.init.zeros_(self.context_gate.weight)

    def forward(self, h, spatial_attention, visible, context=None):
        # 관측 문맥으로 학습한 관계를 재사용하며 물리 supports를 곱하지 않습니다.
        gathered = gather_visible_spatial(h, spatial_attention, visible)
        gate = self.gate if self.context_gate is None else self.gate + self.context_gate(context).float()
        return gate.sigmoid() * self.project(gathered)


class ChannelSpatialReadout(nn.Module):
    """Channel-specific spatial residuals; the shared backbone has no extra learned graph."""
    def __init__(self):
        super().__init__()
        self.attention = MaskSpatialAttention()
        self.project = nn.Linear(WIDTH, WIDTH, bias=False)
        self.correction = nn.Linear(WIDTH, 2)
        self.gate = nn.Linear(10, 2)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.)

    def forward(self, h, visible, node_embedding, context):
        attention = self.attention(h, visible, node_embedding)
        gathered = gather_visible_spatial(h, attention, visible)
        correction = self.correction(F.gelu(self.project(gathered))).float()
        # 관측소 전체가 비어 있는 시각에는 bias만으로 공간 보정을 만들지 않습니다.
        available = visible.any(-1).any(-1)[:, None, :, None]
        return correction * self.gate(context).float().sigmoid() * available


class GatedTemporalUnit(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.conv = nn.Conv2d(WIDTH, 2 * WIDTH, kernel_size=(1, kernel), padding=(0, kernel // 2))

    def forward(self, h):
        a, b = self.conv(h).chunk(2, dim=1)
        return a.tanh() * b.sigmoid()


class MASTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_attention = MaskTemporalAttention()
        self.spatial_attention = MaskSpatialAttention()
        self.spatial_conv = MaskAwareChebConv()
        self.temporal_direct = TEMPORAL_DIRECT
        self.learned_spatial = (LearnedSpatialConv(contextual=SPATIAL_MODE == 'context')
                                if LEARNED_GRAPH and SPATIAL_MODE != 'channel' else None)
        self.gates = nn.ModuleList(GatedTemporalUnit(k) for k in (3, 5, 7))
        self.time_mix = nn.Conv2d(3 * WIDTH, WIDTH, kernel_size=1)
        self.norm = nn.LayerNorm(WIDTH)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, h, visible, node_embedding, supports, residual_attention, context=None):
        attended, residual_attention = self.temporal_attention(h, visible, residual_attention)
        spatial_attention = self.spatial_attention(attended, visible, node_embedding)
        signal = attended if self.temporal_direct else h
        spatial = self.spatial_conv(signal, spatial_attention, supports)
        if self.learned_spatial is not None:
            spatial = spatial + self.learned_spatial(signal, spatial_attention, visible, context)
        channels = spatial.permute(0, 3, 1, 2)
        gated = self.time_mix(torch.cat([gate(channels) for gate in self.gates], dim=1)).permute(0, 2, 3, 1)
        output = self.norm(h + self.dropout(F.relu(spatial + gated)))
        return output, residual_attention


class MagiNetTask13(nn.Module):
    def __init__(self, station_count, panel_count, family_count, normalization):
        super().__init__()
        mean = torch.as_tensor(normalization['mean'], dtype=torch.float32)
        std = torch.as_tensor(normalization['std'], dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError('입력 표준화의 평균·표준편차가 잘못되었습니다.')
        self.register_buffer('input_mean', mean)
        self.register_buffer('input_std', std)
        self.AMSTenc = AMSTEncoder(station_count, panel_count, family_count)
        self.MASTdec = nn.ModuleList(MASTBlock() for _ in range(BLOCKS))
        self.fuse = nn.Linear((BLOCKS + 1) * WIDTH, WIDTH)
        self.readout = nn.GRU(WIDTH, WIDTH // 2, batch_first=True, bidirectional=True)
        self.speed_head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))
        self.flow_head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))
        for head in (self.speed_head, self.flow_head):
            # 보정량은 양·음 모두 허용하며 시작 출력은 기존 보간과 같습니다.
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.spatial_mode = SPATIAL_MODE
        self.channel_spatial = ChannelSpatialReadout() if SPATIAL_MODE == 'channel' else None

    def forward(self, values, visible, aux, base, slot, weekday, regime, ct):
        normalized = torch.where(visible, (values.float() - self.input_mean) / self.input_std, 0.)
        normalized_base = (base.float() - self.input_mean[:2]) / self.input_std[:2]
        normalized_aux = aux.float().clone()
        normalized_aux[..., 8:11] = (normalized_aux[..., 8:11] - self.input_mean) / self.input_std
        h, node_embedding = self.AMSTenc(normalized, visible, normalized_aux, normalized_base, slot, weekday, regime, ct)
        context = observation_context(visible, aux, ct['neighbors']) if self.spatial_mode != 'legacy' else None
        outputs, residual_attention = [h], None
        for block in self.MASTdec:
            h, residual_attention = block(h, visible, node_embedding, ct['supports'], residual_attention, context)
            outputs.append(h)
        h = F.gelu(self.fuse(torch.cat(outputs, dim=-1)))
        b, s, t, d = h.shape
        # cuDNN의 BF16 RNN 지원 여부와 무관하게 시간 readout은 FP32로 계산합니다.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, _ = self.readout(h.float().reshape(b * s, t, d))
        h = h.reshape(b, s, t, d).permute(0, 2, 1, 3)
        correction = torch.cat((self.speed_head(h), self.flow_head(h)), -1).float()
        if self.channel_spatial is not None:
            correction = correction + self.channel_spatial(h.permute(0, 2, 1, 3), visible,
                                                           node_embedding, context).permute(0, 2, 1, 3)
        # 입력 표준화와 별개로 출력·State 손실은 /25·/600 단위를 유지합니다.
        return (base.float() + correction).clamp_min(0.)


def tensor_context(context, device=None):
    device = DEVICE if device is None else device
    panel = context['panel']
    mapping = torch.as_tensor(context['mapping'], dtype=torch.long, device=device)
    neighbors = np.asarray(context['station_graph'], dtype=np.float32).copy()
    np.fill_diagonal(neighbors, 0)
    return dict(station_codes=torch.as_tensor(context['station_codes'], dtype=torch.long, device=device),
                panel_code=torch.tensor(context['panel_code'], dtype=torch.long, device=device),
                family_code=torch.tensor(context['family_code'], dtype=torch.long, device=device),
                direction_code=torch.tensor(context['direction_code'], dtype=torch.long, device=device),
                supports=torch.as_tensor(chebyshev_supports(context['station_graph']), device=device),
                neighbors=torch.as_tensor(neighbors, device=device),
                matrix=F.one_hot(mapping, num_classes=len(panel.links)).float(),
                lanes=torch.as_tensor(context['lanes'], device=device),
                link_lanes=torch.as_tensor(panel.lanes, device=device),
                lengths=torch.as_tensor(panel.lengths, device=device), free=torch.as_tensor(panel.free_speed, device=device),
                capacity=torch.as_tensor(panel.capacity, device=device), jam=torch.as_tensor(panel.jam_density, device=device))


def window_batch(d, starts, device=None):
    device = DEVICE if device is None else device
    lengths = [min(WINDOW, len(d['values']) - start) for start in starts]
    if len(set(lengths)) != 1:
        raise ValueError('한 batch의 시간 길이가 다릅니다.')
    return {key: torch.as_tensor(np.stack([d[key][start:start+length] for start, length in zip(starts, lengths)]),
                                 device=device, dtype=dtype)
            for key, dtype in (('values', torch.float32), ('visible', torch.bool), ('aux', torch.float32),
                                ('base', torch.float32), ('slot', torch.long), ('weekday', torch.long))} | {
               'regime': torch.full((len(starts),), int(d['regime']), dtype=torch.long, device=device)}


def panel_batch_size(context):
    name = context['panel'].name
    if name in PANEL_BATCH:
        return PANEL_BATCH[name]
    if BATCH_POLICY == 'adaptive':
        return 8 if len(context['mapping']) <= 128 else 4
    return BATCH_SIZE


@torch.inference_mode()
def maginet_predict(model, d, ct, batch_size=None):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    model.eval()
    device = next(model.parameters()).device
    sums = np.zeros_like(d['base'], np.float64)
    weights = np.zeros((len(d['base']), 1, 1), np.float64)
    starts = window_starts(len(d['values']))
    for offset in range(0, len(starts), batch_size):
        chunk = starts[offset:offset+batch_size]
        batch = window_batch(d, chunk, device)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=USE_AMP and device.type == 'cuda'):
            outputs = model(**batch, ct=ct).float().cpu().numpy()
        for start, output in zip(chunk, outputs):
            w = np.hanning(len(output) + 2)[1:-1, None, None]
            sums[start:start+len(output)] += output * w
            weights[start:start+len(output)] += w
    if (weights == 0).any():
        raise ValueError('겹치는 창 예측에 누락 시각이 있습니다.')
    result = (sums / weights).astype(np.float32)
    if not np.isfinite(result).all() or (result < 0).any():
        raise FloatingPointError('MagiNet 출력이 잘못되었습니다.')
    return result

def observability_mask(context, counts, ramp_valid, affected, consecutive):
    panel = context['panel']
    present = (counts > 0).any(0)
    topology_nodes = context['topology_nodes']
    observable = present & np.array([i in topology_nodes for i in range(len(panel.links))])
    for a, b in panel.edges.T:
        if a in topology_nodes and not present[a]:
            observable[b] = False
        if b in topology_nodes and not present[b]:
            observable[a] = False
    if context['mode'] == 'C':
        for link, _ in panel.ramp_map.values():
            observable[link] = False
    transition = (counts[:-1] > 0) & (counts[1:] > 0) & observable[None] & consecutive[:, None]
    if context['mode'] in ('A', 'B'):
        transition &= ramp_valid[:-1]
    return transition, (affected[:-1] | affected[1:])

def completed_link_fields(prediction_physical, d, context):
    values = d['observed'].copy()
    ti, si = d['ti'], d['si']
    values[ti, si, 0] = prediction_physical[:, 0]
    values[ti, si, 1] = prediction_physical[:, 1] * context['lanes'][si]
    links, counts = link_mean_numpy(values, d['active'], context['mapping'], len(context['panel'].links))
    affected = link_mean_numpy(d['target'][..., None].astype(float), d['active'],
                              context['mapping'], len(context['panel'].links))[0][..., 0] > 0
    return links, counts, affected

def physics_totals(prediction, d, context):
    panel = context['panel']
    links, counts, affected = completed_link_fields(prediction, d, context)
    truth = d['truth'][d['ti'], d['si']] * SCALE
    reference, _, _ = completed_link_fields(truth, d, context)
    n = links[..., 1] / np.maximum(links[..., 0], 1) * panel.lengths
    true_n = reference[..., 1] / np.maximum(reference[..., 0], 1) * panel.lengths
    good_n = (counts > 0) & affected
    error_n = (n - true_n)[good_n]
    transition, target_transition = observability_mask(context, counts, d['ramp_valid'], affected, d['consecutive'])
    error_dn_all = (np.diff(n, axis=0) - np.diff(true_n, axis=0))[transition]
    truth_dn_all = np.diff(true_n, axis=0)[transition]
    error_dn_target = (np.diff(n, axis=0) - np.diff(true_n, axis=0))[transition & target_transition]
    density = links[..., 1] / np.maximum(links[..., 0], 1)
    critical = panel.capacity / panel.free_speed
    wave = panel.capacity / np.maximum(panel.jam_density - critical, 1e-6)
    qfd = np.where(density <= critical, panel.free_speed * density,
                   wave * (panel.jam_density - density)).clip(0)
    good_fd = counts > 0
    fd_error = ((links[..., 1] - qfd) / panel.lanes)[good_fd]
    q_lane = (links[..., 1] / panel.lanes)[good_fd]
    fraction = link_mean_numpy(d['target'][..., None].astype(float), d['active'],
                               context['mapping'], len(panel.links))[0][..., 0]
    submitted = good_fd & (fraction > 0.5)
    return dict(n_sae=float(np.abs(error_n).sum()), n_sse=float(np.square(error_n).sum()), n_count=len(error_n),
                dn_sae=float(np.abs(error_dn_target).sum()), dn_sse=float(np.square(error_dn_target).sum()),
                dn_count=len(error_dn_target), lwr_num=float(np.abs(error_dn_all).sum()),
                lwr_den=float(np.abs(truth_dn_all).sum()), lwr_count=len(error_dn_all),
                fd_num=float(np.abs(fd_error).sum()), fd_den=float(np.abs(q_lane).sum()),
                fd_count=int(good_fd.sum()), submitted_count=int(submitted.sum()),
                low_flow=int((links[..., 1][submitted] < 50).sum()))

def state_totals(truth, prediction):
    error = np.asarray(prediction, np.float64) - truth
    return dict(count=len(error), speed_sse=float(np.square(error[:, 0]).sum()),
                flow_sse=float(np.square(error[:, 1]).sum()), speed_sae=float(np.abs(error[:, 0]).sum()),
                flow_sae=float(np.abs(error[:, 1]).sum()))

def summarize(records):
    daily = pd.DataFrame(records)
    if daily.empty:
        raise ValueError('평가 대상이 없습니다.')
    keys = ['candidate', 'alpha', 'panel', 'regime']
    sums = daily.groupby(keys, as_index=False).sum(numeric_only=True)
    expected = {(p, r) for p in daily.panel.unique() for r in range(3)}
    for _, part in sums.groupby('candidate'):
        if set(zip(part.panel, part.regime)) != expected or (part['count'] == 0).any():
            raise ValueError('평가의 panel×regime이 누락되었습니다.')
    sums['speed_rmse_kmh'] = np.sqrt(sums.speed_sse / sums['count'])
    sums['flow_rmse_per_lane'] = np.sqrt(sums.flow_sse / sums['count'])
    sums['speed_mae_kmh'] = sums.speed_sae / sums['count']
    sums['flow_mae_per_lane'] = sums.flow_sae / sums['count']
    sums['State'] = 0.54 * np.maximum(0, 1 - sums.speed_rmse_kmh / 25) + 0.46 * np.maximum(0, 1 - sums.flow_rmse_per_lane / 600)
    if 'lwr_num' in sums:
        sums['N_MAE_proxy'] = sums.n_sae / sums.n_count.replace(0, np.nan)
        sums['DN_MAE_proxy'] = sums.dn_sae / sums.dn_count.replace(0, np.nan)
        sums['LWR_proxy'] = np.maximum(0, 1 - sums.lwr_num / (sums.lwr_den + 1e-6))
        sums['FD_proxy'] = np.maximum(0, 1 - sums.fd_num / (sums.fd_den + 1e-6))
        sums.loc[sums.lwr_count == 0, 'LWR_proxy'] = 0.
        sums.loc[sums.fd_count == 0, 'FD_proxy'] = 0.
        sums['low_flow_fraction'] = sums.low_flow / sums.submitted_count.replace(0, np.nan)
        sums.loc[sums.low_flow_fraction > 0.20, 'FD_proxy'] = 0.
        sums['Physics_proxy'] = (sums.FD_proxy + 2 * sums.LWR_proxy) / 3
    metrics = ['State', 'speed_rmse_kmh', 'flow_rmse_per_lane', 'speed_mae_kmh', 'flow_mae_per_lane']
    metrics += [n for n in ('N_MAE_proxy', 'DN_MAE_proxy', 'FD_proxy', 'LWR_proxy', 'Physics_proxy') if n in sums]
    rows = []
    for name, part in sums.groupby('candidate', sort=False):
        per_panel = part.groupby('panel')[metrics].mean()
        row = dict(candidate=name, alpha=float(part.alpha.iloc[0]),
                   **{m: macro_mean(per_panel[m].to_dict()) for m in metrics})
        if 'FD_proxy' in row:
            row['selection_proxy'] = sum(row[k] * w for k, w in SELECT_WEIGHTS.items()) / sum(SELECT_WEIGHTS.values())
        rows.append(row)
    return pd.DataFrame(rows), sums, daily

def training_transitions(d, context):
    if '_transition' not in d:
        _, counts, affected = completed_link_fields(d['truth'][d['ti'], d['si']] * SCALE, d, context)
        transition, target_transition = observability_mask(context, counts, d['ramp_valid'], affected, d['consecutive'])
        d['_transition'] = transition & target_transition
    return d['_transition']


def joint_loss(output, d, starts, context, ct, physics_factor):
    device, length = output.device, output.shape[1]
    take = lambda key: torch.as_tensor(np.stack([d[key][s:s+length] for s in starts]), device=device)
    target, truth = take('target').bool(), take('truth').float()
    if not target.any():
        raise ValueError('학습 창에 대상 셀이 없습니다.')
    state_error = (output.float() - truth)[target]
    state = 0.54 * state_error[:, 0].square().mean() + 0.46 * state_error[:, 1].square().mean()
    zero = state * 0.
    losses = dict(State=state, N=zero, dN=zero, FD=zero)
    if physics_factor > 0:
        active, observed = take('active').float(), take('observed').float()
        units = torch.stack((torch.full_like(ct['lanes'], 25.), 600 * ct['lanes']), -1)
        predicted = torch.where(target[..., None], output.float() * units, observed)
        reference = torch.where(target[..., None], truth * units, observed)
        counts = torch.einsum('bts,sl->btl', active, ct['matrix'])
        aggregate = lambda values: torch.einsum('btsc,sl->btlc', values * active[..., None], ct['matrix']) / counts.clamp_min(1)[..., None]
        links, true_links = aggregate(predicted), aggregate(reference)
        affected = torch.einsum('bts,sl->btl', target.float(), ct['matrix']) > 0
        good_n = (counts > 0) & affected
        n = links[..., 1] / links[..., 0].clamp_min(1) * ct['lengths']
        true_n = true_links[..., 1] / true_links[..., 0].clamp_min(1) * ct['lengths']
        losses['N'] = (n - true_n)[good_n].abs().sum() / (true_n[good_n].abs().sum() + 1.)
        transition = training_transitions(d, context)
        good_dn = torch.as_tensor(np.stack([transition[s:s+length-1] for s in starts]), device=device)
        dn, true_dn = torch.diff(n, dim=1), torch.diff(true_n, dim=1)
        losses['dN'] = (dn - true_dn)[good_dn].abs().sum() / (true_dn[good_dn].abs().sum() + 1.)
        if LOSS_WEIGHTS['FD'] > 0:
            density = links[..., 1] / links[..., 0].clamp_min(1)
            critical = ct['capacity'] / ct['free']
            wave = ct['capacity'] / (ct['jam'] - critical).clamp_min(1e-6)
            qfd = torch.where(density <= critical, ct['free'] * density, wave * (ct['jam'] - density)).clamp_min(0)
            error = ((links[..., 1] - qfd) / ct['link_lanes'])[good_n]
            losses['FD'] = error.abs().sum() / ((links[..., 1] / ct['link_lanes'])[good_n].detach().abs().sum() + 1.)
    total = state + physics_factor * sum(LOSS_WEIGHTS[k] * losses[k] for k in LOSS_WEIGHTS)
    if not torch.isfinite(total):
        raise FloatingPointError('학습 손실에 NaN/Inf가 있습니다.')
    return total, {k: float(v.detach()) for k, v in losses.items()}


def error_slice_axes(d, context):
    ti, si = d['ti'], d['si']
    ages = d['aux'][ti, si, 4:6]
    gap = np.rint(ages.sum(-1) * 360)
    gap_bucket = np.select([np.any(ages >= 4., axis=-1), gap <= 10, gap <= 30, gap <= 60],
                           ['one_sided', '0-10', '10-30', '30-60'], default='60+')
    neighbors = context['station_graph'].astype(np.int32).copy()
    np.fill_diagonal(neighbors, 0)
    counts = (d['visible'][..., :2].all(-1).astype(np.int32) @ neighbors.T)[ti, si]
    neighbor_bucket = np.select([counts == 0, counts <= 2, counts <= 5], ['0', '1-2', '3-5'], default='6+')
    return {'hour_utc': d['slot'][ti] // 12, 'speed_gap_minutes': gap_bucket,
            'station': np.asarray(context['targets'].stations)[si], 'observed_neighbors': neighbor_bucket}


def error_slice_records(truth, prediction, axes, metadata):
    rows = []
    for dimension, buckets in axes.items():
        for bucket in np.unique(buckets):
            selected = buckets == bucket
            rows.append(dict(metadata, dimension=dimension, bucket=str(bucket),
                             **state_totals(truth[selected], prediction[selected])))
    return rows


def summarize_error_slices(records):
    keys = ['candidate', 'alpha', 'panel', 'regime', 'dimension', 'bucket']
    frame = pd.DataFrame(records).groupby(keys, as_index=False).sum(numeric_only=True)
    frame['speed_rmse_kmh'] = np.sqrt(frame.speed_sse / frame['count'])
    frame['flow_rmse_per_lane'] = np.sqrt(frame.flow_sse / frame['count'])
    frame['speed_mae_kmh'] = frame.speed_sae / frame['count']
    frame['flow_mae_per_lane'] = frame.flow_sae / frame['count']
    frame['slice_State'] = (0.54 * np.maximum(0, 1 - frame.speed_rmse_kmh / 25)
                            + 0.46 * np.maximum(0, 1 - frame.flow_rmse_per_lane / 600))
    return frame


def evaluate(entries, contexts, tensors, model, period, alphas, folder, collect_diagnostics=False,
             postprocessors=None):
    started, records, slice_records = time.perf_counter(), [], []
    selected = [e for e in entries if e['period'] == period]
    if not selected:
        raise ValueError(f'평가 날짜가 없습니다: {period}')
    for number, entry in enumerate(selected, 1):
        d, context = load_day(entry['path']), contexts[entry['panel']]
        output = maginet_predict(model, d, tensors[entry['panel']], batch_size=panel_batch_size(context))
        if postprocessors:
            output = postprocessors[entry['panel']](d, output)
        ti, si = d['ti'], d['si']
        truth, base, prediction = d['truth'][ti, si] * SCALE, d['base'][ti, si] * SCALE, output[ti, si] * SCALE
        axes = error_slice_axes(d, context) if collect_diagnostics else None
        for alpha in alphas:
            mixed = (1 - alpha) * base + alpha * prediction
            records.append(dict(candidate=f'alpha_{float(alpha)!r}', alpha=float(alpha),
                                panel=entry['panel'], date=entry['date'],
                                regime=entry['regime'], **state_totals(truth, mixed), **physics_totals(mixed, d, context)))
            if collect_diagnostics:
                slice_records.extend(error_slice_records(truth, mixed, axes,
                    {key: records[-1][key] for key in ('candidate', 'alpha', 'panel', 'date', 'regime')}))
        if number == 1 or number % 20 == 0 or number == len(selected):
            print(f'평가 {period}: {number}/{len(selected)}일', flush=True)
    macro, grouped, daily = summarize(records)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in (('macro', macro), ('panel_regime', grouped), ('daily', daily)):
        frame.to_csv(folder / f'{period}_{name}.csv', index=False)
    if collect_diagnostics:
        summarize_error_slices(slice_records).to_csv(folder / f'{period}_error_slices.csv', index=False)
    if (grouped.lwr_count == 0).any():
        print('주의: LWR 대리 평가 대상이 없는 그룹이 있습니다. panel_regime의 lwr_count를 확인해 주세요.')
    return macro, grouped, daily, time.perf_counter() - started


def notebook_signature(contexts, entries, normalization):
    settings = dict(architecture='maginet-task13-residual-v2',
                    input_normalization={k: normalization[k] for k in ('mode', 'mean', 'std')},
                    output_scale=SCALE.tolist(), aux_names=list(AUX_NAMES), seed=SEED, window=WINDOW, stride=STRIDE,
                    width=WIDTH, blocks=BLOCKS, heads=HEADS, cheb_k=CHEB_K, graph_dim=GRAPH_DIM,
                    dropout=DROPOUT, mgl=MGL_THRESHOLD, batch=BATCH_SIZE, updates=UPDATES_PER_GROUP,
                    lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE,
                    warmup=WARMUP_EPOCHS, ramp=PHYSICS_RAMP_EPOCHS, loss_weights=LOSS_WEIGHTS,
                    select_weights=SELECT_WEIGHTS, periods=PERIODS, release=str(RELEASE_ROOT.resolve()),
                    training_dates={n: sorted(e['date'] for e in entries if e['panel'] == n and e['period'] == 'fit')
                                    for n in contexts},
                    panels={name: dict(stations=c['targets'].stations, links=c['targets'].links,
                                      station_codes=c['station_codes'].tolist(), panel_code=c['panel_code'],
                                      family=c['family_code'], direction=c['direction_code'], mode=c['mode'],
                                      dates=c['dates'], profile_hash=hashlib.sha256(c['profile'].tobytes()).hexdigest())
                            for name, c in contexts.items()})
    if TEMPORAL_DIRECT or MASK_RESAMPLE or LEARNED_GRAPH:
        settings['improvements'] = dict(temporal_direct=TEMPORAL_DIRECT, learned_graph=LEARNED_GRAPH,
                                        mask_resample='fit-link-v2' if MASK_RESAMPLE else None)
    if SPATIAL_MODE != 'legacy':
        settings['improvements']['spatial_mode'] = SPATIAL_MODE
    if CHECKPOINT_METRIC != 'selection_proxy':
        settings['checkpoint_metric'] = CHECKPOINT_METRIC
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(), settings


def run_signature(contexts, entries, normalization):
    _, settings = notebook_signature(contexts, entries, normalization)
    settings.pop('release')
    settings['format_version'] = 3
    settings['days_per_regime'] = DAYS_PER_REGIME
    # 기본 fixed 설정은 이전 로컬 체크포인트의 서명을 유지합니다.
    if BATCH_POLICY != 'fixed' or PANEL_BATCH:
        settings.update(batch_policy=BATCH_POLICY, panel_batch=dict(PANEL_BATCH),
                        panel_batches={n: panel_batch_size(c) for n, c in contexts.items()})
    inputs = []
    for entry in sorted(entries, key=lambda e: (e['panel'], e['period'], e['date'])):
        with zipfile.ZipFile(entry['path']) as archive:
            arrays = sorted((m.filename, m.file_size, m.CRC) for m in archive.infolist())
        inputs.append((entry['panel'], entry['period'], entry['date'], arrays))
    settings['input_hash'] = hashlib.sha256(json.dumps(inputs).encode()).hexdigest()
    settings['network_hashes'] = network_hashes(contexts)
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(), settings


def network_hashes(contexts):
    hashes = {}
    for name, context in contexts.items():
        digest = hashlib.sha256()
        for values in (context['station_graph'], context['mapping'], context['lanes'],
                       context['panel'].free_speed, context['panel'].lanes, context['panel'].lengths,
                       context['panel'].capacity, context['panel'].jam_density, context['panel'].edges):
            array = np.ascontiguousarray(values)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        digest.update(json.dumps(context['panel'].ramp_map, sort_keys=True).encode())
        hashes[name] = digest.hexdigest()
    return hashes


def save_checkpoint(payload, path):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def optimizer_step(model, optimizer, scaler):
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
    scaler.step(optimizer)
    scaler.update()


def load_best_maginet(entries, contexts, folder):
    best_path = folder / 'best_maginet.pt'
    if not best_path.exists() or not (folder / 'history.csv').exists():
        raise FileNotFoundError(f'저장 모델·history.csv가 필요합니다. 먼저 TRAIN_MODEL=True로 학습해 주세요: {folder}')
    normalization = fit_input_normalization(entries, CACHE_METADATA_DIR / 'normalization', legacy_folder=folder)
    signature, _ = run_signature(contexts, entries, normalization)
    saved = torch.load(best_path, map_location=DEVICE, weights_only=True)
    if saved['signature'] != signature:
        raise ValueError('저장 모델과 설정·패널·기간이 다릅니다. 해당 실행의 설정을 복원해 주세요.')
    model = MagiNetTask13(max(int(c['station_codes'].max()) for c in contexts.values()) + 1,
                         max(c['panel_code'] for c in contexts.values()) + 1,
                         max(c['family_code'] for c in contexts.values()) + 1, normalization).to(DEVICE)
    model.load_state_dict(saved['model'])
    model.eval()
    print({'loaded_best_epoch': saved['epoch'], 'checkpoint': str(best_path)})
    return model, pd.read_csv(folder / 'history.csv')


def train_maginet(entries, contexts, tensors, folder, epochs=None, resume=None):
    epochs = EPOCHS if epochs is None else epochs
    resume = RESUME if resume is None else resume
    reset_seed()
    folder.mkdir(parents=True, exist_ok=True)
    best_path, last_path = folder / 'best_maginet.pt', folder / 'last_maginet.pt'
    if not resume and any(p.exists() for p in (best_path, last_path, folder / 'best_state_maginet.pt')):
        raise FileExistsError(f'기존 학습을 보호합니다. RESUME=True 또는 새 REPORT_DIR를 사용해 주세요: {folder}')
    normalization = fit_input_normalization(entries, CACHE_METADATA_DIR / 'normalization', legacy_folder=folder)
    signature, settings = run_signature(contexts, entries, normalization)
    station_count = max(int(c['station_codes'].max()) for c in contexts.values()) + 1
    panel_count = max(c['panel_code'] for c in contexts.values()) + 1
    family_count = max(c['family_code'] for c in contexts.values()) + 1
    model = MagiNetTask13(station_count, panel_count, family_count, normalization).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and AMP_DTYPE == torch.float16)
    best_path, last_path = folder / 'best_maginet.pt', folder / 'last_maginet.pt'
    start_epoch, best, bad_epochs, best_epoch, history = 1, -float('inf'), 0, 0, []
    if resume:
        if not last_path.exists() or not best_path.exists():
            raise FileNotFoundError('재개하려면 last_maginet.pt와 best_maginet.pt가 모두 필요합니다.')
        state = torch.load(last_path, map_location=DEVICE, weights_only=True)
        if state['signature'] != signature:
            raise ValueError('설정·패널·학습 날짜가 달라 재개할 수 없습니다. REPORT_DIR를 바꾸어 새로 학습해 주세요.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, best, bad_epochs, best_epoch, history = state['epoch'] + 1, state['best'], state['bad_epochs'], state['best_epoch'], state['history']
        torch.set_rng_state(state['rng_cpu'].cpu())
        if DEVICE.type == 'cuda' and state['rng_cuda']:
            torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
        if epochs < start_epoch:
            raise ValueError(f'EPOCHS를 마지막 완료 epoch {start_epoch-1}보다 크게 지정해 주세요.')
    elif best_path.exists() or last_path.exists():
        raise FileExistsError(f'기존 학습을 보호합니다. RESUME=True 또는 새 REPORT_DIR를 사용해 주세요: {folder}')
    fit_entries = [e for e in entries if e['period'] == 'fit']
    pools = {(name, r): [e for e in fit_entries if e['panel'] == name and e['regime'] == r]
             for name in contexts for r in range(3)}
    if any(not p for p in pools.values()):
        raise ValueError('공동 학습에 panel×regime별 날짜가 모두 필요합니다.')
    metadata = checkpoint_metadata(contexts, normalization, settings)
    # 재개 설정과 학습 그룹을 확인한 뒤 결과·캐시를 각각 저장합니다.
    (folder / 'config.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    (folder / 'input_normalization.json').write_text(json.dumps(normalization, indent=2, ensure_ascii=False))
    manifest_key = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]
    manifest_dir = CACHE_METADATA_DIR / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f'cache_manifest-{manifest_key}.csv'
    temporary = manifest_path.with_suffix('.csv.tmp')
    pd.DataFrame(entries).to_csv(temporary, index=False)
    temporary.replace(manifest_path)
    direction_count = {n: sum(p.rsplit('_', 1)[0] == n.rsplit('_', 1)[0] for p in contexts) for n in contexts}
    panel_batches = {n: panel_batch_size(c) for n, c in contexts.items()}
    print({'panel_batches': panel_batches}, flush=True)
    began = time.perf_counter()
    prior_elapsed = history[-1]['elapsed_seconds'] if history else 0.
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        mask_rng = np.random.default_rng(np.random.SeedSequence([SEED, epoch, 1]))
        groups = list(pools)
        rng.shuffle(groups)
        totals = dict(State=0., N=0., dN=0., FD=0.)
        updates, train_windows, train_started = 0, 0, time.perf_counter()
        physics_factor = min(1., max(0., (epoch - WARMUP_EPOCHS) / max(PHYSICS_RAMP_EPOCHS, 1)))
        for name, regime in groups:
            pool = pools[(name, regime)]
            probabilities = np.array([e['targets'] for e in pool], np.float64)
            probabilities /= probabilities.sum()
            for step in range(UPDATES_PER_GROUP):
                if step % 4 == 0:
                    entry = pool[int(rng.choice(len(pool), p=probabilities))]
                    d = load_day(entry['path'])
                    if MASK_RESAMPLE:
                        d = resample_fit_mask(d, contexts[name], mask_rng, entry['period'])
                    starts = [s for s in window_starts(len(d['values'])) if d['target'][s:s+WINDOW].any()]
                    if not starts:
                        raise ValueError('대상 셀이 있는 학습 창이 없습니다.')
                chunk = rng.choice(starts, size=panel_batches[name], replace=True).tolist()
                batch = window_batch(d, chunk)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    output = model(**batch, ct=tensors[name])
                loss, terms = joint_loss(output.float(), d, chunk, contexts[name], tensors[name], physics_factor)
                # 도로군마다 방향 수가 달라도 누적 가중치가 같도록 맞춥니다.
                scaler.scale(loss / direction_count[name]).backward()
                optimizer_step(model, optimizer, scaler)
                updates += 1
                train_windows += len(chunk)
                for k in totals:
                    totals[k] += terms[k]
            print(f'epoch {epoch}/{epochs}: {name}/{REGIMES[regime]} 학습 완료', flush=True)
        train_seconds = time.perf_counter() - train_started
        macro, _, _, eval_seconds = evaluate(entries, contexts, tensors, model, 'early', (1.,), folder / 'early')
        proxy = float(macro.iloc[0].selection_proxy)
        score = float(macro.iloc[0][CHECKPOINT_METRIC])
        if not np.isfinite(score):
            raise FloatingPointError('체크포인트 선택 점수가 유한하지 않습니다.')
        scheduler.step(score)
        if score > best + 1e-7:
            best, bad_epochs, best_epoch = score, 0, epoch
            payload = dict(model=model.state_dict(), signature=signature, epoch=epoch, score=score,
                           checkpoint_metric=CHECKPOINT_METRIC, **metadata)
            save_checkpoint(payload, best_path)
            if CHECKPOINT_METRIC == 'State':
                save_checkpoint(payload, folder / 'best_state_maginet.pt')
        else:
            bad_epochs += 1
        row = dict(epoch=epoch, **{f'loss_{k}': v / updates for k, v in totals.items()},
                   optimizer_updates=updates, train_windows=train_windows,
                   early_State=float(macro.iloc[0].State), early_selection_proxy=proxy,
                   early_checkpoint_score=score,
                   physics_factor=physics_factor, lr=optimizer.param_groups[0]['lr'],
                   train_seconds=train_seconds, eval_seconds=eval_seconds,
                   elapsed_seconds=prior_elapsed + time.perf_counter() - began)
        history.append(row)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        save_checkpoint(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        scaler=scaler.state_dict(), signature=signature, epoch=epoch, best=best,
                        bad_epochs=bad_epochs, best_epoch=best_epoch, history=history,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all() if DEVICE.type == 'cuda' else [],
                        **metadata), last_path)
        print({'epoch': epoch, 'State': row['early_State'], 'selection_proxy': proxy,
               'checkpoint_metric': CHECKPOINT_METRIC, 'checkpoint_score': score,
               'best_epoch': best_epoch, 'physics_factor': physics_factor}, flush=True)
        if bad_epochs >= PATIENCE and epoch >= WARMUP_EPOCHS + PHYSICS_RAMP_EPOCHS + 2:
            break
    best_state = torch.load(best_path, map_location=DEVICE, weights_only=True)
    if best_state['signature'] != signature:
        raise ValueError('저장된 best 체크포인트의 설정이 다릅니다.')
    model.load_state_dict(best_state['model'])
    model.eval()
    print({'best_epoch': best_state['epoch'], 'checkpoint_metric': CHECKPOINT_METRIC,
           'best_early_score': best_state['score']})
    return model, pd.DataFrame(history)

def predict_split(contexts, model, tensors, alpha, split, folder, postprocessors=None):
    if split not in ('validation', 'private') or not 0 <= alpha <= 1:
        raise ValueError('추론 split 또는 alpha가 잘못되었습니다.')
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, c in contexts.items():
        template_path = c['release'] / 'task1' / name / split / 'sample_submission_state.csv'
        template = pd.read_csv(template_path, dtype={k: str for k in KEYS})
        if template.empty or template.duplicated(KEYS).any() or set(template.panel) != {name}:
            raise ValueError(f'잘못된 제출 템플릿: {template_path}')
        files = dated_files(c['release'] / 'corridors' / name / split / 'mainline_states_masked')
        if not files:
            raise FileNotFoundError(f'공개 masked 입력이 없습니다: {name}/{split}')
        first = StationDay.load(next(iter(files.values())), c['panel'])
        target_index = TargetIndex(c['release'], name, first, split)
        if (first.station_ids, first.station_links) != (c['targets'].stations, c['targets'].links):
            raise ValueError(f'추론 관측소 구성이 다릅니다: {name}')
        out = template.copy()
        out[['speed_kmh', 'flow_vph']] = np.nan
        for number, (date, group) in enumerate(template.groupby(template.timestamp.str[:10], sort=False), 1):
            if date not in files:
                raise ValueError(f'추론 날짜가 누락되었습니다: {name}/{date}')
            d = make_day(c, date, split=split, target_index=target_index, input_files=files)
            ti, si = d['ti'], d['si']
            base = d['base'][ti, si]
            if alpha > 0:
                output = maginet_predict(model, d, tensors[name], batch_size=panel_batch_size(c))
                neural = (postprocessors[name](d, output) if postprocessors else output)[ti, si]
            else:
                neural = base
            values = ((1 - alpha) * base + alpha * neural) * SCALE
            local = pd.DataFrame(dict(timestamp=d['timestamps'][ti], station_id=np.asarray(c['targets'].stations)[si],
                                      speed_kmh=values[:, 0], flow_vph=values[:, 1] * c['lanes'][si]))
            aligned = group[['timestamp', 'station_id']].merge(local, on=['timestamp', 'station_id'],
                                                               how='left', validate='one_to_one', sort=False)
            if aligned[['speed_kmh', 'flow_vph']].isna().any().any():
                raise ValueError(f'추론 대상 키가 일치하지 않습니다: {name}/{date}')
            out.loc[group.index, ['speed_kmh', 'flow_vph']] = aligned[['speed_kmh', 'flow_vph']].to_numpy()
            if number == 1 or number % 10 == 0:
                print(f'CSV 예측 {name}/{split}: {number}일', flush=True)
        values = out[['speed_kmh', 'flow_vph']].to_numpy()
        if not out[KEYS].equals(template[KEYS]) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'출력 값 또는 템플릿 키가 잘못되었습니다: {name}')
        path = folder / f'task1-{name}-{split}.csv'
        temporary = path.with_suffix('.csv.tmp')
        out.to_csv(temporary, index=False)
        temporary.replace(path)
        paths.append(str(path))
    return paths


def checkpoint_metadata(contexts, normalization, settings):
    """Store inference inputs as plain containers and CPU tensors (weights_only safe)."""
    packed = {}
    for name, context in contexts.items():
        panel = context['panel']
        network = {}
        for key in PanelInfo.__dataclass_fields__:
            value = getattr(panel, key)
            network[key] = torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
        packed[name] = dict(network=network, stations=context['targets'].stations,
                            station_links=context['targets'].links, dates=context['dates'], mode=context['mode'],
                            panel_code=int(context['panel_code']), family_code=int(context['family_code']),
                            direction_code=int(context['direction_code']), topology_nodes=sorted(context['topology_nodes']),
                            cache_key=context.get('cache_key'))
        for key in ('mapping', 'lanes', 'station_codes', 'station_graph', 'profile'):
            packed[name][key] = torch.from_numpy(np.asarray(context[key]).copy())
    return dict(format_version=3, settings=settings, normalization=normalization, contexts=packed,
                experiment=EXPERIMENT, runtime=dict(torch=str(torch.__version__), device=str(DEVICE),
                                                   precision=str(AMP_DTYPE) if USE_AMP else 'float32',
                                                   cpu_threads=torch.get_num_threads()))


def unpack_contexts(packed, release):
    contexts = {}
    for name, saved in packed.items():
        network = {k: v.cpu().numpy().copy() if isinstance(v, torch.Tensor) else v
                   for k, v in saved['network'].items()}
        context = {k: saved[k] for k in ('dates', 'mode', 'panel_code', 'family_code', 'direction_code')}
        context.update(release=release, panel=PanelInfo(**network), topology_nodes=set(saved['topology_nodes']),
                       targets=SimpleNamespace(stations=saved['stations'], links=saved['station_links']),
                       cache_key=saved.get('cache_key'))
        for key in ('mapping', 'lanes', 'station_codes', 'station_graph', 'profile'):
            context[key] = saved[key].cpu().numpy().copy()
        contexts[name] = context
    return contexts


SETTING_NAMES = dict(input_normalization='INPUT_NORMALIZATION', seed='SEED', window='WINDOW', stride='STRIDE',
                     width='WIDTH', blocks='BLOCKS', heads='HEADS', cheb_k='CHEB_K', graph_dim='GRAPH_DIM',
                     dropout='DROPOUT', mgl='MGL_THRESHOLD', batch='BATCH_SIZE', updates='UPDATES_PER_GROUP',
                     lr='LR', weight_decay='WEIGHT_DECAY', patience='PATIENCE', warmup='WARMUP_EPOCHS',
                     ramp='PHYSICS_RAMP_EPOCHS', loss_weights='LOSS_WEIGHTS', select_weights='SELECT_WEIGHTS',
                     periods='PERIODS', days_per_regime='DAYS_PER_REGIME',
                     batch_policy='BATCH_POLICY', panel_batch='PANEL_BATCH', checkpoint_metric='CHECKPOINT_METRIC')
DEFAULT_CONFIG = {key: globals()[key] for key in SETTING_NAMES.values()}


def restore_settings(settings):
    if settings['architecture'] != 'maginet-task13-residual-v2':
        raise ValueError('지원하지 않는 MagiNet 구조입니다.')
    globals().update(BATCH_POLICY='fixed', PANEL_BATCH={}, CHECKPOINT_METRIC='selection_proxy')
    for saved_key, variable in SETTING_NAMES.items():
        if saved_key in settings:
            value = settings[saved_key]
            globals()[variable] = value['mode'] if saved_key == 'input_normalization' else value
    improvements = settings.get('improvements', {})
    globals().update(TEMPORAL_DIRECT=bool(improvements.get('temporal_direct')),
                     MASK_RESAMPLE=bool(improvements.get('mask_resample')),
                     LEARNED_GRAPH=bool(improvements.get('learned_graph')),
                     SPATIAL_MODE=improvements.get('spatial_mode', 'legacy'))
    if (SPATIAL_MODE not in ('legacy', 'context', 'channel')
            or SPATIAL_MODE != 'legacy' and not LEARNED_GRAPH
            or CHECKPOINT_METRIC not in ('selection_proxy', 'State')):
        raise ValueError('저장된 공간 경로·체크포인트 선택 기준이 잘못되었습니다.')


def load_local_checkpoint(path, release, saved=None):
    saved = torch.load(path, map_location='cpu', weights_only=True) if saved is None else saved
    if saved.get('format_version') != 3:
        raise ValueError('Colab 체크포인트는 먼저 import-colab 명령으로 가져와 주세요.')
    signature = hashlib.sha256(json.dumps(saved['settings'], sort_keys=True).encode()).hexdigest()
    if signature != saved['signature']:
        raise ValueError('체크포인트 설정 서명이 일치하지 않습니다.')
    restore_settings(saved['settings'])
    contexts = unpack_contexts(saved['contexts'], release)
    model = MagiNetTask13(max(int(c['station_codes'].max()) for c in contexts.values()) + 1,
                         max(c['panel_code'] for c in contexts.values()) + 1,
                         max(c['family_code'] for c in contexts.values()) + 1, saved['normalization']).to(DEVICE)
    model.load_state_dict(saved['model'])
    return model.eval(), contexts, saved


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_release(path=None):
    if path is None:
        pointer = PROJECT_ROOT / 'data' / 'release_root.txt'
        path = pointer.read_text(encoding='utf-8-sig').strip() if pointer.is_file() else RELEASE_ROOT
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not (path / 'corridors').is_dir() or not (path / 'task1').is_dir():
        raise FileNotFoundError(f'corridors/와 task1/이 있는 데이터 경로를 지정해 주세요: {path}')
    return path


def checkpoint_folder(path):
    return path.parent.parent if path.parent.name == 'full' else path.parent


def configure(args):
    globals().update(DEFAULT_CONFIG)
    globals()['SPATIAL_MODE'] = 'legacy'
    globals()['DATE_OVERRIDES'] = {}
    experiment = args.experiment or 'temporal_direct'
    report_dir = (args.output_dir or PROJECT_ROOT / 'reports' / 'maginet' / experiment).resolve()
    checkpoint = (args.checkpoint or report_dir / 'full' / 'best_maginet.pt').resolve()
    source = None
    if args.command in ('evaluate', 'predict') or args.command == 'train' and args.resume:
        source_path = report_dir / 'full' / 'last_maginet.pt' if args.command == 'train' else checkpoint
        source = torch.load(source_path, map_location='cpu', weights_only=True)
        if source.get('format_version') != 3:
            raise ValueError('Colab 체크포인트는 먼저 import-colab 명령으로 가져와 주세요.')
        if args.command == 'train':
            restore_settings(source['settings'])
        experiment = source['experiment']
        if args.output_dir is None and args.command in ('evaluate', 'predict'):
            report_dir = checkpoint_folder(checkpoint)
        if args.command == 'train':
            globals()['DATE_OVERRIDES'] = {n: c['dates'] for n, c in source['settings']['panels'].items()}
    elif args.command == 'import-colab':
        source_folder = args.source_run / 'full' if (args.source_run / 'full' / 'config.json').is_file() else args.source_run
        old_settings = json.loads((source_folder / 'config.json').read_text(encoding='utf-8'))
        restore_settings(old_settings)
        globals()['DATE_OVERRIDES'] = {n: c['dates'] for n, c in old_settings['panels'].items()}
        if args.experiment is None:
            for name, (dn, temporal, resample, learned) in EXPERIMENTS.items():
                if (LOSS_WEIGHTS['dN'], TEMPORAL_DIRECT, MASK_RESAMPLE, LEARNED_GRAPH) == (dn, temporal, resample, learned):
                    experiment = name
                    break
        if args.output_dir is None:
            report_dir = PROJECT_ROOT / 'reports' / 'maginet' / experiment
    else:
        dn, temporal, resample, learned = EXPERIMENTS[experiment]
        globals().update(LOSS_WEIGHTS=dict(N=0.05, dN=dn, FD=0.), TEMPORAL_DIRECT=temporal,
                         MASK_RESAMPLE=resample, LEARNED_GRAPH=learned)
        if experiment in SPATIAL_EXPERIMENTS:
            globals().update(SPATIAL_MODE=SPATIAL_EXPERIMENTS[experiment], CHECKPOINT_METRIC='State')
    if args.checkpoint_metric is not None:
        globals()['CHECKPOINT_METRIC'] = 'State' if args.checkpoint_metric == 'state' else 'selection_proxy'
    for argument, variable in (('input_normalization', 'INPUT_NORMALIZATION'), ('seed', 'SEED'),
                               ('batch_size', 'BATCH_SIZE'), ('updates_per_group', 'UPDATES_PER_GROUP'),
                               ('epochs', 'EPOCHS'), ('patience', 'PATIENCE'), ('lr', 'LR'),
                               ('days_per_regime', 'DAYS_PER_REGIME'), ('width', 'WIDTH'),
                               ('blocks', 'BLOCKS'), ('heads', 'HEADS'), ('batch_policy', 'BATCH_POLICY')):
        value = getattr(args, argument)
        if value is not None:
            globals()[variable] = value
    if args.panel_batch is not None:
        overrides = {}
        for item in args.panel_batch:
            try:
                name, size = item.split('=', 1)
                size = int(size)
            except ValueError as error:
                raise ValueError('--panel-batch는 PANEL=양의정수 형식이어야 합니다.') from error
            if not name or size < 1 or name in overrides:
                raise ValueError('--panel-batch의 패널 중복·이름·배치 크기를 확인해 주세요.')
            overrides[name] = size
        globals()['PANEL_BATCH'] = overrides
    if source and args.experiment and args.experiment != source['experiment']:
        raise ValueError('저장 모델과 --experiment가 다릅니다.')
    if args.command == 'smoke':
        globals().update(DAYS_PER_REGIME=1 if args.days_per_regime is None else args.days_per_regime,
                         UPDATES_PER_GROUP=1 if args.updates_per_group is None else args.updates_per_group,
                         EPOCHS=2 if args.epochs is None else args.epochs)
        if args.output_dir is None:
            report_dir = report_dir / 'smoke'
    if BATCH_POLICY == 'adaptive' and args.batch_size is not None:
        raise ValueError('adaptive에서는 --batch-size 대신 --panel-batch로 배치를 지정해 주세요.')
    if (min(BATCH_SIZE, UPDATES_PER_GROUP, EPOCHS, WIDTH, BLOCKS, HEADS) < 1 or WIDTH % HEADS
            or WIDTH % 2 or not np.isfinite(LR) or LR <= 0 or PATIENCE < 1 or args.cpu_threads < 1
            or DAYS_PER_REGIME is not None and DAYS_PER_REGIME < 1):
        raise ValueError('모델 크기·학습량·학습률·날짜 수 설정을 확인해 주세요.')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다. --device cpu로 실행할 수 있습니다.')
    if args.precision == 'bf16' and (device != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('이 장치에서는 BF16을 사용할 수 없습니다.')
    amp = device == 'cuda' and args.precision != 'fp32'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    cache_dir = (args.cache_dir or PROJECT_ROOT / 'data' / 'maginet_cache').resolve()
    selected = args.panel
    if source and args.command == 'train':
        selected = selected or sorted(source['contexts'])
    elif args.command == 'import-colab':
        selected = selected or sorted(old_settings['panels'])
    globals().update(EXPERIMENT=experiment, RELEASE_ROOT=resolve_release(args.release_root), REPORT_DIR=report_dir,
                     REPORT_BASE=report_dir.parent, CACHE_STORE_DIR=cache_dir, CACHE_DIR=cache_dir,
                     CACHE_ROOT=cache_dir, CACHE_CONTAINER=cache_dir, CACHE_METADATA_DIR=cache_dir / 'metadata',
                     CACHE_MODE=args.cache_mode, SAVED_CACHE_ZIP=args.cache_zip,
                     CACHE_PROFILE_DIR=args.legacy_profile_dir or (source_folder if args.command == 'import-colab'
                                                                  else cache_dir / 'legacy_profiles'),
                     PANELS=selected, DEVICE=torch.device(device), USE_AMP=amp, AMP_DTYPE=dtype,
                     TRAIN_MODEL=args.command in ('train', 'smoke'), RESUME=args.resume)
    torch.set_num_threads(args.cpu_threads)
    reset_seed()
    load_day.cache_clear()
    args.checkpoint = checkpoint
    args.saved_checkpoint = source
    if (args.select_retrieval or args.retrieval_selection is not None) and args.output_dir is None:
        globals()['REPORT_DIR'] = REPORT_DIR / 'retrieval'
    print(dict(command=args.command, experiment=experiment, device=device,
               precision=str(dtype) if amp else 'float32', release=str(RELEASE_ROOT),
               cache=str(cache_dir), reports=str(REPORT_DIR)), flush=True)


def prepare_inputs(smoke=False):
    names = available_panels(RELEASE_ROOT)
    if smoke and PANELS is None:
        names = names[:1]
    unknown = set(PANEL_BATCH) - set(names)
    if unknown:
        raise ValueError(f'학습에 포함되지 않은 --panel-batch 패널입니다: {sorted(unknown)}')
    root, container = (restore_input_cache(names, SAVED_CACHE_ZIP, CACHE_DIR / 'imported') if SAVED_CACHE_ZIP
                       else (CACHE_DIR, CACHE_DIR))
    globals().update(CACHE_ROOT=root, CACHE_CONTAINER=container)
    contexts, entries, offset = {}, [], 0
    for code, name in enumerate(names):
        context = prepare_panel(RELEASE_ROOT, name, code, offset)
        offset += len(context['targets'].stations)
        contexts[name] = context
    attach_vocabulary(contexts)
    identity = dict(format='maginet-local-input-v1', networks=network_hashes(contexts),
                    panels={n: dict(stations=c['targets'].stations, links=c['targets'].links,
                                    fit_dates=c['dates']['fit'],
                                    profile_hash=hashlib.sha256(c['profile'].tobytes()).hexdigest())
                            for n, c in contexts.items()})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    destination = CACHE_DIR / 'inputs' / key
    for name, context in contexts.items():
        context['cache_key'] = key
        for period in ('fit', 'early', 'blend'):
            for date in context['dates'][period]:
                source = root / name / period / f'{date}.npz'
                target = destination / name / period / f'{date}.npz'
                if source.is_file() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.npz.tmp')
                    shutil.copyfile(source, temporary)
                    temporary.replace(target)
        entries.extend(cache_context(context, destination))
    globals()['CACHE_ROOT'] = destination
    return contexts, entries


def evaluation_inputs(contexts, period):
    entries = []
    for name, context in contexts.items():
        folder = RELEASE_ROOT / 'corridors' / name / 'train'
        context['files'] = dated_files(folder / 'mainline_states')
        context['masked'] = dated_files(folder / 'mainline_states_masked')
        dates = context['dates'][period]
        if not dates:
            raise ValueError(f'체크포인트의 {name}/{period} 날짜가 없습니다.')
        first = StationDay.load(context['files'][dates[0]], context['panel'])
        target_index = TargetIndex(RELEASE_ROOT, name, first)
        if (target_index.stations, target_index.links) != (context['targets'].stations, context['targets'].links):
            raise ValueError(f'체크포인트의 관측소와 로컬 데이터가 다릅니다: {name}')
        context['targets'] = target_index
        root = CACHE_DIR / 'inputs' / context['cache_key'] if context.get('cache_key') else CACHE_ROOT
        entries.extend(cache_context(context, root, periods=(period,)))
    return entries


def select_alpha(entries, contexts, model, checkpoint):
    tensors = {n: tensor_context(c) for n, c in contexts.items()}
    macro, _, _, _ = evaluate(entries, contexts, tensors, model, 'blend', ALPHAS, REPORT_DIR,
                               collect_diagnostics=True)
    best = macro.sort_values([CHECKPOINT_METRIC, 'alpha'], ascending=[False, True]).iloc[0]
    selection = dict(alpha=float(best.alpha), selection_proxy=float(best.selection_proxy),
                     State=float(best.State), criterion=dict(State=1.) if CHECKPOINT_METRIC == 'State' else SELECT_WEIGHTS,
                     selected_on='2031-01-16~31',
                     panels=sorted(contexts), checkpoint_sha256=file_digest(checkpoint))
    path = REPORT_DIR / 'selected_alpha.json'
    path.write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    print(macro.to_string(index=False), flush=True)
    return selection['alpha']


def prediction_alpha(args, contexts):
    if args.alpha is not None:
        return args.alpha
    selection = args.selection or checkpoint_folder(args.checkpoint) / 'selected_alpha.json'
    saved = json.loads(selection.read_text(encoding='utf-8'))
    alpha = float(saved['alpha'])
    if (not 0 <= alpha <= 1 or saved['checkpoint_sha256'] != file_digest(args.checkpoint)
            or not set(contexts) <= set(saved['panels'])):
        raise ValueError('선택한 alpha의 값·체크포인트·패널이 다릅니다.')
    return alpha


def retrieval_memories(contexts):
    from historical_retrieval import HistoricalRetrieval
    entries = evaluation_inputs(contexts, 'fit')
    memories = {}
    for name in contexts:
        memories[name] = HistoricalRetrieval(e['path'] for e in entries if e['panel'] == name)
        print(f'과거 검색 준비: {name}, fit {len(memories[name].dates)}일', flush=True)
    return memories


def retrieval_metrics(frame, weights):
    weights = np.broadcast_to(weights, (len(frame), 2))
    mix = np.stack([1 - weights, weights], -1)
    sse = np.einsum('gci,gcij,gcj->gc', mix, np.stack(frame.gram), mix)
    rms = np.sqrt(np.maximum(sse, 0) / frame['count'].to_numpy()[:, None])
    state = .54 * np.maximum(0, 1 - rms[:, 0] / 25) + .46 * np.maximum(0, 1 - rms[:, 1] / 600)
    values = pd.DataFrame(dict(State=state, speed_rmse_kmh=rms[:, 0], flow_rmse_per_lane=rms[:, 1]), index=frame.index)
    return {k: macro_mean(v.to_dict()) for k, v in values.groupby(level='panel').mean().items()}


def select_retrieval(entries, contexts, tensors, model, memories, checkpoint, folder):
    from historical_retrieval import CONFIG, METHODS
    from scipy.optimize import minimize_scalar
    records = []
    for number, entry in enumerate(entries, 1):
        if entry['period'] != 'early':
            raise ValueError('검색 혼합 비율은 early 구간에서만 선택합니다.')
        day = load_day(entry['path'])
        output = maginet_predict(model, day, tensors[entry['panel']], batch_size=panel_batch_size(contexts[entry['panel']]))
        candidates = memories[entry['panel']].candidates(day, output)
        ti, si = day['ti'], day['si']
        truth = day['truth'][ti, si].astype(np.float64) * SCALE
        baseline_error = output[ti, si].astype(np.float64) * SCALE - truth
        for method, candidate in candidates.items():
            errors = np.stack([baseline_error, candidate[ti, si].astype(np.float64) * SCALE - truth], -1)
            records.append(dict(panel=entry['panel'], regime=entry['regime'], method=method,
                                count=len(truth), gram=np.einsum('nci,ncj->cij', errors, errors)))
        if number == 1 or number % 20 == 0 or number == len(entries):
            print(f'검색 비율 선택 early: {number}/{len(entries)}일', flush=True)
    comparisons = []
    for method in METHODS:
        frame = pd.DataFrame([r for r in records if r['method'] == method]).groupby(['panel', 'regime']).agg(
            count=('count', 'sum'), gram=('gram', 'sum'))
        if len(frame) != 3 * len(contexts):
            raise ValueError('검색 비율 선택에 panel×regime이 누락됐습니다.')
        weights = []
        for channel in (0, 1):
            def loss(w):
                return -retrieval_metrics(frame, np.eye(2)[channel] * w)['State']
            optimum = minimize_scalar(loss, bounds=(0, 1), method='bounded', options=dict(xatol=1e-7))
            if not optimum.success:
                raise RuntimeError('검색 혼합 비율 최적화에 실패했습니다.')
            weights.append(min((0., 1., float(optimum.x)), key=loss))
        comparisons.append(dict(method=method, speed_weight=weights[0], flow_weight=weights[1],
                                **retrieval_metrics(frame, weights), baseline_State=retrieval_metrics(frame, 0.)['State']))
    best = max(comparisons, key=lambda row: row['State'])
    selection = dict(format='maginet-retrieval-v1', config=CONFIG, method=best['method'],
                     weights=[best['speed_weight'], best['flow_weight']], selected_on='early',
                     checkpoint_sha256=file_digest(checkpoint),
                     panels={n: dict(fit_dates=memory.dates, memory_sha256=memory.sha256) for n, memory in memories.items()})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'selected_retrieval.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    pd.DataFrame(comparisons).to_csv(folder / 'retrieval_early_comparison.csv', index=False)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)
    return selection


def retrieval_postprocessors(path, contexts, memories, checkpoint, folder):
    from historical_retrieval import CONFIG, METHODS
    saved = json.loads(path.read_text(encoding='utf-8'))
    weights = np.asarray(saved['weights'], dtype=float)
    if (saved.get('format') != 'maginet-retrieval-v1' or saved['config'] != CONFIG
            or saved['method'] not in METHODS or weights.shape != (2,) or not np.isfinite(weights).all()
            or (weights < 0).any() or (weights > 1).any()
            or saved['checkpoint_sha256'] != file_digest(checkpoint) or not set(contexts) <= set(saved['panels'])):
        raise ValueError('검색 선택 파일의 설정·비율·체크포인트·패널이 다릅니다.')
    for name, memory in memories.items():
        if saved['panels'][name]['memory_sha256'] != memory.sha256:
            raise ValueError(f'검색 비율을 선택한 fit 자료가 다릅니다: {name}')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'retrieval_run.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
    return {name: partial(memory.apply, method=saved['method'], weights=weights) for name, memory in memories.items()}


def import_colab(contexts, entries, source_run):
    source = source_run / 'full' if (source_run / 'full' / 'config.json').is_file() else source_run
    original = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    normalization = fit_input_normalization(entries, CACHE_METADATA_DIR / 'normalization')
    _, expected = notebook_signature(contexts, entries, normalization)
    expected['release'] = original.get('release')
    if json.dumps(expected, sort_keys=True) != json.dumps(original, sort_keys=True):
        raise ValueError('Colab 설정과 로컬 데이터·기간·과거 평균·모델 설정이 다릅니다.')
    old_signature = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    signature, settings = run_signature(contexts, entries, normalization)
    target = REPORT_DIR / 'full'
    if (target / 'best_maginet.pt').exists() or (target / 'last_maginet.pt').exists():
        raise FileExistsError('가져오기 결과 폴더에 체크포인트가 이미 있습니다.')
    payloads = {}
    for name in ('best_maginet.pt', 'last_maginet.pt'):
        path = source / name
        if path.is_file():
            state = torch.load(path, map_location='cpu', weights_only=True)
            if state['signature'] != old_signature:
                raise ValueError(f'Colab config.json과 체크포인트 서명이 다릅니다: {path}')
            state.update(signature=signature, **checkpoint_metadata(contexts, normalization, settings))
            payloads[name] = state
    if 'best_maginet.pt' not in payloads:
        raise FileNotFoundError(source / 'best_maginet.pt')
    target.mkdir(parents=True, exist_ok=True)
    for name, state in payloads.items():
        save_checkpoint(state, target / name)
    for name, value in (('config.json', settings), ('input_normalization.json', normalization)):
        (target / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    if (source / 'history.csv').is_file():
        shutil.copyfile(source / 'history.csv', target / 'history.csv')
    print(f'Colab 모델을 별도 로컬 폴더에 가져왔습니다: {target}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train', 'evaluate', 'predict', 'smoke', 'import-colab'))
    parser.add_argument('--release-root', type=Path, help='기본값: data/release_root.txt')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--cache-mode', choices=('build', 'reuse'), default='build')
    parser.add_argument('--cache-zip', type=Path)
    parser.add_argument('--legacy-profile-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--panel', action='append')
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS))
    parser.add_argument('--input-normalization', choices=('fixed', 'standard'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--batch-policy', choices=('fixed', 'adaptive'),
                        help='adaptive: 관측소 128개 이하 배치 8, 초과 4; 기본 fixed')
    parser.add_argument('--panel-batch', action='append', metavar='PANEL=SIZE',
                        help='패널별 배치 재지정; 반복 가능, batch-policy보다 우선')
    parser.add_argument('--updates-per-group', type=int)
    parser.add_argument('--days-per-regime', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--blocks', type=int)
    parser.add_argument('--heads', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--checkpoint-metric', choices=('state', 'proxy'),
                        help='최적 모델·scheduler·조기 종료 기준; 새 공간 실험 기본 state, 기존 실험 proxy')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--precision', choices=('auto', 'fp32', 'bf16'), default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--period', choices=('early', 'blend', 'february'), default='blend')
    parser.add_argument('--select-alpha', action='store_true')
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--split', choices=('validation', 'private', 'both'), default='validation')
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--select-retrieval', action='store_true', help='early에서 과거 검색 방법과 채널별 비율 선택')
    parser.add_argument('--retrieval-selection', type=Path, help='고정한 selected_retrieval.json으로 평가·추론')
    args = parser.parse_args(argv)
    if args.command in ('evaluate', 'predict'):
        training_only = ('input_normalization', 'seed', 'updates_per_group', 'epochs', 'patience', 'lr',
                         'days_per_regime', 'width', 'blocks', 'heads', 'checkpoint_metric')
        supplied = ['--' + key.replace('_', '-') for key in training_only if getattr(args, key) is not None]
        if supplied:
            parser.error(f'평가·추론에서는 저장된 학습 설정을 사용합니다. 지정할 수 없는 옵션: {", ".join(supplied)}')
    if args.command in ('evaluate', 'predict', 'import-colab') and (
            args.batch_policy is not None or args.panel_batch is not None):
        parser.error('패널별 배치는 train·smoke·prepare에서 지정하며 평가·추론은 저장된 설정을 사용합니다.')
    if args.command == 'import-colab' and args.source_run is None:
        parser.error('import-colab에는 --source-run이 필요합니다.')
    if args.alpha is not None and not 0 <= args.alpha <= 1:
        parser.error('--alpha는 0~1이어야 합니다.')
    if args.select_alpha and (args.command != 'evaluate' or args.period != 'blend'):
        parser.error('--select-alpha는 evaluate --period blend에서만 사용할 수 있습니다.')
    if args.resume and args.command != 'train':
        parser.error('--resume은 train에서만 사용할 수 있습니다.')
    retrieval = args.select_retrieval or args.retrieval_selection is not None
    if args.select_retrieval and (args.command != 'evaluate' or args.period != 'early'):
        parser.error('--select-retrieval은 evaluate --period early에서만 사용할 수 있습니다.')
    if retrieval and (args.command not in ('evaluate', 'predict') or args.select_alpha or args.selection
                      or args.alpha is not None and args.alpha != 1. or args.select_retrieval and args.retrieval_selection):
        parser.error('검색 옵션은 evaluate·predict에서만 쓰며 기존 alpha 선택과 함께 지정하지 않습니다.')
    if retrieval:
        args.alpha = 1.
    configure(args)
    if args.command in ('prepare', 'train', 'smoke', 'import-colab'):
        if args.command in ('train', 'smoke') and not RESUME:
            if any((REPORT_DIR / 'full' / n).exists() for n in
                   ('best_maginet.pt', 'last_maginet.pt', 'best_state_maginet.pt')):
                raise FileExistsError('기존 체크포인트가 있습니다. --resume 또는 새 --output-dir를 사용해 주세요.')
        contexts, entries = prepare_inputs(smoke=args.command == 'smoke')
        if args.command == 'prepare':
            normalization = fit_input_normalization(entries, CACHE_METADATA_DIR / 'normalization')
            print(dict(cached_days=len(entries), panels=sorted(contexts), normalization=normalization), flush=True)
        elif args.command == 'import-colab':
            import_colab(contexts, entries, args.source_run)
        else:
            tensors = {n: tensor_context(c) for n, c in contexts.items()}
            model, history = train_maginet(entries, contexts, tensors, REPORT_DIR / 'full')
            print(history.tail().to_string(index=False), flush=True)
            checkpoint = REPORT_DIR / 'full' / 'best_maginet.pt'
            select_alpha(entries, contexts, model, checkpoint)
    else:
        model, contexts, _ = load_local_checkpoint(args.checkpoint, RELEASE_ROOT, saved=args.saved_checkpoint)
        if args.batch_size is not None:
            globals().update(BATCH_SIZE=args.batch_size, BATCH_POLICY='fixed', PANEL_BATCH={})
        if args.panel:
            unknown = set(args.panel) - set(contexts)
            if unknown:
                raise ValueError(f'체크포인트에 없는 패널입니다: {sorted(unknown)}')
            contexts = {n: contexts[n] for n in sorted(set(args.panel))}
        postprocessors = None
        if retrieval:
            memories = retrieval_memories(contexts)
            if args.select_retrieval:
                entries = evaluation_inputs(contexts, 'early')
                select_retrieval(entries, contexts, {n: tensor_context(c) for n, c in contexts.items()}, model,
                                 memories, args.checkpoint, REPORT_DIR)
                return
            postprocessors = retrieval_postprocessors(args.retrieval_selection, contexts, memories,
                                                      args.checkpoint, REPORT_DIR)
        if args.command == 'evaluate':
            entries = evaluation_inputs(contexts, args.period)
            if args.select_alpha:
                select_alpha(entries, contexts, model, args.checkpoint)
            else:
                alpha = prediction_alpha(args, contexts) if args.period == 'february' else (
                    args.alpha if args.alpha is not None else 1.)
                alphas = tuple(sorted({0., alpha}))
                macro, _, _, _ = evaluate(entries, contexts, {n: tensor_context(c) for n, c in contexts.items()},
                                           model, args.period, alphas, REPORT_DIR, collect_diagnostics=True,
                                           postprocessors=postprocessors)
                print(macro.to_string(index=False), flush=True)
        else:
            alpha = prediction_alpha(args, contexts)
            splits = ('validation', 'private') if args.split == 'both' else (args.split,)
            tensors = {n: tensor_context(c) for n, c in contexts.items()}
            for split in splits:
                paths = predict_split(contexts, model, tensors, alpha, split, REPORT_DIR / 'submissions',
                                      postprocessors=postprocessors)
                print('\n'.join(paths), flush=True)


if __name__ == '__main__':
    main()
