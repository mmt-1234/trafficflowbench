"""Fit-only historical patterns for optional frozen MagiNet inference."""
import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = dict(neighbors=4, window=36, stride=18, shifts=[-2, 0, 2], min_observations=8)
METHODS = ('global', 'local', 'local_gated')


class HistoricalRetrieval:
    def __init__(self, paths):
        paths = sorted(map(Path, paths))
        if not paths or len({p.stem for p in paths}) != len(paths):
            raise ValueError('중복 없는 fit 날짜별 NPZ가 필요합니다.')
        self.dates = [p.stem for p in paths]
        residuals, valid = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as day:
                active = day['active'].astype(bool)
                residual = np.where(active[..., None], day['truth'] - day['aux'][..., 8:10], 0.)
                if residual.shape != (*active.shape, 2) or not np.isfinite(residual).all():
                    raise ValueError(f'잘못된 fit 관측 배열입니다: {path}')
                residuals.append(residual.astype(np.float32))
                valid.append(active)
        # ponytail: keep one fit memory in RAM; persist only if rebuilding becomes a bottleneck.
        self.residuals, self.valid = np.stack(residuals), np.stack(valid)
        digest = hashlib.sha256(json.dumps(self.dates).encode())
        digest.update(self.residuals.tobytes())
        digest.update(self.valid.tobytes())
        self.sha256 = digest.hexdigest()

    def candidates(self, day, network, methods=METHODS):
        if not set(methods) <= set(METHODS):
            raise ValueError('지원하지 않는 과거 검색 방법입니다.')
        values, visible, prior = day['values'][..., :2], day['visible'][..., :2], day['aux'][..., 8:10]
        if (values.shape != prior.shape or visible.shape != prior.shape or network.shape != prior.shape
                or prior.shape != self.residuals.shape[1:]):
            raise ValueError('과거 검색의 시각·관측소·채널 구성이 다릅니다.')
        if self.dates[-1] >= str(day['timestamps'][0])[:10]:
            raise ValueError('과거 검색은 추론 날짜보다 이른 fit 자료만 사용해야 합니다.')
        visible = visible.astype(bool) & np.isfinite(values)
        residual = np.where(visible, values - prior, 0.)
        results = {}
        if 'global' in methods:
            estimate = np.zeros_like(prior)
            for channel in (0, 1):
                estimate[..., channel], _ = self._retrieve(residual, visible, 0, len(prior), channel, [0])
            results['global'] = np.maximum(prior + estimate, 0.)
        if set(methods) & {'local', 'local_gated'}:
            sums = np.zeros_like(prior, dtype=np.float64)
            gated = np.zeros_like(sums)
            weights = np.zeros((len(prior), 1, 1), dtype=np.float64)
            width = min(CONFIG['window'], len(prior))
            starts = sorted(set(range(0, len(prior) - width + 1, CONFIG['stride'])) | {len(prior) - width})
            taper = np.hanning(width + 2)[1:-1, None]
            for start in starts:
                stop = start + width
                for channel in (0, 1):
                    estimate, confidence = self._retrieve(residual, visible, start, stop, channel, CONFIG['shifts'])
                    candidate = np.maximum(prior[start:stop, :, channel] + estimate, 0.)
                    sums[start:stop, :, channel] += taper * candidate
                    gated[start:stop, :, channel] += taper * (
                        network[start:stop, :, channel] + confidence * (candidate - network[start:stop, :, channel]))
                weights[start:stop] += taper[..., None]
            if 'local' in methods:
                results['local'] = (sums / weights).astype(np.float32)
            if 'local_gated' in methods:
                results['local_gated'] = (gated / weights).astype(np.float32)
        return results

    def _retrieve(self, residual, visible, start, stop, channel, shifts):
        known = visible[start:stop, :, channel]
        query = residual[start:stop, :, channel]
        if known.sum() < CONFIG['min_observations']:
            return np.zeros_like(query), 0.
        candidates, masks, scores = [], [], []
        indices = np.arange(start, stop)
        for shift in shifts:
            shifted = indices + shift
            inside = (shifted >= 0) & (shifted < self.residuals.shape[1])
            indices_safe = shifted.clip(0, self.residuals.shape[1] - 1)
            patterns = self.residuals[..., channel][:, indices_safe]
            valid = self.valid[:, indices_safe] & inside[None, :, None]
            shared = valid & known[None]
            count = shared.sum(axis=(1, 2))
            error = np.where(shared, patterns - query, 0.)
            distance = np.square(error).sum(axis=(1, 2), dtype=np.float64) / np.maximum(count, 1)
            distance[count < max(CONFIG['min_observations'], known.sum() // 2)] = np.inf
            candidates.append(patterns)
            masks.append(valid)
            scores.append(distance)
        scores = np.stack(scores)
        # One shift per historical day prevents the same day occupying all nearest slots.
        best_shift = scores.argmin(axis=0)
        distance = scores[best_shift, np.arange(len(self.dates))]
        neighbors = np.argsort(distance, kind='stable')[:CONFIG['neighbors']]
        neighbors = neighbors[np.isfinite(distance[neighbors])]
        if not len(neighbors):
            return np.zeros_like(query), 0.
        patterns = np.stack([candidates[best_shift[n]][n] for n in neighbors])
        valid = np.stack([masks[best_shift[n]][n] for n in neighbors])
        estimate = np.where(valid, patterns, 0.).sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        # Compare only public observations; target truth/active are never read here.
        reference = np.square(query[known]).mean(dtype=np.float64)
        mismatch = np.square((estimate - query)[known]).mean(dtype=np.float64)
        uncertainty = np.where(valid, np.square(patterns - estimate), 0.).sum(dtype=np.float64) / max(valid.sum(), 1)
        confidence = np.clip(1 - mismatch / (reference + 1e-8), 0, 1) * reference / (reference + uncertainty + 1e-8)
        return estimate, float(confidence)

    def apply(self, day, network, method, weights):
        candidate = self.candidates(day, network, (method,))[method]
        return (network + np.asarray(weights, np.float32) * (candidate - network)).astype(np.float32)
