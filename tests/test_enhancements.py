"""Trend/detail reconstruction, masked station context, and fit-only balancing."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from test_v2 import day, panel
from data import FEATURE_NAMES, STATE_CONTEXT_NAMES, features_v2, slice_day, state_features_v2
from fast_model import AdaptiveGraphResidual, TrafficEncoder, TrafficQueueModel, TrafficStateModel, haar_merge, haar_split
from learning import check_model_requests, fit_regime_weights, model_from_checkpoint, state_loss


class Enhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_haar_reconstructs_odd_windows_and_preserves_details(self):
        for steps in (1, 5, 12, 72):
            x = torch.randn(2, steps, 3, 4, requires_grad=True)
            low, details, lengths = x, [], []
            for _ in range(3):
                lengths.append(low.shape[1])
                low, high = haar_split(low)
                details.append(high)
            for high, length in zip(reversed(details), reversed(lengths)):
                low = haar_merge(low, high, length)
            torch.testing.assert_close(low, x)
            low.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
        _, high = haar_split(torch.ones(1, 5, 2, 4))
        self.assertEqual(high.abs().sum().item(), 0)

    def test_station_endpoints_have_separate_masks_and_window_boundaries(self):
        observation = day(24)
        observation.speed[:12] = np.nan
        observation.flow_lane[:12] = np.nan
        observation.speed[2, 0], observation.speed[6, 0] = 0, 4
        observation.flow_lane[4, 0] = 2
        prior = np.full((2016, 2, 3), 3, np.float32)
        index = np.arange(12)
        x, context = state_features_v2(observation, panel(), index, prior, rich=True)
        _, legacy = state_features_v2(observation, panel(), index, prior)
        self.assertEqual(context.shape, (12, 2, len(STATE_CONTEXT_NAMES)))
        np.testing.assert_array_equal(context[..., :3], legacy)
        self.assertTrue(np.isfinite(context).all())
        self.assertEqual(context[3, 0, 0], 1)
        np.testing.assert_array_equal(context[3, 0, 3:5], [0, 4])
        np.testing.assert_allclose(context[3, 0, 5:7], [1 / 72, 3 / 72])
        np.testing.assert_array_equal(context[3, 0, 7:10], [1, 1, 0])
        self.assertEqual(context[2, 0, 9], 1)  # An observed zero is an endpoint.
        np.testing.assert_array_equal(context[0, 0, 7:9], [0, 1])
        np.testing.assert_array_equal(context[11, 0, 7:9], [1, 0])
        np.testing.assert_array_equal(context[:, 1, 3:5], np.full((12, 2), 3))
        self.assertEqual(context[:, 1, 7:10].sum(), 0)
        self.assertEqual(context[3, 0, 16], 0)  # Flow has its own observation mask.
        observation.speed[12:], observation.flow_lane[12:] = 999, 999
        changed_x, changed_context = state_features_v2(observation, panel(), index, prior, rich=True)
        np.testing.assert_array_equal(x, changed_x)
        np.testing.assert_array_equal(context, changed_context)

    def test_adaptive_graph_connects_components_with_sparse_trainable_weights(self):
        torch.manual_seed(19)
        graph = AdaptiveGraphResidual(8, 2)
        identities = torch.randn(4, 8, requires_grad=True)
        neighbors, weights = graph.layout(identities)
        self.assertEqual(neighbors.shape, (4, 2))
        self.assertFalse((neighbors == torch.arange(4)[:, None]).any())
        torch.testing.assert_close(weights.sum(-1), torch.ones(4))
        x = torch.randn(1, 3, 4, 8, requires_grad=True)
        observation = torch.zeros(1, 3, 4, 31)
        output = graph(x, (neighbors, weights), observation)
        changed = x.detach().clone()
        changed[:, :, neighbors[0, 0]] += 5
        self.assertGreater((graph(changed, (neighbors, weights), observation)[:, :, 0] - output[:, :, 0]).abs().sum().item(), 0)
        output.square().sum().backward()
        self.assertGreater(graph.source.weight.grad.abs().sum().item(), 0)
        self.assertGreater(identities.grad.abs().sum().item(), 0)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = graph(x[:, :, permutation], graph.layout(identities[permutation]), observation[:, :, permutation])
        torch.testing.assert_close(permuted, output[:, :, permutation])
        single = graph(torch.ones(1, 3, 1, 8), graph.layout(torch.ones(1, 8)), torch.zeros(1, 3, 1, 31))
        self.assertEqual(single.abs().sum().item(), 0)

    def test_vectorized_station_context_matches_scalar_reference(self):
        rng = np.random.default_rng(42)
        observation = day(72)
        observation.timestamps = pd.date_range("2030-06-01", periods=72, freq="10min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        observation.station_to_link = np.array([0, 1])
        for values in (observation.speed, observation.flow_lane):
            values[:] = rng.uniform(0, 10, values.shape)
            values[rng.random(values.shape) < 0.7] = np.nan
        observation.speed[:, 1] = np.nan
        observation.speed[20, 0] = 0
        observation.flow_lane[10, 1] = np.inf
        prior = rng.random((2016, 2, 3), dtype=np.float32)
        # Duplicate indices are the existing day-boundary padding contract.
        index = np.clip(np.arange(-12, 60), 0, 71)
        window = slice_day(observation, index)
        dates = pd.to_datetime(window.timestamps, utc=True)
        slots = np.asarray(dates.dayofweek * 288 + dates.hour * 12 + dates.minute // 5)
        elapsed = np.asarray((dates - dates[0]).total_seconds() / 300, dtype=np.float32)
        expected = np.zeros((72, 2, len(STATE_CONTEXT_NAMES)), np.float32)
        for station, link in enumerate(observation.station_to_link):
            for channel, values in enumerate((window.speed, window.flow_lane)):
                good = np.flatnonzero(np.isfinite(values[:, station]))
                fallback = prior[slots, link, channel]
                expected[:, station, channel] = np.interp(np.arange(72), good, values[good, station]) if len(good) else fallback
                offset = 3 + 7 * channel
                for t in range(72):
                    for side, candidates in enumerate((good[good <= t], good[good >= t])):
                        valid = len(candidates) > 0
                        nearest = candidates[-1 if side == 0 else 0] if valid else t
                        expected[t, station, offset + side] = values[nearest, station] if valid else fallback[t]
                        expected[t, station, offset + 2 + side] = min(abs(elapsed[t] - elapsed[nearest]), 72) / 72 if valid else 1
                        expected[t, station, offset + 4 + side] = valid
                expected[:, station, offset + 6] = np.isfinite(values[:, station])
                expected[:, station, 18 + channel] = fallback
        expected[..., 2] = observation.lanes / 5
        expected[..., 17] = observation.quality[index]
        for rich in (False, True):
            _, actual = state_features_v2(observation, panel(), index, prior, rich=rich)
            np.testing.assert_array_equal(actual, expected if rich else expected[..., :3])

    def test_signal_wavelet_excludes_embeddings_and_other_features(self):
        encoder = TrafficEncoder(features=31, node_count=2, width=8, blocks=1, heads=2,
                                 dropout=0, backend="scan", missing_aware=True,
                                 wavelet_levels=2, wavelet_input="signals").eval()
        x = torch.rand(1, 12, 2, 31)
        x[..., 10:13] = 1
        x[:, 2:7, 0, 10:13] = 0
        x[:, 0, 1, 0] = 0
        ids, edges = torch.tensor([1, 2]), torch.empty(2, 0, dtype=torch.long)
        captured = []
        hook = encoder.wavelet.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        output = encoder(x, ids, ((0, 2),), edges)
        expected = torch.where(x[..., 10:13] > 0, x[..., (0, 1, 9)], x[..., 25:28])
        torch.testing.assert_close(captured[-1], expected)
        self.assertEqual(captured[-1][0, 0, 1, 0].item(), 0)
        changed = x.clone()
        changed[..., 28:31] = 0
        for column in (0, 1, 9):
            changed[:, 2:7, 0, column] = float("nan")
        with torch.no_grad():
            for embedding in (encoder.slot, encoder.weekday, encoder.relative):
                embedding.weight.add_(100)
            encoder.missing_embedding.add_(100)
        encoder(changed, ids, ((0, 2),), edges)
        torch.testing.assert_close(captured[0], captured[1])
        hook.remove()
        # Gradients also reach every band and the measurement-to-hidden projection.
        encoder(x, ids, ((0, 2),), edges).square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(encoder.wavelet_project.weight.grad.abs().sum().item(), 0)
        for band in encoder.wavelet.bands:
            self.assertGreater(band[0].weight.grad.abs().sum().item(), 0)

    def test_split_decoder_is_channel_independent_and_legacy_load_is_exact(self):
        config = dict(features=31, node_count=2, width=8, blocks=1, heads=2, dropout=0, backend="scan", wavelet_levels=2)
        model = TrafficStateModel(station_decoder="split", wavelet_input="signals", **config).eval()
        x = torch.rand(1, 12, 2, 31)
        x[..., 10:13] = 1
        context = torch.ones(1, 12, 2, len(STATE_CONTEXT_NAMES))
        ids, edges = torch.tensor([1, 2]), torch.empty(2, 0, dtype=torch.long)
        forward = lambda m: m(x, context, ids - 1, ids, ((0, 2),), edges)
        torch.testing.assert_close(forward(model), context[..., :2])
        with torch.no_grad():
            model.head[1][-1].bias.fill_(1)
        output = forward(model)
        torch.testing.assert_close(output[..., 0], context[..., 0])
        self.assertGreater((output[..., 1] - context[..., 1]).min().item(), 0)
        output[..., 1].sum().backward()
        for parameter in list(model.head[0].parameters()) + list(model.gate[0].parameters()):
            self.assertEqual(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(model.gate[1].bias.grad.abs().sum().item(), 0)
        legacy = TrafficStateModel(station_decoder="gated", **config).eval()
        torch.nn.init.normal_(legacy.head[-1].weight, std=0.01)
        payload = dict(format_version=2, task="state", feature_names=list(FEATURE_NAMES),
                       model_config=dict(config, station_decoder="gated"),
                       state_context_names=list(STATE_CONTEXT_NAMES), weights=legacy.state_dict())
        restored = model_from_checkpoint(payload, "cpu").eval()
        self.assertEqual(restored.encoder.wavelet_input, "embedded")
        self.assertIsNone(restored.encoder.wavelet_project)
        torch.testing.assert_close(forward(legacy), forward(restored), rtol=0, atol=0)
        options = SimpleNamespace(graph=False, missing_aware=False, graph_mode="mean", wavelet_input="signals")
        with self.assertRaisesRegex(ValueError, "new checkpoint"):
            check_model_requests(options, payload["model_config"])

    def test_wavelet_ignores_masked_placeholders_and_trains_all_bands(self):
        torch.manual_seed(13)
        encoder = TrafficEncoder(features=31, node_count=4, width=8, blocks=1, heads=2,
                                 dropout=0, backend="scan", wavelet_levels=2).eval()
        x = torch.rand(1, 12, 4, 31)
        x[..., 10:13] = 1
        x[:, 2:7, :2, 10:13] = 0
        changed = x.clone()
        for column in (0, 1, 9):
            changed[:, 2:7, :2, column] = float("nan")
        ids, edges, segments = torch.arange(1, 5), torch.empty(2, 0, dtype=torch.long), ((0, 2), (2, 4))
        output = encoder(x, ids, segments, edges)
        torch.testing.assert_close(output, encoder(changed, ids, segments, edges))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for band in encoder.wavelet.bands:
            self.assertGreater(band[0].weight.grad.abs().sum().item(), 0)

    def test_queue_wavelet_uses_only_the_released_history(self):
        observation, p = day(24), panel()
        profile = np.zeros((2016, 2, 3), np.float32)
        history = features_v2(slice_day(observation, np.arange(12)), p, profile)
        model = TrafficQueueModel(features=31, node_count=2, width=8, blocks=1, heads=2,
                                  dropout=0, backend="scan", wavelet_levels=2, adaptive_graph_k=2,
                                  wavelet_input="signals").eval()
        ids, edges = torch.tensor([1, 2]), torch.tensor(p.edges)
        expected = model(torch.tensor(history[None]), ids, p.segments, edges)
        observation.speed[12:] = 999
        observation.flow_lane[12:] = 999
        observation.occupancy[12:] = 999
        actual = model(torch.tensor(features_v2(slice_day(observation, np.arange(12)), p, profile)[None]), ids, p.segments, edges)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)

    def test_regime_loss_is_macro_and_accumulation_keeps_its_gradient(self):
        archive = {"2030-06-01": np.array([0, 2, 24, 26]), "2030-06-02": np.array([1]),
                   "2030-06-03": np.array([0, 2]), "2031-01-01": np.arange(999)}
        targets = SimpleNamespace(archive=archive, stations=["s1", "s2"],
                                  regimes=dict(zip(archive, ["R1", "R2", "R3", "R3"])))
        files = {date: Path(date) for date in archive}
        item = dict(targets=targets, fit_paths=list(files.values())[:3], release=Path("unused"), panel=panel())
        with patch("learning.file_index", return_value=files):
            weights = fit_regime_weights(item)
        self.assertEqual(weights, {"R1": 1 / 3, "R2": 4 / 3, "R3": 2 / 3})
        prediction = torch.tensor([1., 1., 2., 3.])[:, None, None, None].expand(4, 2, 2, 2).clone().requires_grad_()
        mask = torch.zeros(4, 2, 2, dtype=torch.bool)
        mask[:2, 0] = True
        mask[2, 0, 0] = True
        mask[3, 0] = True
        batch = dict(truth=torch.zeros_like(prediction), mask=mask,
                     regime_weight=torch.tensor([weights[r] for r in ("R1", "R1", "R2", "R3")]))
        args = SimpleNamespace(lambda_n=0, lambda_dn=0, lambda_fd=0)
        full = state_loss(prediction, batch, None, None, None, args)
        self.assertAlmostEqual(full.item(), (1 + 4 + 9) / 3, places=5)
        expected = torch.autograd.grad(full, prediction, retain_graph=True)[0]
        parts = sum(state_loss(prediction[a:b], {k: v[a:b] for k, v in batch.items()}, None, None, None, args, normalizer=4)
                    for a, b in ((0, 1), (1, 4)))
        torch.testing.assert_close(full, parts)
        torch.testing.assert_close(expected, torch.autograd.grad(parts, prediction)[0])


if __name__ == "__main__":
    unittest.main()
