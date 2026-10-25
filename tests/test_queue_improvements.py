"""Queue origin visibility, paired sampling, feature masks, and checkpoint contracts."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import FEATURE_NAMES, queue_feature_names, queue_features, slice_day
from fast_model import TrafficQueueModel
from learning import (check_model_requests, model_from_checkpoint, predict_queue_v2, queue_examples, run_period,
                      schedule, masked_file_index)
from run import main
from test_v2 import day, panel, release_fixture


def options(**overrides):
    values = dict(history_coverage=0.5, future_coverage=0.5, seed=42,
                  windows_per_day=300, normal_windows=300, window_gap_minutes=0,
                  label_temperature=0, uncertainty_margin=0,
                  queue_origin=False, queue_changes=False, queue_resample=False,
                  queue_condition_gap=False, queue_validation_spread=False)
    return SimpleNamespace(**(values | overrides))


class QueueImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_origin_reads_masked_t_and_never_future_or_unmasked_t(self):
        truth = day(72)
        truth.speed[13:17, 0] = 1.6
        masked = copy.deepcopy(truth)
        masked.speed[12] = [1.2, np.nan]
        masked.flow_lane[12] = np.nan
        context = dict(panel=panel(), release=Path("unused"), profile=np.zeros((2016, 2, 3), np.float32),
                       origin_paths={"2030-06-01": Path("masked")})
        args = options(queue_origin=True, queue_changes=True)

        def sample():
            with patch("learning.StationDay.load", side_effect=lambda path, _: masked if path == Path("masked") else truth), \
                    patch("learning.ramp_features", return_value=np.zeros((72, 2, 4), np.float32)):
                return next(row for row in queue_examples(context, Path("truth"), args, "fit") if row["origin"] == 12)

        original = sample()
        self.assertEqual(original["x"].shape, (13, 2, 44))
        self.assertAlmostEqual(original["x"][-1, 0, 0], 1.2)
        self.assertEqual(original["x"][-1, 0, 11], 0)
        self.assertAlmostEqual(original["x"][-1, 0, 15], 1 / 72)
        truth.speed[12:] = 20
        masked.speed[13:] = 90
        masked.flow_lane[13:] = 80
        changed = sample()
        np.testing.assert_array_equal(original["x"], changed["x"])
        self.assertFalse(np.array_equal(original["truth"], changed["truth"]))
        masked.speed[12] = np.nan
        missing = sample()["x"]
        self.assertEqual(missing[-1, 0, 10], 0)
        self.assertEqual(missing[-1, 0, 0], 0)
        self.assertAlmostEqual(missing[-1, 0, 14], 1 / 72)

    def test_change_features_mask_invalid_pairs_and_use_only_supplied_past(self):
        observation = day(12)
        observation.speed[:, 0] = np.arange(12)
        observation.speed[:, 1] = np.nan
        observation.speed[4, 0] = np.nan
        prior = np.zeros((2016, 2, 3), np.float32)
        x = queue_features(observation, panel(), prior, changes=True)
        names = queue_feature_names(True)
        self.assertEqual(x.shape[-1], len(names))
        delta = names.index("speed_delta_5m")
        valid = names.index("speed_valid_5m")
        self.assertEqual(x[3, 0, delta], 1)
        self.assertEqual(x[4, 0, delta], 0)
        self.assertEqual(x[5, 0, valid], 0)
        self.assertTrue((x[:, 1, len(FEATURE_NAMES):] == 0).all())
        prefix = queue_features(slice_day(observation, np.arange(8)), panel(), prior, changes=True)
        np.testing.assert_array_equal(prefix[..., 31:], x[:8, ..., 31:])
        self.assertTrue(np.isfinite(x).all())

    def test_resampling_is_epoch_specific_only_for_fit_and_gap_is_per_condition(self):
        observation = day(288)
        for start in (40, 110, 190):
            observation.speed[start:start + 15, 0] = 1
        context = dict(panel=panel(), release=Path("unused"), profile=np.zeros((2016, 2, 3), np.float32))
        args = options(queue_resample=True, queue_condition_gap=True, queue_validation_spread=True,
                       windows_per_day=20, normal_windows=20, window_gap_minutes=120)
        with patch("learning.StationDay.load", return_value=observation), \
                patch("learning.ramp_features", return_value=np.zeros((288, 2, 4), np.float32)):
            keys = lambda period, epoch: [(s["condition"], s["origin"]) for s in queue_examples(context, Path("day"), args, period, epoch)]
            first, second = keys("fit", 0), keys("fit", 1)
            self.assertNotEqual(first, second)
            self.assertEqual(first, keys("fit", 0))
            self.assertEqual(keys("select", 0), keys("select", 9))
            args.queue_resample = False
            self.assertEqual(keys("fit", 0), keys("fit", 9))
        for kind in ("onset", "ongoing", "normal"):
            origins = sorted(origin for condition, origin in first if condition == kind)
            self.assertTrue(all((right - left) * 5 >= 120 for left, right in zip(origins, origins[1:])))
        self.assertTrue(any(a != b and abs(x - y) * 5 < 120 for a, x in first for b, y in first))

    def test_validation_spreads_dates_keeps_one_per_day_and_fixed_window_signature(self):
        paths = [Path(f"2031-01-{day:02d}.parquet") for day in range(1, 32)]
        args = options(validation_windows=3, release_root=Path("unused"), max_days=None, batch_size=2, panel_batch=[],
                       update_windows=None, use_bf16=False, bidirectional=True,
                       model_config={"backend": "scan"}, queue_validation_spread=True)
        context = {"P": {"panel": panel()}}
        vocabulary = {"P|A": 1, "P|B": 2}
        model = TrafficQueueModel(features=31, node_count=2, width=8, blocks=1, heads=2, dropout=0, backend="scan")

        def prepare(context, items, args, task, period, epoch):
            for name, path in items:
                samples = []
                for kind in ("onset", "ongoing", "normal"):
                    for origin in (12, 18):
                        target = np.ones((6, 2), np.float32)
                        samples.append(dict(x=np.zeros((12, 2, 31), np.float32), truth=target, soft=target,
                                            speed=target, mask=target.astype(bool), weight=target, baseline=target.astype(bool),
                                            date=path.stem, condition=kind, origin=origin))
                yield name, path, samples, 0

        args.normal_weight, args.soft_iou_weight, args.speed_weight = 0.2, 0.2, 0.1
        with patch("learning.period_files", return_value=paths), patch("learning.prepared_days", side_effect=prepare):
            order = schedule(context, "select", args, 0)
            self.assertEqual([path.stem for _, path in order[:3]], ["2031-01-01", "2031-01-16", "2031-01-31"])
            report, _ = run_period(model, context, vocabulary, args, "queue", "select", 0, "cpu")
            again, _ = run_period(model, context, vocabulary, args, "queue", "select", 9, "cpu")
        self.assertEqual(report["windows"], 9)
        self.assertEqual(report["window_signature"], again["window_signature"])
        self.assertEqual({row["date"] for row in report["daily_records"]}, {"2031-01-01", "2031-01-16", "2031-01-31"})

    def test_pooled_head_gradients_and_legacy_checkpoint_load(self):
        config = dict(features=31, node_count=2, width=8, blocks=1, heads=2, dropout=0, backend="scan")
        legacy = TrafficQueueModel(**config).eval()
        restored = model_from_checkpoint(dict(format_version=2, task="queue", feature_names=list(FEATURE_NAMES),
                                              model_config=config, weights=legacy.state_dict()), "cpu").eval()
        with self.assertRaisesRegex(ValueError, "new checkpoint"):
            check_model_requests(SimpleNamespace(graph_mode="mean", queue_head="pooled-mlp"), config)
        x, ids, edges = torch.rand(2, 12, 2, 31), torch.tensor([1, 2]), torch.empty(2, 0, dtype=torch.long)
        for original, loaded in zip(legacy(x, ids, ((0, 2),), edges), restored(x, ids, ((0, 2),), edges)):
            torch.testing.assert_close(original, loaded, rtol=0, atol=0)
        model = TrafficQueueModel(**(config | dict(features=44)), queue_head="pooled-mlp", queue_changes=True)
        logits, speed = model(torch.rand(2, 13, 2, 44), ids, ((0, 2),), edges)
        self.assertEqual(logits.shape, (2, 6, 2))
        (logits.square().mean() + speed.square().mean()).backward()
        for parameter in (model.encoder.input.weight, model.head[0].weight, model.head[-1].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_resume_matches_continuous_and_prediction_preserves_keys_and_future_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            full, staged = Path(directory) / "full.pt", Path(directory) / "staged.pt"
            common = ["run.py", "train-queue", "--release-root", str(root), "--backend", "scan",
                      "--width", "8", "--heads", "2", "--blocks", "1", "--precision", "fp32", "--workers", "0",
                      "--max-days", "2", "--validation-windows", "2", "--queue-validation-spread",
                      "--queue-origin", "--queue-resample", "--queue-condition-gap", "--queue-changes", "--queue-head", "pooled-mlp"]
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for path, epochs, extra in ((full, "2", []), (staged, "1", []),
                                            (staged, "2", ["--resume", str(staged.with_name("staged.last.pt"))])):
                    with patch.object(sys, "argv", common + ["--checkpoint", str(path), "--epochs", epochs] + extra):
                        main()
            complete = torch.load(full.with_name("full.last.pt"), weights_only=True)
            resumed = torch.load(staged.with_name("staged.last.pt"), weights_only=True)
            for key in complete["weights"]:
                torch.testing.assert_close(complete["weights"][key], resumed["weights"][key], rtol=0, atol=0)
            self.assertEqual(len(resumed["feature_names"]), 44)
            folder = root / "task2/P/validation"
            folder.mkdir(parents=True)
            times = pd.date_range("2031-03-01", periods=24, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            rows = pd.DataFrame([dict(window_id="w", timestamp=t, link_id=link, station_id=f"s{i + 1}",
                                      speed_kmh=90, flow_vph=1200, occupancy=0.1, pct_observed=100, is_score_eligible=True)
                                 for t in times for i, link in enumerate(["A", "B"])])
            rows[rows.timestamp.isin(times[:12])].drop(columns="station_id").to_parquet(folder / "window_history.parquet", index=False)
            pd.DataFrame(dict(window_id=["w"], forecast_origin=[times[12]])).to_csv(folder / "window_index.csv", index=False)
            template = pd.DataFrame([dict(window_id="w", timestamp=t, link_id=link) for t in times[13:19] for link in ["A", "B"]])
            template.to_csv(folder / "sample_submission_queue.csv", index=False)
            masked_path = root / "corridors/P/validation/mainline_states_masked/day_2031_03_01.parquet"
            masked_path.parent.mkdir(parents=True)
            rows.loc[rows.timestamp.eq(times[12]), "speed_kmh"] = 35
            rows.drop(columns="window_id").to_parquet(masked_path, index=False)
            model = model_from_checkpoint(resumed, "cpu").eval()
            captured = []
            hook = model.encoder.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
            output = Path(directory) / "queue.csv"
            predict_queue_v2(model, resumed, root, "validation", output)
            before = pd.read_csv(output)
            pd.testing.assert_frame_equal(before[template.columns], template)
            self.assertTrue(before.queue_pred.isin([0, 1]).all())
            self.assertEqual(tuple(captured[0].shape), (1, 13, 2, 44))
            torch.testing.assert_close(captured[0][0, -1, :, 0], torch.full((2,), 35 / 25))
            rows.loc[rows.timestamp.gt(times[12]), "speed_kmh"] = 1
            rows.drop(columns="window_id").to_parquet(masked_path, index=False)
            predict_queue_v2(model, resumed, root, "validation", output)
            torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
            pd.testing.assert_frame_equal(before, pd.read_csv(output))
            hook.remove()
            # A resume must also reject changes to the released fit-time origin source.
            source = next((root / "corridors/P/train/mainline_states_masked").glob("**/*2030_06_01.parquet"))
            frame = pd.read_parquet(source)
            frame.loc[0, "occupancy"] = 0.9
            frame.to_parquet(source, index=False)
            with patch.object(sys, "argv", common + ["--checkpoint", str(staged), "--epochs", "3", "--resume", str(staged.with_name("staged.last.pt"))]), \
                    patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, "origin dataset changed"):
                main()
            masked_file_index.cache_clear()

    def test_state_transfer_accepts_extended_queue_features_and_evaluation_keeps_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            state, queue = Path(directory) / "state.pt", Path(directory) / "queue.pt"
            common = ["--release-root", str(root), "--backend", "scan", "--width", "8", "--heads", "2",
                      "--blocks", "1", "--precision", "fp32", "--workers", "0", "--max-days", "1", "--epochs", "1"]
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for command, path, extra in (("train-state", state, []), ("train-queue", queue,
                        ["--init-state", str(state), "--queue-changes", "--queue-head", "pooled-mlp", "--queue-origin"])):
                    with patch.object(sys, "argv", ["run.py", command, "--checkpoint", str(path)] + common + extra):
                        main()
                checkpoint = torch.load(queue, weights_only=True)
                checkpoint["threshold"] = 0.37
                torch.save(checkpoint, queue)
                report = Path(directory) / "evaluation.json"
                with patch.object(sys, "argv", ["run.py", "evaluate", "--release-root", str(root), "--checkpoint", str(queue),
                                               "--period", "holdout", "--validation-windows", "3", "--queue-validation-spread",
                                               "--output", str(report)]):
                    main()
            import json
            result = json.loads(report.read_text())
            self.assertEqual(result["threshold"], 0.37)
            self.assertTrue(result["validation_spread"])
            self.assertEqual(checkpoint["weights"]["encoder.input.weight"].shape, (8, 44))


if __name__ == "__main__":
    unittest.main()
