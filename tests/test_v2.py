"""Contracts for real features, exact targets, physical units, and restartable training."""
import contextlib
import io
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import (FEATURE_NAMES, STATE_CONTEXT_NAMES, PanelInfo, StationDay, TargetIndex, features_v2,
                  ramp_features, slice_day)
from fast_model import GraphResidual, MambaSpatial, TrafficEncoder, TrafficQueueModel, TrafficStateModel
from learning import (file_index, fit_profile, physics_loss, queue_examples,
                      state_examples, state_metrics, model_from_checkpoint,
                      predict_state_v2, predict_queue_v2, queue_loss, run_period, check_model_requests)
from run import main, solve_odme


def panel():
    return PanelInfo("P", ["A", "B"], ((0, 2),), {"A": 0, "B": 1},
                     np.full(2, 100, np.float32), np.full(2, 2, np.float32),
                     np.ones(2, np.float32), np.full(2, 60, np.float32), {},
                     np.full(2, 6000, np.float32), np.full(2, 400, np.float32),
                     np.array([0, 1], np.float32), np.array([[0], [1]]))


def day(steps=24):
    times = pd.date_range("2030-06-01", periods=steps, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    return StationDay(times, ["s1", "s2"], ["A", "A"], np.array([0, 0]),
                      np.full(2, 2, np.float32), np.full((steps, 2), 4, np.float32),
                      np.ones((steps, 2), np.float32), np.ones((steps, 2), bool),
                      np.full((steps, 2), 0.1, np.float32), np.ones((steps, 2), np.float32))


def release_fixture(root):
    network = root / "corridors/P/network"
    network.mkdir(parents=True)
    pd.DataFrame({"link_id": ["A", "B"], "incoming_link_ids": ["", "A"],
                  "outgoing_link_ids": ["B", ""], "order_index": [0, 1]}).to_csv(network / "lwr_mainline_topology.csv", index=False)
    pd.DataFrame({"link_id": ["A", "B"], "free_speed_kmh": [100, 100],
                  "lanes": [2, 2], "length_km": [1, 1], "capacity_vph": [6000, 6000],
                  "k_jam": [400, 400]}).to_csv(network / "links.csv", index=False)
    pd.DataFrame({"link_id": ["A", "B"], "station_id": ["s1", "s2"],
                  "lanes": [2, 2], "v_cut": [60, 60]}).to_csv(network / "fd_parameters.csv", index=False)
    template = []
    for date, regime in [("2030-06-01", "R1"), ("2031-01-01", "R1"),
                         ("2031-01-02", "R2"), ("2031-01-03", "R3"), ("2031-02-01", "R1")]:
        times = pd.date_range(date, periods=24, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        rows = [{"timestamp": timestamp, "station_id": f"s{i+1}", "link_id": link,
                 "speed_kmh": 100 - 15 * (slot % 3), "flow_vph": 1200 + slot * 10,
                 "occupancy": 0.1, "pct_observed": 100, "is_score_eligible": True}
                for slot, timestamp in enumerate(times) for i, link in enumerate(["A", "B"])]
        original = pd.DataFrame(rows)
        name = f"synthetic_mainline_{date.replace('-', '_')}.parquet"
        folder = root / "corridors/P/train/mainline_states"
        folder.mkdir(parents=True, exist_ok=True)
        original.to_parquet(folder / name, index=False)
        masked = original.copy()
        target = (masked.station_id == "s1") & masked.timestamp.isin([times[6], times[19]])
        for row in masked[target].itertuples(index=False):
            template.append({"panel": "P", "timestamp": row.timestamp, "station_id": row.station_id,
                             "link_id": row.link_id, "mask_regime": regime})
        masked.loc[target, ["speed_kmh", "flow_vph"]] = np.nan
        masked.loc[0, ["speed_kmh", "flow_vph"]] = np.nan  # Blank buffer, outside the template.
        folder = root / f"corridors/P/train/mainline_states_masked/mask_regime={regime}"
        folder.mkdir(parents=True, exist_ok=True)
        masked.to_parquet(folder / name, index=False)
    folder = root / "task1/P/train"
    folder.mkdir(parents=True)
    pd.DataFrame(template).to_csv(folder / "sample_submission_state.csv", index=False)


class V2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("flock"), "Launcher lock requires WSL/Linux flock")
    def test_launcher_blocks_duplicate_runs_and_holds_lock_through_both_tasks(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "train_wsl.sh", root / "train_wsl.sh")
            (root / "data").mkdir()
            (root / "data/release_root.txt").write_text("unused-test-release\n")
            fake = root / ".venv/bin/python"
            fake.parent.mkdir(parents=True)
            fake.write_text("#!" + sys.executable + "\n"
                            "import fcntl, os, sys\nfrom pathlib import Path\n"
                            "folder = Path(os.environ['CHECKPOINT_DIR'])\n"
                            "with (folder / '.train.lock').open('a') as lock:\n"
                            "    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                            "    except BlockingIOError: pass\n"
                            "    else: sys.exit(99)\n"
                            "with (folder / 'calls.txt').open('a') as calls:\n"
                            "    calls.write(sys.argv[3] + '\\n')\n")
            fake.chmod(0o755)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            env = dict(os.environ, CHECKPOINT_DIR=str(checkpoints))
            with (checkpoints / ".train.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = subprocess.run(["bash", str(root / "train_wsl.sh"), "--graph"], env=env,
                                         capture_output=True, text=True, timeout=10)
                self.assertEqual(blocked.returncode, 2)
                self.assertIn("already running", blocked.stderr)
                self.assertFalse((checkpoints / "calls.txt").exists())
            completed = subprocess.run(["bash", str(root / "train_wsl.sh"), "--graph"], env=env,
                                       capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((checkpoints / "calls.txt").read_text().splitlines(), ["train-state", "train-queue"])

    def test_channel_masks_quality_and_causal_history(self):
        observation = day()
        observation.speed[:3] = np.nan
        observation.occupancy[1] = np.nan
        prior = np.zeros((2016, 2, 3), np.float32)
        x = features_v2(slice_day(observation, np.arange(12)), panel(), prior)
        self.assertEqual(x.shape, (12, 2, 31))
        self.assertEqual(x[1, 0, FEATURE_NAMES.index("speed_observed")], 0)
        self.assertEqual(x[1, 0, FEATURE_NAMES.index("flow_observed")], 1)
        self.assertEqual(x[1, 0, FEATURE_NAMES.index("pct_observed")], 1)
        self.assertEqual(x[1, 0, 1], 1)
        observation.speed[12:] = 999
        observation.flow_lane[12:] = 999
        observation.occupancy[12:] = 999
        np.testing.assert_array_equal(x, features_v2(slice_day(observation, np.arange(12)), panel(), prior))

    def test_exact_template_fit_dates_and_ramp_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            p = PanelInfo.load(root, "P")
            files = file_index(root, "P")
            original = StationDay.load(files["2030-06-01"], p)
            profile = fit_profile(root, p, [files["2030-06-01"]])
            targets = TargetIndex(root, "P", original.station_ids)
            _, mask = targets.get(original)
            self.assertEqual(mask.sum(), 2)
            self.assertFalse(mask[0, 0])
            context = {"panel": p, "release": root, "profile": profile, "targets": targets,
                       "masked": {path.name: path for path in (root / "corridors/P/train/mainline_states_masked").glob("**/*.parquet")}}
            samples = state_examples(context, files["2030-06-01"])
            self.assertEqual(sum(s["mask"].sum() for s in samples), 2)
            # Selection observations cannot change a fit-only cache/profile.
            january = pd.read_parquet(files["2031-01-01"])
            january.speed_kmh = 999
            january.to_parquet(files["2031-01-01"], index=False)
            np.testing.assert_array_equal(profile, fit_profile(root, p, [files["2030-06-01"]]))
            p.ramp_map = {"ramp": (0, 0)}
            ramp = root / "corridors/P/train/ramp_states/synthetic_ramp_2030_06_01.parquet"
            ramp.parent.mkdir(parents=True)
            pd.DataFrame({"timestamp": [original.timestamps[0], original.timestamps[12]],
                          "ramp_link_id": ["ramp", "ramp"], "flow_vph": [120, 9999],
                          "pct_observed": [100, 100]}).to_parquet(ramp, index=False)
            values = ramp_features(root, p, "train", original.timestamps[:12])
            self.assertEqual(values[0, 0, 0], 120)
            self.assertEqual(values[1, 0, 2], 0)
            self.assertEqual(values[:, 0, 0].sum(), 120)
            targets.archive.close()

    def test_queue_any_station_and_origin_gap(self):
        observation = day()
        observation.speed[13, 0] = 1.6  # Mean 70km/h, but one station is below 60.
        args = SimpleNamespace(history_coverage=0.5, future_coverage=0.5, seed=42,
                               windows_per_day=20, normal_windows=20, window_gap_minutes=0,
                               label_temperature=0, uncertainty_margin=0)
        context = {"panel": panel(), "release": Path("unused"), "profile": np.zeros((2016, 2, 3), np.float32)}
        with patch("learning.StationDay.load", return_value=observation), patch("learning.ramp_features", return_value=np.zeros((24, 2, 4), np.float32)):
            samples = queue_examples(context, Path("day"), args, "fit")
        sample = next(s for s in samples if s["origin"] == 12)
        self.assertEqual(sample["condition"], "onset")
        self.assertEqual(sample["truth"][0, 0], 1)
        self.assertEqual(sample["x"][-1, 0, 0], observation.speed[11].mean())
        self.assertGreater(sample["x"][..., 1].sum(), 0)
        self.assertGreater(sample["x"][..., 9].sum(), 0)

    def test_physical_units_and_graph_direction(self):
        truth = torch.tensor([[[[4., 1.]], [[4., 1.]]]])
        prediction = truth.clone()
        prediction[..., 1] *= 2
        valid = torch.ones((1, 2, 1), dtype=torch.bool)
        n, dn, _ = physics_loss(prediction, truth, valid, valid, ~valid,
                                torch.tensor([0]), torch.tensor([2.]), panel())
        self.assertAlmostEqual(n.item(), 0.04, places=6)  # N: 12 -> 24, scale: 60 vehicles.
        self.assertEqual(dn.item(), 0)
        graph = GraphResidual(1)
        graph.up.weight.data.fill_(2)
        graph.down.weight.data.fill_(3)
        output = graph(torch.tensor([[[[5.], [7.], [11.]]]]), torch.tensor([[0], [1]]))
        torch.testing.assert_close(output, torch.tensor([[[[21.], [10.], [0.]]]]))

    def test_missing_embedding_ignores_placeholders_and_loads_legacy(self):
        config = dict(features=31, node_count=2, width=8, blocks=1, heads=2, dropout=0, backend="scan")
        encoder = TrafficEncoder(**config, missing_aware=True)
        x = torch.randn(1, 12, 2, 31)
        x[..., 10:13] = 1
        x[:, :6, 0, 10] = 0
        x[:, 6:, 1, 11:13] = 0
        changed = x.clone()
        changed[:, :6, 0, 0] = float("nan")
        changed[:, 6:, 1, 1] = 1e6
        changed[:, 6:, 1, 9] = -1e6
        torch.testing.assert_close(encoder.project_input(x), encoder.project_input(changed))
        x.requires_grad_()
        encoder.project_input(x).sum().backward()
        self.assertEqual(x.grad[:, :6, 0, 0].abs().sum().item(), 0)
        self.assertGreater(encoder.missing_embedding.grad.abs().sum().item(), 0)
        with torch.no_grad():
            encoder.input.weight.zero_()
            encoder.input.bias.zero_()
            encoder.missing_embedding.fill_(1)
        zero = torch.zeros(1, 1, 1, 31)
        zero[..., 10:13] = 1
        observed = encoder.project_input(zero)
        zero[..., 10] = 0
        torch.testing.assert_close(encoder.project_input(zero) - observed, torch.ones_like(observed))
        legacy = TrafficStateModel(**config)
        payload = dict(format_version=2, task="state", feature_names=list(FEATURE_NAMES),
                       model_config=config, weights=legacy.state_dict())
        restored = model_from_checkpoint(payload, "cpu")
        self.assertIsNone(restored.encoder.missing_embedding)
        self.assertEqual(set(restored.state_dict()), set(legacy.state_dict()))
        for options in (dict(graph=True, missing_aware=False, graph_mode="mean"),
                        dict(graph=False, missing_aware=True, graph_mode="mean"),
                        dict(graph=False, missing_aware=False, graph_mode="reliability")):
            with self.assertRaisesRegex(ValueError, "new checkpoint"):
                check_model_requests(SimpleNamespace(**options), config)

    def test_reliability_graph_suppresses_missing_and_low_quality_neighbors(self):
        graph = GraphResidual(1, "reliability")
        graph.up.weight.data.fill_(1)
        graph.down.weight.data.fill_(1)
        graph.mix_gate.bias.data.zero_()
        x = torch.tensor([[[[5.], [7.], [11.]]]], requires_grad=True)
        observation = torch.zeros(1, 1, 3, 31)
        observation[..., 0, 10:13] = 1
        observation[..., 0, 13] = 1
        edges = torch.tensor([[0], [1]])
        good = graph(x, edges, observation)
        torch.testing.assert_close(good, torch.tensor([[[[0.], [1.25], [0.]]]]))
        observation[..., 0, 13] = 0.1
        torch.testing.assert_close(graph(x, edges, observation), good * 0.1)
        observation[..., 0, 10:13] = 0
        torch.testing.assert_close(graph(x, edges, observation), torch.zeros_like(x))
        observation[..., 0, 10:13] = 1
        good.sum().backward()
        self.assertGreater(graph.edge_gate.weight.grad.abs().sum().item(), 0)
        self.assertGreater(graph.up.weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(graph(x, torch.empty(2, 0, dtype=torch.long), observation), torch.zeros_like(x))

    def test_update_windows_balance_panels_and_keep_partial_updates(self):
        class TinyState(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.2))

            def forward(self, x, context, mapping, *layout):
                return self.scale * x[..., :2]

        samples = []
        for i in range(26):
            mask = np.zeros((12, 2), bool)
            mask[:1 + i % 12] = True
            samples.append(dict(x=np.full((72, 2, 31), 1 + i / 26, np.float32),
                                context=np.zeros((72, 2, 3), np.float32),
                                truth=np.ones((12, 2, 2), np.float32), mask=mask, eligible=mask,
                                observed=~mask, station_link=np.array([0, 1]), lanes=np.ones(2, np.float32)))
        context = {name: {"panel": panel()} for name in ("P", "Q")}
        vocabulary = {f"{name}|{link}": i + 1 for i, (name, link) in enumerate(
            (name, link) for name in context for link in ("A", "B"))}
        weights = []
        for batches, requested, expected in [(["P=8", "Q=2"], 0, {"P": 4, "Q": 4}),
                                              (["P=2", "Q=8"], 0, {"P": 4, "Q": 4}),
                                              (["P=8", "Q=2"], None, {"P": 4, "Q": 13})]:
            args = SimpleNamespace(batch_size=0, panel_batch=batches, update_windows=requested,
                                   use_bf16=False, bidirectional=True, grad_clip=1e6,
                                   lambda_n=0, lambda_dn=0, lambda_fd=0)
            model = TinyState()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            prepared = [(name, Path("day.parquet"), samples, 0) for name in context]
            with patch("learning.schedule", return_value=list(context)), patch("learning.prepared_days", return_value=prepared):
                report, step = run_period(model, context, vocabulary, args, "state", "fit", 0, "cpu", optimizer)
            self.assertEqual(report["updates_by_panel"], expected)
            self.assertEqual(step, sum(expected.values()))
            self.assertEqual(report["windows"], 52)
            weights.append(model.scale.detach().clone())
        torch.testing.assert_close(weights[0], weights[1], rtol=1e-6, atol=1e-7)

    def test_queue_accumulation_preserves_condition_weights(self):
        args = SimpleNamespace(normal_weight=0.2, soft_iou_weight=0.2, speed_weight=0.1)
        conditions = ["onset", "onset", "onset", "ongoing", "normal"]
        counts = {kind: conditions.count(kind) for kind in set(conditions)}
        logits = torch.randn(5, 6, 2, requires_grad=True)
        speeds = torch.randn(5, 6, 2, requires_grad=True)
        mask = torch.rand(5, 6, 2) > 0.25
        truth = (torch.rand(5, 6, 2) > 0.5).float()
        batch = dict(weight=torch.rand(5, 6, 2), mask=mask, truth=truth, soft=truth, speed=torch.rand(5, 6, 2))
        full = queue_loss(logits, speeds, batch, conditions, args)
        expected = torch.autograd.grad(full, (logits, speeds), retain_graph=True)
        parts = sum(queue_loss(logits[start:stop], speeds[start:stop],
                               {key: value[start:stop] for key, value in batch.items()},
                               conditions[start:stop], args, counts)
                    for start, stop in ((0, 2), (2, 4), (4, 5)))
        torch.testing.assert_close(full, parts)
        actual = torch.autograd.grad(parts, (logits, speeds))
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)

    def test_sparse_odme_matches_nonnegative_reference(self):
        from scipy.optimize import nnls
        matrix = np.array([[1., 1., 0.], [0., 1., 1.]])
        counts, prior, lam = np.array([2., 0.1]), np.array([0., 4., 2.]), 0.05
        expected = nnls(np.vstack((matrix, np.sqrt(lam) * np.eye(3))),
                        np.r_[counts, np.sqrt(lam) * prior])[0]
        np.testing.assert_allclose(solve_odme(matrix, counts, prior, lam), expected, atol=1e-5)

    def test_macro_regimes_and_model_gradients(self):
        records = [{"panel": "P_N", "regime": r, "count": count,
                    "model_speed_sse": speed * count, "model_flow_sse": 0}
                   for r, count, speed in [("R1", 100, 0), ("R2", 1, 1), ("R3", 1, 0)]]
        self.assertAlmostEqual(state_metrics(records)["State"], (1 + 0.46 + 1) / 3)
        config = dict(features=31, node_count=2, width=8, blocks=1, heads=2, dropout=0, backend="scan", graph=True)
        x = torch.zeros(1, 12, 2, 31)
        ids, edges = torch.tensor([1, 2]), torch.tensor([[0], [1]])
        state = TrafficStateModel(**config)
        context = torch.ones(1, 12, 2, 3)
        output = state(x, context, torch.tensor([0, 1]), ids, ((0, 2),), edges)
        self.assertEqual(output.shape, (1, 12, 2, 2))
        output.square().sum().backward()
        self.assertIsNotNone(state.head.weight.grad)
        queue = TrafficQueueModel(**config)
        logits, speeds = queue(x, ids, ((0, 2),), edges)
        (logits.sum() + speeds.sum()).backward()
        self.assertEqual(logits.shape, (1, 6, 2))
        self.assertIsNotNone(queue.encoder.input.weight.grad)

    def test_resume_matches_continuous_training_and_state_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            full, staged = Path(directory) / "full.pt", Path(directory) / "staged.pt"
            common = ["run.py", "train-state", "--release-root", str(root), "--panel", "P",
                      "--backend", "scan", "--width", "8", "--blocks", "1", "--heads", "2",
                      "--workers", "0", "--cpu-threads", "1", "--batch-size", "2", "--dropout", "0.1"]
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for checkpoint, epochs, extra in [(full, "2", []), (staged, "1", []),
                                                   (staged, "2", ["--resume", str(staged.with_name("staged.last.pt"))])]:
                    with patch.object(sys, "argv", common + ["--checkpoint", str(checkpoint), "--epochs", epochs] + extra):
                        main()
                    if checkpoint == staged and epochs == "1":
                        saved = staged.with_name("staged.last.pt")
                        old = torch.load(saved, weights_only=True)
                        old["training_args"].pop("update_windows")
                        old["model_config"].pop("missing_aware")
                        old["model_config"].pop("graph_mode")
                        for key in ("station_decoder", "adaptive_graph_k", "wavelet_levels"):
                            old["model_config"].pop(key)
                        old["training_args"].pop("state_regime_balance")
                        old.pop("state_context_names")
                        torch.save(old, saved)
            left = torch.load(full.with_name("full.last.pt"), weights_only=True)
            right = torch.load(staged.with_name("staged.last.pt"), weights_only=True)
            self.assertEqual(right["epoch"], 2)
            self.assertEqual(left["global_step"], right["global_step"])
            for key in left["weights"]:
                torch.testing.assert_close(left["weights"][key], right["weights"][key], rtol=0, atol=0)
            model = model_from_checkpoint(right, "cpu").eval()
            prediction = Path(directory) / "predictions.csv"
            predict_state_v2(model, right, root, "train", prediction)
            result = pd.read_csv(prediction)
            self.assertEqual(len(result), 10)
            self.assertTrue(np.isfinite(result[["speed_kmh", "flow_vph"]]).all().all())

    def test_enhanced_checkpoint_resume_and_queue_transfer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            state, queue = Path(directory) / "state.pt", Path(directory) / "queue.pt"
            common = ["--release-root", str(root), "--panel", "P", "--backend", "scan",
                      "--width", "8", "--blocks", "1", "--heads", "2", "--dropout", "0",
                      "--workers", "0", "--batch-size", "1", "--update-windows", "0",
                      "--missing-aware", "--graph", "--graph-mode", "reliability",
                      "--station-decoder", "split", "--adaptive-graph-k", "2",
                      "--wavelet-levels", "2", "--wavelet-input", "signals", "--state-regime-balance"]
            full = Path(directory) / "full.pt"
            jobs = [("train-state", full, "2", []), ("train-state", state, "1", []),
                    ("train-state", state, "2", ["--resume", str(state.with_name("state.last.pt"))]),
                    ("train-queue", queue, "1", ["--init-state", str(state)])]
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for task, checkpoint, epochs, extra in jobs:
                    with patch.object(sys, "argv", ["run.py", task, "--checkpoint", str(checkpoint), "--epochs", epochs] + common + extra):
                        main()
            complete = torch.load(full.with_name("full.last.pt"), weights_only=True)
            resumed = torch.load(state.with_name("state.last.pt"), weights_only=True)
            for key in complete["weights"]:
                torch.testing.assert_close(complete["weights"][key], resumed["weights"][key], rtol=0, atol=0)
            transferred = torch.load(queue, weights_only=True)
            self.assertTrue(transferred["model_config"]["missing_aware"])
            self.assertEqual(transferred["model_config"]["graph_mode"], "reliability")
            self.assertEqual(transferred["model_config"]["station_decoder"], "split")
            self.assertEqual(transferred["model_config"]["adaptive_graph_k"], 2)
            self.assertEqual(transferred["model_config"]["wavelet_levels"], 2)
            self.assertEqual(transferred["model_config"]["wavelet_input"], "signals")
            self.assertEqual(transferred["training_args"]["update_windows"], 0)
            self.assertEqual(transferred["node_vocab"], resumed["node_vocab"])
            model_from_checkpoint(transferred, "cpu")
            prediction = Path(directory) / "enhanced-state.csv"
            predict_state_v2(model_from_checkpoint(resumed, "cpu").eval(), resumed, root, "train", prediction)
            self.assertTrue(np.isfinite(pd.read_csv(prediction)[["speed_kmh", "flow_vph"]]).all().all())
            incompatible = dict(resumed, state_context_names=["wrong"])
            with self.assertRaisesRegex(ValueError, "station context contract"):
                model_from_checkpoint(incompatible, "cpu")
            old_config = dict(resumed["model_config"], station_decoder="linear", adaptive_graph_k=0, wavelet_levels=0)
            for requested in (dict(station_decoder="gated"), dict(adaptive_graph_k=2), dict(wavelet_levels=2)):
                options = dict(graph=False, missing_aware=False, graph_mode="mean", **requested)
                with self.assertRaisesRegex(ValueError, "new checkpoint"):
                    check_model_requests(SimpleNamespace(**options), old_config)

    def test_queue_inference_keys_and_time_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "release"
            release_fixture(root)
            folder = root / "task2/P/validation"
            folder.mkdir(parents=True)
            times = pd.date_range("2031-03-01", periods=12, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            history = pd.DataFrame([{"window_id": "w", "timestamp": timestamp, "link_id": link,
                                     "speed_kmh": 90, "flow_vph": 1200, "occupancy": 0.1,
                                     "pct_observed": 100, "is_score_eligible": True}
                                    for timestamp in times for link in ["A", "B"]])
            history.to_parquet(folder / "window_history.parquet", index=False)
            pd.DataFrame({"window_id": ["w"], "forecast_origin": ["2031-03-01T01:00:00Z"]}).to_csv(folder / "window_index.csv", index=False)
            future = pd.date_range("2031-03-01T01:05:00Z", periods=6, freq="5min").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            template = pd.DataFrame([{"window_id": "w", "timestamp": t, "link_id": link}
                                     for t in future for link in ["A", "B"]])
            template.to_csv(folder / "sample_submission_queue.csv", index=False)
            model = TrafficQueueModel(features=31, node_count=2, width=8, blocks=1, heads=2,
                                       dropout=0, backend="scan", wavelet_levels=2, adaptive_graph_k=2).eval()
            checkpoint = {"trained_panels": ["P"], "profiles": {"P": torch.zeros(2016, 2, 3)},
                          "links": {"P": ["A", "B"]}, "node_vocab": {"P|A": 1, "P|B": 2},
                          "training_args": {"use_bf16": False}, "threshold": 0.5, "bidirectional": True}
            output = Path(directory) / "queue.csv"
            predict_queue_v2(model, checkpoint, root, "validation", output)
            result = pd.read_csv(output)
            pd.testing.assert_frame_equal(result[template.columns], template)
            self.assertTrue(result.queue_pred.isin([0, 1]).all())
            template.loc[0, "timestamp"] = "2031-03-01T01:00:00Z"
            template.to_csv(folder / "sample_submission_queue.csv", index=False)
            with self.assertRaisesRegex(ValueError, "time boundary"):
                predict_queue_v2(model, checkpoint, root, "validation", output)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm"), "Official Mamba component check requires CUDA and mamba-ssm")
    def test_official_mamba_component_padding_and_bf16(self):
        scan = MambaSpatial(32, "mamba2").cuda()
        x = torch.randn(1, 2, 8, 32, device="cuda", requires_grad=True)
        changed = x.detach().clone()
        changed[:, :, :3] += 100
        for reverse in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = scan(x, ((0, 3), (3, 8)), reverse)
                other = scan(changed, ((0, 3), (3, 8)), reverse)
                isolated = scan(x[:, :, 3:], ((0, 5),), reverse)
            torch.testing.assert_close(output[:, :, 3:], other[:, :, 3:])
            torch.testing.assert_close(output[:, :, 3:], isolated)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        features = torch.rand(1, 72, 8, 31, device="cuda")
        features[..., 10:13] = 1
        features[:, :36, :3, 10:13] = 0
        features[..., 13] = 0.8
        features[:, :36, :3, 0] = float("nan")
        nodes = torch.arange(1, 9, device="cuda")
        edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], device="cuda")
        config = dict(features=31, node_count=8, width=64, blocks=2, heads=4, dropout=0,
                      backend="mamba2", graph=True, graph_mode="reliability", missing_aware=True,
                      station_decoder="split", adaptive_graph_k=4, wavelet_levels=2, wavelet_input="signals")
        for cls in (TrafficStateModel, TrafficQueueModel):
            model = cls(**config).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if cls is TrafficStateModel:
                    result = model(features, torch.ones(1, 72, 8, len(STATE_CONTEXT_NAMES), device="cuda"),
                                   nodes - 1, nodes, ((0, 3), (3, 8)), edges)
                    loss = result.float().square().mean()
                else:
                    logits, speeds = model(features[:, -12:], nodes, ((0, 3), (3, 8)), edges)
                    loss = logits.float().square().mean() + speeds.float().square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None))
            self.assertIsNotNone(model.encoder.missing_embedding.grad)


if __name__ == "__main__":
    unittest.main()
