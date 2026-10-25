"""Single-notebook contracts and CPU end-to-end checks on a tiny release."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/task123_kaggle.ipynb"
COMMIT = "cdf888214fe8a54b73302f3988b764b74b3f30a4"


def notebook():
    return json.loads(NOTEBOOK.read_text())


def source(cell):
    return "".join(cell["source"])


def tagged(tag):
    return next(source(cell) for cell in notebook()["cells"] if tag in cell.get("metadata", {}).get("tags", []))


def release_fixture(root):
    network = root / "corridors/P_N/network"
    network.mkdir(parents=True)
    pd.DataFrame(dict(link_id=["l0", "l1"], lanes=[2, 2], length_km=[1., 1.],
                      free_speed_kmh=[100., 100.], capacity_vph=[6000., 6000.], k_jam=[400., 400.])).to_csv(
        network / "links.csv", index=False)
    pd.DataFrame(dict(link_id=["l0", "l1"], incoming_link_ids=["", "l0"], outgoing_link_ids=["l1", ""])).to_csv(
        network / "lwr_mainline_topology.csv", index=False)
    pd.DataFrame(dict(station_id=["001", "002"], link_id=["l0", "l1"], lanes=[2, 2],
                      free_speed_kmh=[100., 100.], capacity_vph=[6000., 6000.],
                      k_jam=[400., 400.], v_cut=[60., 60.])).to_csv(network / "fd_parameters.csv", index=False)
    pd.DataFrame(dict(path_id=["p1", "p2"], link_id=["l0", "l1"])).to_csv(network / "path_link_incidence.csv", index=False)
    dates = [f"2030-06-{i:02d}" for i in range(1, 7)]
    dates += [f"2031-01-{i:02d}" for i in (1, 2, 3, 16, 17, 18)]
    dates += [f"2031-02-{i:02d}" for i in (1, 2, 3)]
    keys = []
    for split, days in (("train", dates), ("validation", ["2031-03-01"]), ("private", ["2031-04-01"])):
        targets = []
        for number, date in enumerate(days):
            regime = ("R1", "R2", "R3")[number % 3]
            times = pd.date_range(date, periods=72, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
            rows = [dict(timestamp=t, station_id=station, link_id=link,
                         speed_kmh=(40 if station == "001" and (15 <= slot < 24 or 40 <= slot < 50) else 90) + slot % 3,
                         flow_vph=1200 + 10 * (slot % 5), occupancy=.1 + .01 * (slot % 3),
                         pct_observed=100, is_score_eligible=True)
                    for slot, t in enumerate(times) for station, link in (("001", "l0"), ("002", "l1"))]
            frame = pd.DataFrame(rows)
            chosen = frame.station_id.eq("001") & frame.timestamp.isin(times[25:29])
            target = frame.loc[chosen, ["timestamp", "station_id", "link_id"]].copy()
            target["panel"], target["mask_regime"] = "P_N", regime
            targets.append(target[["panel", "timestamp", "station_id", "link_id", "mask_regime"]])
            if split == "train":
                folder = root / "corridors/P_N/train/mainline_states"
                folder.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(folder / f"states_{date}.parquet", index=False)
            masked = frame.copy()
            masked.loc[chosen, ["speed_kmh", "flow_vph"]] = np.nan
            folder = root / f"corridors/P_N/{split}/mainline_states_masked/mask_regime={regime}"
            folder.mkdir(parents=True, exist_ok=True)
            masked.to_parquet(folder / f"states_{date}.parquet", index=False)
            if split != "train":
                window = f"{split}-window"
                folder = root / "task2/P_N" / split
                folder.mkdir(parents=True)
                history = frame[frame.timestamp.isin(times[:12])].drop(columns="station_id")
                history["window_id"] = window
                history.to_parquet(folder / "window_history.parquet", index=False)
                pd.DataFrame(dict(window_id=[window], forecast_origin=[times[12]])).to_csv(folder / "window_index.csv", index=False)
                future = pd.DataFrame([dict(window_id=window, timestamp=t, link_id=link)
                                       for t in times[13:19] for link in ("l0", "l1")])
                future.to_csv(folder / "sample_submission_queue.csv", index=False)
                keys.extend(dict(task="queue", **r) for r in future.to_dict("records"))
        folder = root / "task1/P_N" / split
        folder.mkdir(parents=True)
        state = pd.concat(targets, ignore_index=True)
        state.to_csv(folder / "sample_submission_state.csv", index=False)
        if split != "train":
            keys.extend(dict(task="state", **r) for r in state.to_dict("records"))
            folder = root / "task4/P_N" / split
            folder.mkdir(parents=True)
            odme = pd.DataFrame(dict(panel=["P_N"] * 2, departure_time=[f"{split}-AM"] * 2,
                                     path_id=["p1", "p2"], origin_zone=["O", "O"], destination_zone=["D", "E"]))
            odme.to_csv(folder / "sample_submission_path_flow.csv", index=False)
            odme[["departure_time", "path_id"]].assign(path_flow=[2., 2.]).to_csv(folder / "synthetic_weak_prior.csv", index=False)
            pd.DataFrame(dict(link_id=["l0", "l1"], count=[4., 3.])).to_csv(folder / "synthetic_link_counts.csv", index=False)
            keys.extend(dict(task="odme", **r) for r in odme.to_dict("records"))
    key = pd.DataFrame(keys)
    key.insert(0, "submission_id", np.arange(len(key)))
    key.to_csv(root / "submission_key.csv", index=False)
    return key


class Task123Notebook(unittest.TestCase):
    def test_embedded_sources_are_pinned_and_all_cells_parse(self):
        nb = notebook()
        self.assertEqual(nb["metadata"]["trafficflow"]["source_commit"], COMMIT)
        embedded = []
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(source(cell), filename=f"cell-{i}")
            if "embedded-source" in cell.get("metadata", {}).get("tags", []):
                call = tree.body[0].value
                name, code = (ast.literal_eval(arg) for arg in call.args)
                self.assertEqual(hashlib.sha256(code.encode()).hexdigest(), nb["metadata"]["trafficflow"]["source_sha256"][name])
                ast.parse(code)
                embedded.append(name)
        self.assertEqual(set(embedded), {"data", "fast_model", "learning", "historical_retrieval", "maginet", "run"})
        self.assertNotIn("google.colab", tagged("environment"))
        self.assertNotIn("git clone", tagged("environment"))

    def test_training_action_handles_independent_progress_and_early_stopping(self):
        tree = ast.parse(tagged("backend-check"))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "training_action")
        ns = {"torch": torch}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "training_action", "exec"), ns)
        action = ns["training_action"]
        with tempfile.TemporaryDirectory() as directory:
            best, last = Path(directory) / "best.pt", Path(directory) / "last.pt"
            self.assertEqual(action(best, last, 10, resume=True), "new")
            best.write_bytes(b"existing best checkpoint")
            with self.assertRaises(FileNotFoundError):
                action(best, last, 10, resume=True)
            saved = dict(format_version=2, task="queue", epoch=3, bad_epochs=0, training_args=dict(patience=5))
            torch.save(saved, last)
            before = best.read_bytes(), last.read_bytes()
            with self.assertRaises(FileExistsError):
                action(best, last, 10, resume=False)
            self.assertEqual(action(best, last, 10, resume=True), "resume")
            self.assertEqual(action(best, last, 3, resume=True), "complete")
            self.assertEqual(action(best, last, 2, resume=True), "complete")
            self.assertEqual((best.read_bytes(), last.read_bytes()), before)
            saved["bad_epochs"] = 5
            torch.save(saved, last)
            self.assertEqual(action(best, last, 10, resume=True), "complete")
            saved = dict(format_version=3, epoch=12, bad_epochs=8, settings=dict(patience=8, warmup=5, ramp=5))
            torch.save(saved, last)
            self.assertEqual(action(best, last, 50, resume=True), "complete")
            saved["epoch"] = 9
            torch.save(saved, last)
            self.assertEqual(action(best, last, 50, resume=True), "resume")
            best.unlink()
            with self.assertRaises(FileNotFoundError):
                action(best, last, 50, resume=True)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Notebook runtime needs pyarrow")
    def test_resume_after_state_before_queue_and_finished_state_is_preserved(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "input/kaggle_public"
            release_fixture(release)
            ns = {"display": lambda value: None}
            exec(tagged("configuration"), ns)
            ns.update(RELEASE_ROOT=str(release), WORK_ROOT=root / "working/trafficflow_task123",
                      INSTALL_MISSING_PACKAGES=False, CPU_THREADS=1, USE_RETRIEVAL=False, RESUME=True)
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(io.StringIO()):
                for cell in notebook()["cells"]:
                    tags = cell.get("metadata", {}).get("tags", [])
                    if cell["cell_type"] == "code" and ("embedded-source" in tags or
                            any(tag in tags for tag in ("environment", "module-loader", "backend-check", "train-state"))):
                        exec(source(cell), ns)
                paths = [ns["MAGINET_CHECKPOINT"], ns["MAGINET_DIR"] / "full/last_maginet.pt"]
                before = [path.read_bytes() for path in paths]
                self.assertFalse(ns["QUEUE_CHECKPOINT"].exists())
                # A complete State must be reused while an unstarted Queue begins from scratch.
                exec(tagged("train-state"), ns)
                exec(tagged("train-queue"), ns)
                self.assertEqual([path.read_bytes() for path in paths], before)
                queue_last = ns["QUEUE_DIR"] / "queue.last.pt"
                self.assertEqual(torch.load(queue_last, weights_only=True)["epoch"], 2)
                # Independent total budgets: only Queue continues from 2 to 3 epochs.
                ns.update(RUN_MODE="train", MAGINET_EPOCHS=2, QUEUE_EPOCHS=3)
                exec(tagged("train-state"), ns)
                exec(tagged("train-queue"), ns)
                self.assertEqual([path.read_bytes() for path in paths], before)
                self.assertEqual(torch.load(queue_last, weights_only=True)["epoch"], 3)
                # Recover interruption after the final checkpoint but before alpha selection.
                alpha = ns["MAGINET_DIR"] / "selected_alpha.json"
                alpha.unlink()
                exec(tagged("evaluate"), ns)
                self.assertTrue(alpha.is_file())
                self.assertEqual(json.loads(alpha.read_text())["checkpoint_sha256"], hashlib.sha256(paths[0].read_bytes()).hexdigest())
                self.assertEqual([path.read_bytes() for path in paths], before)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Notebook runtime needs pyarrow")
    def test_notebook_training_resume_retrieval_predictions_and_strict_merge(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "input/competition/kaggle_public"
            key = release_fixture(release)
            before = {str(p.relative_to(release)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in release.rglob("*") if p.is_file()}
            ns = {"display": lambda value: None}
            exec(compile(tagged("configuration"), "configuration", "exec"), ns)
            ns.update(RELEASE_ROOT=str(release), WORK_ROOT=root / "working/trafficflow_task123",
                      INSTALL_MISSING_PACKAGES=False, CPU_THREADS=1, RUN_FEBRUARY=True)
            log = io.StringIO()
            with patch("torch.cuda.is_available", return_value=False), contextlib.redirect_stdout(log):
                for cell in notebook()["cells"]:
                    tags = cell.get("metadata", {}).get("tags", [])
                    if cell["cell_type"] == "code" and "configuration" not in tags:
                        exec(compile(source(cell), str(tags), "exec"), ns)
                self.assertTrue((ns["RELEASE"].parent / "processed_v2").is_dir())
                self.assertFalse((release.parent / "processed_v2").exists())
                self.assertTrue((ns["MAGINET_DIR"] / "retrieval/selected_retrieval.json").is_file())
                self.assertTrue((ns["QUEUE_DIR"] / "february.json").is_file())
                self.assertFalse(ns["SUBMISSION_PATH"].exists())
                for path in ns["STATE_FILES"] + ns["QUEUE_FILES"]:
                    frame = pd.read_csv(path, dtype=str)
                    self.assertGreater(len(frame), 0)
                    if "queue_pred" in frame:
                        self.assertTrue(pd.to_numeric(frame.queue_pred).isin([0, 1]).all())
                saved = torch.load(ns["QUEUE_CHECKPOINT"], map_location="cpu", weights_only=True)
                self.assertEqual(len(saved["feature_names"]), 44)
                self.assertEqual(saved["model_config"]["queue_head"], "pooled-mlp")
                self.assertTrue(saved["training_args"]["queue_origin"])

                # Same notebook cells resume both models, then generate fresh outputs.
                ns.update(RESUME=True, SMOKE_EPOCHS=3)
                for tag in ("train-state", "train-queue", "evaluate", "predict"):
                    exec(compile(tagged(tag), tag, "exec"), ns)
                for path in (ns["MAGINET_DIR"] / "full/last_maginet.pt", ns["QUEUE_DIR"] / "queue.last.pt"):
                    self.assertEqual(torch.load(path, map_location="cpu", weights_only=True)["epoch"], 3)

                # The complete merger runs with both split outputs plus the unchanged ODME baseline.
                odme_files = []
                for split in ("validation", "private"):
                    path = root / f"odme-{split}.csv"
                    ns["runner"].predict_odme(ns["RELEASE"], split, path, .05, 0, .1)
                    odme_files.append(path)
                files = dict(state=ns["STATE_FILES"], queue=ns["QUEUE_FILES"], odme=odme_files)
                output = ns["SUBMISSION_PATH"]
                count = ns["merge_submission"](files, release / "submission_key.csv", output, chunksize=3)
                self.assertEqual(count, len(key))
                actual = pd.read_csv(output)
                self.assertEqual(list(actual.columns), ns["OUTPUT_COLUMNS"])
                pd.testing.assert_frame_equal(actual[["submission_id", "task"]], key[["submission_id", "task"]])
                self.assertTrue(np.isfinite(actual[ns["OUTPUT_COLUMNS"][2:]].to_numpy()).all())
                good = output.read_bytes()
                state_path = files["state"][0]
                original = pd.read_csv(state_path, dtype=str)
                for label, bad in (("missing", original.iloc[1:]),
                                   ("duplicate", pd.concat([original, original.iloc[:1]])),
                                   ("negative", original.assign(speed_kmh=-1)),
                                   ("nonfinite", original.assign(flow_vph="nan")),
                                   ("outside", pd.concat([original, original.iloc[:1].assign(station_id="outside")]))):
                    with self.subTest(label=label):
                        bad.to_csv(state_path, index=False)
                        with self.assertRaises(ValueError):
                            ns["merge_submission"](files, release / "submission_key.csv", output, chunksize=3)
                        self.assertEqual(output.read_bytes(), good)
                        self.assertFalse(output.with_suffix(".pending.csv").exists())
                original.to_csv(state_path, index=False)
                qpath = files["queue"][0]
                queue = pd.read_csv(qpath, dtype=str)
                queue.assign(queue_pred="0.5").to_csv(qpath, index=False)
                with self.assertRaisesRegex(ValueError, "예측값"):
                    ns["merge_submission"](files, release / "submission_key.csv", output)
                queue.to_csv(qpath, index=False)
                # Full-coverage notebook integration uses the provided Task 4 CSVs.
                ns.update(RUN_MODE="predict", TASK4_FILES=odme_files)
                exec(compile(tagged("merge"), "merge", "exec"), ns)
                self.assertEqual(len(pd.read_csv(output)), len(key))

                # Restore a saved Output into a new writable workspace and predict without evaluating.
                previous = root / "previous-output"
                shutil.copytree(ns["RESULTS"], previous / "reports/full")
                restored = {"display": lambda value: None}
                exec(compile(tagged("configuration"), "configuration", "exec"), restored)
                restored.update(RUN_MODE="predict", RELEASE_ROOT=str(release), WORK_ROOT=root / "restored/trafficflow_task123",
                                RESTORE_ROOT=previous, QUEUE_BACKEND="scan", TASK4_FILES=odme_files,
                                INSTALL_MISSING_PACKAGES=False, CPU_THREADS=1)
                selection = previous / "reports/full/maginet" / ns["MAGINET_EXPERIMENT"] / "retrieval/selected_retrieval.json"
                selected_before = selection.read_bytes()
                for cell in notebook()["cells"]:
                    tags = cell.get("metadata", {}).get("tags", [])
                    if cell["cell_type"] == "code" and "configuration" not in tags:
                        exec(compile(source(cell), str(tags), "exec"), restored)
                self.assertEqual(len(pd.read_csv(restored["SUBMISSION_PATH"])), len(key))
                self.assertEqual(restored["QUEUE_CHECKPOINT"].read_bytes(), ns["QUEUE_CHECKPOINT"].read_bytes())
                self.assertEqual(restored["RETRIEVAL_SELECTION"].read_bytes(), selected_before)

                # Run every cell with no January originals or train masked data, without retrieval.
                inference = root / "inference/kaggle_public"
                shutil.copytree(release, inference)
                originals = sorted((inference / "corridors/P_N/train/mainline_states").glob("*.parquet"))
                for path in originals[1:]:
                    path.unlink()
                shutil.rmtree(inference / "corridors/P_N/train/mainline_states_masked")
                portable = {"display": lambda value: None}
                exec(tagged("configuration"), portable)
                portable.update(RUN_MODE="predict", RELEASE_ROOT=str(inference), WORK_ROOT=root / "portable/trafficflow_task123",
                                RESTORE_ROOT=previous, QUEUE_BACKEND="scan", TASK4_FILES=odme_files,
                                INSTALL_MISSING_PACKAGES=False, CPU_THREADS=1, USE_RETRIEVAL=False)
                for cell in notebook()["cells"]:
                    if cell["cell_type"] == "code" and "configuration" not in cell.get("metadata", {}).get("tags", []):
                        exec(source(cell), portable)
                self.assertEqual(len(pd.read_csv(portable["SUBMISSION_PATH"])), len(key))
                self.assertEqual(portable["RETRIEVAL_SELECTION"].read_bytes(), selected_before)

            after = {str(p.relative_to(release)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in release.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(ns["locate_release"](None, release.parent), release.resolve())
            for module_name in ns.get("MODULE_NAMES", {}).values():
                sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
