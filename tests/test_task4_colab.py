"""Standalone Task 4 notebook contracts and synthetic end-to-end execution."""
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/task4_odme_colab.ipynb"


def notebook_cells():
    book = json.loads(NOTEBOOK.read_text())
    return book, {cell["id"]: "".join(cell["source"]) for cell in book["cells"]}


class Task4NotebookContracts(unittest.TestCase):
    def test_cells_are_clean_valid_python_and_standalone(self):
        book, _ = notebook_cells()
        self.assertEqual(book["nbformat"], 4)
        self.assertEqual(len({cell["id"] for cell in book["cells"]}), len(book["cells"]))
        for cell in book["cells"]:
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "".join(cell["source"])
            self.assertNotIn("%%writefile", source)
            self.assertNotIn("sys.path", source)
            source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
            tree = ast.parse(source)
            compile(tree, cell["id"], "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, {"run", "learning", "data", "fast_model", "torch"})
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name, {"run", "learning", "data", "fast_model", "torch"})

    def test_embedded_functions_match_the_cli(self):
        book, _ = notebook_cells()
        embedded = {}
        for cell in book["cells"]:
            if cell["cell_type"] == "code" and cell["id"].startswith("odme-"):
                embedded.update({node.name: node for node in ast.parse("".join(cell["source"])).body
                                 if isinstance(node, ast.FunctionDef)})
        for filename, names in (("run.py", ("panels", "odme_problem", "odme_prior_variance", "solve_odme",
                                              "predict_odme", "odme_structure", "odme_metrics")),
                                ("learning.py", ("stable_seed", "family_mean", "write_json"))):
            canonical = {node.name: node for node in ast.parse((ROOT / "src" / filename).read_text()).body
                         if isinstance(node, ast.FunctionDef)}
            for name in names:
                with self.subTest(function=name):
                    self.assertEqual(ast.dump(embedded[name]), ast.dump(canonical[name]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("IPython"),
                         "Notebook display packages are required")
    def test_validation_graph_and_both_split_exports(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_odme import release_fixture
        import numpy as np
        import pandas as pd

        _, sources = notebook_cells()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            namespace = {"__name__": "task4_notebook_test"}
            setup = "\n".join(line for line in sources["install"].splitlines() if not line.startswith("%"))
            with patch.dict(os.environ, {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(root / "matplotlib")}), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(setup, "install", "exec"), namespace)
                namespace.update(release=root, report_dir=root / "reports", PANELS=None,
                                 LAMBDAS=(0.05, 0.2), PRIOR_POWERS=(0, 2),
                                 PRIOR_FLOOR_RATIO=0.1, REPEATS=2, SEED=42)
                for identifier in ("odme-utils", "odme-core", "odme-diagnostics", "odme-validation",
                                   "run-validation", "show-results", "prediction-settings"):
                    exec(compile(sources[identifier], identifier, "exec"), namespace)
                namespace.update(PREDICTION_LAMBDA=0.2, PREDICTION_POWER=2)
                exec(compile(sources["export-predictions"], "export-predictions", "exec"), namespace)
            report = json.loads((root / "reports/validation_report.json").read_text())
            self.assertTrue(report["proxy_only"])
            self.assertEqual(report["split"], "validation")
            self.assertEqual(len(report["candidates"]), 5)
            self.assertEqual(len(report["panels"]), 10)
            for filename in ("candidate_summary.csv", "panel_repeats.csv", "network_structure.csv",
                             "candidate_comparison.png", "validation_settings.json"):
                self.assertGreater((root / "reports" / filename).stat().st_size, 0)
            latest_path = root / "reports/predictions/latest.json"
            output_dir = Path(json.loads(latest_path.read_text())["run_dir"])
            settings = json.loads((output_dir / "settings.json").read_text())
            self.assertEqual(settings["lambda"], 0.2)
            self.assertEqual(settings["prior_power"], 2)
            self.assertTrue(settings["all_panels_included"])
            combined = pd.read_csv(output_dir / "path_flow_both.csv")
            self.assertEqual(len(combined), 8)
            self.assertEqual(set(combined.departure_time), {"validation-PM", "private-PM"})
            self.assertFalse(combined[["panel", "departure_time", "path_id"]].duplicated().any())
            self.assertTrue(np.isfinite(combined.path_flow).all())
            self.assertTrue((combined.path_flow >= 0).all())
            previous_combined = (output_dir / "path_flow_both.csv").read_bytes()
            namespace["PREDICTION_PRIOR_ONLY"] = True
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(sources["export-predictions"], "export-predictions", "exec"), namespace)
            prior_dir = Path(json.loads(latest_path.read_text())["run_dir"])
            self.assertNotEqual(prior_dir, output_dir)
            self.assertEqual((output_dir / "path_flow_both.csv").read_bytes(), previous_combined)
            prior = pd.read_csv(prior_dir / "path_flow_both.csv")
            np.testing.assert_array_equal(prior.path_flow, [1, 4, 0, 2, 2, 8, 0, 4])
            settings = json.loads((prior_dir / "settings.json").read_text())
            self.assertEqual(settings["method"], "prior")
            self.assertIsNone(settings["lambda"])

    def test_interrupted_export_preserves_previous_complete_run(self):
        import hashlib
        import time
        import numpy as np
        import pandas as pd
        from test_odme import release_fixture

        _, sources = notebook_cells()
        namespace = dict(Path=Path, np=np, pd=pd, hashlib=hashlib, json=json, time=time, display=lambda *_: None)
        for identifier in ("odme-utils", "odme-core", "odme-diagnostics", "odme-validation"):
            exec(compile(sources[identifier], identifier, "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            namespace.update(release=root, report_dir=root / "reports", PANELS=None,
                             PRIOR_FLOOR_RATIO=0.1, RUN_PREDICTION=True,
                             PREDICTION_PRIOR_ONLY=False, PREDICTION_LAMBDA=0.05, PREDICTION_POWER=0)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(sources["export-predictions"], "export-predictions", "exec"), namespace)
            prediction_root = root / "reports/predictions"
            latest_path = prediction_root / "latest.json"
            previous_latest = latest_path.read_bytes()
            previous_dir = Path(json.loads(previous_latest)["run_dir"])
            previous_files = {path: path.read_bytes() for path in previous_dir.rglob("*") if path.is_file()}
            predict = namespace["predict_odme"]

            def interrupted_predict(release, split, *args):
                if split == "private":
                    raise KeyboardInterrupt("Simulated Colab stop")
                return predict(release, split, *args)

            namespace.update(PREDICTION_LAMBDA=0.2, PREDICTION_POWER=2)
            with patch.dict(namespace, predict_odme=interrupted_predict), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
                exec(compile(sources["export-predictions"], "export-predictions", "exec"), namespace)
            self.assertEqual(latest_path.read_bytes(), previous_latest)
            for path, content in previous_files.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(list(prediction_root.glob("run-*")), [previous_dir])
            pending = list(prediction_root.glob(".pending-*"))
            self.assertEqual(len(pending), 1)
            self.assertTrue((pending[0] / "validation/path_flow.csv").is_file())
            self.assertFalse((pending[0] / "settings.json").exists())


if __name__ == "__main__":
    unittest.main()
