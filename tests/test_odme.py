"""Task 4 contracts: weighted solutions, released evidence, and paired proxies."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.sparse import csr_matrix, issparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run import (main, odme_metrics, odme_prior_variance, odme_problem,
                 odme_structure, solve_odme, validate_odme)


def release_fixture(root):
    network = root / "corridors/P_N/network"
    network.mkdir(parents=True)
    pd.DataFrame({"path_id": ["p1", "p1", "p1", "p2", "p2", "p3", "p3", "p4"],
                  "link_id": ["l1", "l1", "l2", "l1", "l3", "l4", "l5", "connector"]}).to_csv(
        network / "path_link_incidence.csv", index=False)
    for split, scale in (("validation", 1), ("private", 2)):
        folder = root / "task4/P_N" / split
        folder.mkdir(parents=True)
        template = pd.DataFrame({"panel": ["P_N"] * 4, "departure_time": [f"{split}-PM"] * 4,
                                 "path_id": ["p3", "p1", "p4", "p2"],
                                 "origin_zone": ["X", "O", "Y", "O"],
                                 "destination_zone": ["E", "D", "E", "D"]})
        template.to_csv(folder / "sample_submission_path_flow.csv", index=False)
        pd.DataFrame({"path_id": ["p1", "p2", "p3", "p4"],
                      "departure_time": [f"{split}-PM"] * 4,
                      "path_flow": np.array([4., 2., 1., 0.]) * scale}).to_csv(
            folder / "synthetic_weak_prior.csv", index=False)
        pd.DataFrame({"link_id": ["l1", "l2", "l3", "l4", "l5"],
                      "count": np.array([8., 4., 2., 3., 3.]) * scale}).to_csv(
            folder / "synthetic_link_counts.csv", index=False)


class OdmeContracts(unittest.TestCase):
    def test_weighted_dual_matches_independent_augmented_nnls(self):
        matrix = np.array([[1., 1., 0.], [0., 1., 1.]])
        counts, prior, lam = np.array([2., 0.1]), np.array([0., 4., 2.]), 0.05
        for power in (0, 1, 2):
            with self.subTest(power=power):
                variance = (prior + 0.3) ** power
                variance /= np.median(variance)
                penalty = np.diag(np.sqrt(lam / variance))
                expected = nnls(np.vstack((matrix, penalty)), np.r_[counts, penalty @ prior])[0]
                actual = solve_odme(csr_matrix(matrix), counts, prior, lam, power)
                np.testing.assert_allclose(actual, expected, atol=1e-5)
                self.assertTrue((actual >= 0).all())
                gradient = matrix.T @ (matrix @ actual - counts) + lam / variance * (actual - prior)
                self.assertTrue((gradient[actual == 0] >= -1e-5).all())
                np.testing.assert_allclose(gradient[actual > 0], 0, atol=1e-5)

    def test_variance_scale_invariance_and_zero_prior(self):
        prior = np.array([0., 2., 4.])
        for power in (0, 1, 2):
            with self.subTest(power=power):
                variance = odme_prior_variance(prior, power)
                np.testing.assert_allclose(variance, odme_prior_variance(prior * 100, power))
                self.assertEqual(np.median(variance), 1)
                self.assertGreater(variance.min(), 0)
                np.testing.assert_array_equal(odme_prior_variance(np.zeros(3), power), np.ones(3))
        np.testing.assert_allclose(solve_odme(np.eye(2), [1, 2], [0, 0], 0, 2), [1, 2], atol=1e-6)

    def test_large_flow_scale_matches_feasible_analytic_solution(self):
        rng = np.random.default_rng(20261025)
        rows, columns, index = [], [], 0
        for start in range(100):
            for stop in range(start + 1, 101):
                rows.extend(range(start, stop))
                columns.extend([index] * (stop - start))
                index += 1
        matrix = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(100, index))
        unit_prior = rng.lognormal(2, 1.2, index)
        unit_counts = matrix @ (unit_prior * (1 + rng.normal(0, 0.001, index)))
        for scale in (1, 1000):
            prior, counts = unit_prior * scale, unit_counts * scale
            for power in (0, 1, 2):
                variance = (prior + 0.1 * np.median(prior)) ** power
                variance /= np.median(variance)
                for lam in (0.01, 0.05, 0.2):
                    with self.subTest(scale=scale, power=power, lam=lam):
                        inverse_precision = variance / lam
                        hessian = (matrix.multiply(inverse_precision) @ matrix.T).toarray() + np.eye(100)
                        dual = np.linalg.solve(hessian, matrix @ prior - counts)
                        expected = prior - inverse_precision * (matrix.T @ dual)
                        self.assertGreater(expected.min(), 0)
                        actual = solve_odme(matrix, counts, prior, lam, power)
                        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                        self.assertLess(relative_error, 1e-5)

    def test_invalid_solver_inputs_are_rejected(self):
        cases = [dict(lam=-1), dict(lam=np.nan), dict(lam=np.inf), dict(prior_power=3),
                 dict(prior_floor_ratio=0), dict(prior_floor_ratio=np.nan)]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                solve_odme(np.eye(2), [1, 2], [1, 1], **(dict(lam=0.05) | values))
        for matrix, counts, prior in ((np.eye(2), [1], [1, 1]),
                                       (np.eye(2), [1, np.nan], [1, 1]),
                                       (np.eye(2), [1, -1], [1, 1]),
                                       (np.eye(2), [1, 2], [1, -1]),
                                       (np.eye(2), [1, 2], [1, np.inf])):
            with self.subTest(counts=counts, prior=prior), self.assertRaises(ValueError):
                solve_odme(matrix, counts, prior, 0.05)

    def test_loader_aligns_paths_and_uses_only_split_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            template, matrix, counts, prior = odme_problem(root, "P_N", "validation")
            self.assertTrue(issparse(matrix))
            self.assertEqual(matrix.shape, (5, 4))
            self.assertEqual(template.path_id.tolist(), ["p3", "p1", "p4", "p2"])
            np.testing.assert_array_equal(matrix.toarray()[0], [0, 1, 0, 1])
            np.testing.assert_array_equal(matrix.toarray()[:, 2], 0)
            np.testing.assert_array_equal(prior, [1, 4, 0, 2])
            private_template, _, private_counts, private_prior = odme_problem(root, "P_N", "private")
            self.assertTrue(private_template.departure_time.eq("private-PM").all())
            np.testing.assert_array_equal(private_counts, counts * 2)
            np.testing.assert_array_equal(private_prior, prior * 2)
            structure = odme_structure(template, matrix)
            self.assertEqual(structure["unobserved_paths"], 1)
            self.assertEqual(structure["od_pairs"], 3)
            self.assertEqual(structure["od_pairs_with_multiple_paths"], 1)
            self.assertEqual(structure["max_paths_per_od"], 2)
            prior_path = root / "task4/P_N/validation/synthetic_weak_prior.csv"
            malformed = pd.read_csv(prior_path)
            malformed.loc[0, "path_flow"] = np.nan
            malformed.to_csv(prior_path, index=False)
            with self.assertRaisesRegex(ValueError, "Invalid ODME prior"):
                odme_problem(root, "P_N", "validation")

    def test_structure_detects_indistinguishable_paths(self):
        template = pd.DataFrame({"origin_zone": ["A", "B", "C"], "destination_zone": ["D"] * 3})
        structure = odme_structure(template, csr_matrix([[1, 1, 0], [0, 0, 0]]))
        self.assertEqual(structure["distinct_measured_path_signatures"], 2)
        self.assertEqual(structure["ambiguous_path_groups"], 1)
        self.assertEqual(structure["paths_in_ambiguous_groups"], 2)

    def test_metrics_use_l1_and_destination_distribution(self):
        template = pd.DataFrame({"destination_zone": ["D", "E"]})
        result = odme_metrics(template, csr_matrix(np.eye(2)), np.array([2., 2.]),
                              np.array([1., 1.]), np.array([2., 0.]), np.array([True, False]))
        self.assertEqual(result["fit_link_s_link"], 1)
        self.assertEqual(result["hidden_link_normalized_l1"], 1)
        self.assertEqual(result["hidden_link_s_link"], 0)
        self.assertEqual(result["prior_movement_l1"], 2)
        self.assertEqual(result["prior_relative_l1"], 1)
        self.assertEqual(result["destination_distribution_l1_from_prior"], 1)
        self.assertEqual(result["zero_path_fraction"], 0.5)

    def test_validation_pairs_masks_keeps_baselines_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            args = SimpleNamespace(release_root=root, panel=None, split="validation", seed=42,
                                   lambdas=[0.2], prior_powers=[1], prior_floor_ratio=0.1, repeats=3,
                                   output=root / "report.json")
            with contextlib.redirect_stdout(io.StringIO()), patch("run.solve_odme", wraps=solve_odme) as solver:
                validate_odme(args)
            self.assertEqual(solver.call_count, 6)
            self.assertTrue(all(call.args[0].shape[0] == 4 for call in solver.call_args_list))
            report = json.loads(args.output.read_text())
            self.assertTrue(report["proxy_only"])
            self.assertEqual(report["selection_metric"], "hidden_link_normalized_l1")
            self.assertEqual(len(report["candidates"]), 3)
            self.assertEqual(len(report["panels"]), 9)
            self.assertTrue(any(row["lambda"] == 0.05 and row["prior_power"] == 0 for row in report["candidates"]))
            for repeat in range(3):
                rows = [row for row in report["panels"] if row["repeat"] == repeat]
                self.assertEqual(len({tuple(row["hidden_link_indices"]) for row in rows}), 1)
            with contextlib.redirect_stdout(io.StringIO()):
                validate_odme(args)
            self.assertEqual(report, json.loads(args.output.read_text()))

    def test_prediction_cli_preserves_template_and_writes_prior_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_fixture(root)
            output = root / "path-flow.csv"
            argv = ["run.py", "predict-odme", "--release-root", str(root), "--split", "validation",
                    "--panel", "P_N", "--output", str(output)]
            with patch.object(sys, "argv", argv + ["--prior-only"]):
                main()
            result = pd.read_csv(output)
            self.assertEqual(result.path_id.tolist(), ["p3", "p1", "p4", "p2"])
            np.testing.assert_array_equal(result.path_flow, [1, 4, 0, 2])
            with patch.object(sys, "argv", argv + ["--prior-power", "2", "--odme-lambda", "0.2"]):
                main()
            result = pd.read_csv(output)
            self.assertTrue(np.isfinite(result.path_flow).all())
            self.assertTrue((result.path_flow >= 0).all())
            self.assertTrue(result.departure_time.eq("validation-PM").all())


if __name__ == "__main__":
    unittest.main()
