"""Small checks for the contracts that are easy to break without release data."""

import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import PanelInfo, StationDay, scan_layout, station_window_features  # noqa: E402
from run import center_windows, queue_windows  # noqa: E402


class PipelineChecks(unittest.TestCase):
    def panel(self):
        return PanelInfo("P", ["A", "B"], ((0, 2),), {"A": 0, "B": 1},
                         np.array([100, 100], np.float32), np.array([2, 2], np.float32),
                         np.array([1, 1], np.float32), np.array([60, 60], np.float32), {})

    def test_disconnected_scans_reset(self):
        topology = pd.DataFrame({"link_id": ["A", "B", "C"], "incoming_link_ids": ["", "A", ""]})
        order, segments = scan_layout(["A", "B", "C"], topology)
        self.assertEqual(order, ["A", "B", "C"])
        self.assertEqual(segments, ((0, 2), (2, 3)))

    def test_panel_layout_keeps_unobserved_connector(self):
        with tempfile.TemporaryDirectory() as directory:
            panel_root = Path(directory) / "corridors" / "P"
            train = panel_root / "train" / "mainline_states"
            network = panel_root / "network"
            train.mkdir(parents=True)
            network.mkdir()
            (train / "day.parquet").touch()
            pd.DataFrame({"link_id": ["A", "B", "C"],
                          "incoming_link_ids": ["", "A", "B"]}).to_csv(network / "lwr_mainline_topology.csv", index=False)
            pd.DataFrame({"link_id": ["A", "B", "C"], "free_speed_kmh": [100] * 3,
                          "lanes": [2] * 3, "length_km": [1] * 3}).to_csv(network / "links.csv", index=False)
            pd.DataFrame({"link_id": ["A", "C"], "station_id": ["s1", "s2"],
                          "lanes": [2, 2], "v_cut": [55, 55]}).to_csv(network / "fd_parameters.csv", index=False)
            with patch("data.pd.read_parquet", return_value=pd.DataFrame({"link_id": ["A", "C"]})):
                panel = PanelInfo.load(Path(directory), "P")
            self.assertEqual(panel.links, ["A", "B", "C"])
            self.assertEqual(panel.segments, ((0, 3),))
            np.testing.assert_allclose(panel.cutoffs, [55, 60, 55])

    def test_mask_before_link_aggregation_and_station_context(self):
        timestamps = [f"2030-06-01T00:{minute:02d}:00Z" for minute in (0, 5, 10)]
        day = StationDay(timestamps, ["s1", "s2"], ["A", "A"], np.array([0, 0]),
                         np.array([2, 2], np.float32),
                         np.array([[2, 4], [99, 99], [3, 5]], np.float32),
                         np.array([[1, 2], [99, 99], [1.5, 2.5]], np.float32),
                         np.ones((3, 2), bool))
        visible = np.array([[True, True], [False, False], [True, True]])
        links, station = station_window_features(day, self.panel(), visible, np.arange(3))
        self.assertEqual(links[1, 0, 0], 0)
        self.assertEqual(links[1, 0, 2], 0)
        self.assertEqual(station[1, 0, 0], 2.5)
        self.assertEqual(station[1, 1, 0], 4.5)

    def test_center_windows_cover_each_step_once(self):
        windows = list(center_windows(29))
        covered = [step for _, _, positions in windows for step in range(positions.start, positions.stop)]
        self.assertEqual(covered, list(range(29)))
        self.assertTrue(all(len(index) == 72 for index, _, _ in windows))

    def test_queue_history_ends_before_label(self):
        timestamps = pd.date_range("2030-06-01", periods=24, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        speed = np.full((24, 2), 4.0, np.float32)
        speed[13, 0] = 1.0
        day = StationDay(timestamps, ["s1", "s2"], ["A", "B"], np.array([0, 1]),
                         np.array([2, 2], np.float32), speed, np.ones_like(speed),
                         np.ones_like(speed, dtype=bool))
        windows = list(queue_windows(day, self.panel(), 10, np.random.default_rng(1)))
        self.assertTrue(any(kind == "onset" for kind, *_ in windows))
        for _, history, label, _, valid in windows:
            self.assertEqual(history.shape, (12, 2, 9))
            self.assertEqual(label.shape, (6, 2))
            self.assertEqual(valid.shape, (6, 2))
            self.assertEqual(history[-1, 0, 0], day.speed[11, 0])
            self.assertTrue(label[0, 0])


if __name__ == "__main__":
    unittest.main()
