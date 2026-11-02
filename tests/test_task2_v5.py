"""Queue v5 loss, sampling, calibration separation, and notebook contracts."""
import ast
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from test_task123_colab import NOTEBOOK, KAGGLE_NOTEBOOK, tagged
import test_task2_v4 as v4


class Task2V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v4.Task2V4Tests.setUpClass()
        cls.learning = v4.Task2V4Tests.learning
        cls.runner = v4.Task2V4Tests.runner
        cls.model = v4.Task2V4Tests.model

    args = v4.Task2V4Tests.args

    def test_prior_diagnostics_preserve_default_output_and_checkpoint_shapes(self):
        model = self.model.TrafficQueueModel(features=44, node_count=2, width=8, blocks=1, heads=2,
            dropout=0., backend='scan', graph=False, queue_head='conditional', queue_future_prior=True).eval()
        with torch.no_grad():
            model.future_head[-1].bias.fill_(.4)
            inputs = (torch.zeros(1, 12, 2, 44), torch.tensor([1, 2]), ((0, 2),), torch.empty(2, 0, dtype=torch.long))
            keywords = dict(condition=torch.tensor([0]), future=torch.zeros(1, 6, 2, model.future_head[0].in_features))
            normal = model(*inputs, **keywords)
            diagnostic = model(*inputs, **keywords, return_components=True)
        self.assertEqual(len(normal), 2)
        torch.testing.assert_close(normal[0], diagnostic[0])
        torch.testing.assert_close(normal[1], diagnostic[1])
        parts = diagnostic[2]
        torch.testing.assert_close(parts['queue_logit'], parts['history_logit'] + parts['prior_logit'])

    def test_soft_bce_hard_iou_and_invalid_cells(self):
        logits = torch.tensor([[[1., -1., 9.], [0., 2., -9.]]], requires_grad=True)
        hard = torch.tensor([[[1., 0., 1.], [1., 0., 1.]]])
        soft = torch.tensor([[[.6, .4, 0.], [.8, .2, 0.]]])
        mask = torch.tensor([[[True, True, False], [True, True, False]]])
        batch = dict(truth=hard, soft=hard, train_target=soft, mask=mask,
                     weight=torch.ones_like(hard), speed=torch.zeros_like(hard))
        args = types.SimpleNamespace(queue_iou_target='hard', soft_iou_weight=.2, speed_weight=0.,
                                     normal_weight=.2, queue_balanced=True)
        got = self.learning.queue_loss(logits, batch['speed'], batch, ['onset'], args)
        bce = F.binary_cross_entropy_with_logits(logits[mask], soft[mask])
        p, y = logits.sigmoid()[mask], hard[mask]
        expected = bce + .2 * (1 - ((p * y).sum() + 1e-6) / ((p + y - p * y).sum() + 1e-6))
        torch.testing.assert_close(got, expected)
        got.backward()
        self.assertTrue((logits.grad[~mask] == 0).all())
        changed = {**batch, 'train_target': 1 - soft}
        args.soft_iou_weight = 0.
        bce_change = self.learning.queue_loss(logits, batch['speed'], changed, ['onset'], args) - bce
        args.soft_iou_weight = .2
        new = self.learning.queue_loss(logits, batch['speed'], changed, ['onset'], args)
        torch.testing.assert_close(new - got, bce_change)

    def test_confirmed_clear_requires_three_eligible_target_cells(self):
        truth = np.array([[1, 0], [0, 1], [0, 0], [0, 1], [0, 0], [0, 0]], bool)
        valid, target = np.ones_like(truth), np.zeros_like(truth)
        target[:, 0] = True
        self.assertTrue(self.learning.confirmed_queue_clear(truth, valid, target))
        valid[1, 0] = False
        self.assertFalse(self.learning.confirmed_queue_clear(truth, valid, target))
        target[:] = False
        self.assertFalse(self.learning.confirmed_queue_clear(truth, np.ones_like(valid), target))

    def test_clear_sampling_preserves_quota_caps_repetition_and_leaves_onset_alone(self):
        values = [('day', i, i < 2) for i in range(12)]
        sampled = self.learning.sample_queue_group(values, 20, 'ongoing', .5, np.random.default_rng(4))
        self.assertEqual(len(sampled), 20)
        self.assertEqual(sum(values[i][2] for i in sampled), 4)
        self.assertLessEqual(max(np.bincount(sampled, minlength=12)[:2]), 2)
        for kind, fraction in (('onset', .5), ('ongoing', 0.)):
            got = self.learning.sample_queue_group(values, 20, kind, fraction, np.random.default_rng(4))
            expected = np.random.default_rng(4).choice(len(values), 20, replace=True)
            np.testing.assert_array_equal(got, expected)
        for only in (False, True):
            fallback = [('day', i, only) for i in range(3)]
            got = self.learning.sample_queue_group(fallback, 8, 'ongoing', .5, np.random.default_rng(4))
            self.assertEqual(len(got), 8)

    def test_balanced_sampling_changes_only_ongoing_and_reuses_candidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f'day-{i}' for i in range(2)]
            for path in paths:
                path.write_bytes(b'day')
            context = {'P': dict(fit_paths=paths, queue_target_nodes=[0])}
            args = types.SimpleNamespace(release_root=root / 'release', history_coverage=.5, future_coverage=.5,
                queue_sampling_version=5, update_windows=12, seed=42, queue_clear_fraction=0.,
                queue_resample=True, queue_condition_gap=True, windows_per_day=10, normal_windows=0,
                window_gap_minutes=0)
            metadata = dict(candidates=dict(onset=list(range(12, 18)), ongoing=list(range(30, 37)), normal=[]),
                            clear_origins=[30])
            with patch.object(self.learning, 'queue_examples', return_value=metadata) as examples, \
                    contextlib.redirect_stdout(io.StringIO()):
                baseline = self.learning.balanced_queue_items(context, args, 0)
                args.queue_clear_fraction = .5
                changed = self.learning.balanced_queue_items(context, args, 0)
            self.assertEqual(examples.call_count, 2)
            onset = lambda items: Counter((name, str(path), origin) for name, path, selected in items
                                          for kind, origin in selected if kind == 'onset')
            self.assertEqual(onset(baseline), onset(changed))
            counts = Counter(kind for _, _, selected in changed for kind, _ in selected)
            self.assertEqual(counts['onset'], counts['ongoing'])
            self.assertEqual(args.queue_sampling_report['P:ongoing']['selected_clear'], 4)

    def test_clear_repetition_cap_cannot_reduce_the_original_sampling_rate(self):
        values = [('day', i, i == 0) for i in range(10)]
        for quota in (1, 21, 100):
            with self.subTest(quota=quota):
                expected = self.learning.sample_queue_group(values, quota, 'ongoing', 0., np.random.default_rng(4))
                actual = self.learning.sample_queue_group(values, quota, 'ongoing', .5, np.random.default_rng(4))
                np.testing.assert_array_equal(actual, expected)

    def test_window_manifests_distinguish_date_spread_and_effective_coverage_settings(self):
        class Queue(torch.nn.Module):
            def forward(self, x, *args, **kwargs):
                logits = torch.zeros(len(x), 6, 2)
                return logits, torch.zeros_like(logits)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            day = root / 'states_2031-01-01.parquet'
            day.write_bytes(b'day')
            args = self.args(root / 'release', root / 'queue.pt', '--queue-public-contract',
                             '--queue-iou-target', 'hard', '--queue-validation-seed', '17',
                             '--queue-policy-scope', 'global', '--validation-windows', '2')
            self.learning.queue_recipe(args)
            args.queue_sampling_version, args.model_config, args.use_bf16 = 5, {}, False
            panel = types.SimpleNamespace(links=['a', 'b'], segments=((0, 2),), edges=np.empty((2, 0), np.int64))
            context = {'P_N': dict(panel=panel)}
            truth = np.tile([1., 0.], (6, 1)).astype(np.float32)
            samples = [dict(x=np.zeros((12, 2, 44), np.float32), truth=truth, soft=truth,
                speed=np.zeros_like(truth), mask=np.ones_like(truth, bool), weight=np.ones_like(truth),
                target_mask=np.ones_like(truth, bool), baseline=np.zeros_like(truth, bool),
                condition=kind, date='2031-01-01', origin=hour * 12,
                forecast_origin=f'2031-01-01T{hour:02d}:00:00Z')
                for kind in ('onset', 'ongoing') for hour in (1, 7)]
            counts = []
            with patch.object(self.learning, 'schedule', return_value=[('P_N', day)]), \
                    patch.object(self.learning, 'prepared_days', return_value=[('P_N', day, samples, 0.)]), \
                    patch.object(self.learning, 'period_files', return_value=[day]):
                for spread in (False, True, True):
                    args.queue_validation_spread = spread
                    report, _ = self.learning.run_period(Queue(), context, {}, args, 'queue', 'select', 0, 'cpu')
                    counts.append(report['windows'])
                args.queue_public_contract = False
                for coverage in (.5, .9):
                    args.future_coverage = coverage
                    self.learning.run_period(Queue(), context, {}, args, 'queue', 'select', 0, 'cpu')
            self.assertEqual(counts, [4, 2, 2])
            manifests = [json.loads(path.read_text()) for path in (root / 'processed_v2').glob('queue-windows-*.json')]
            self.assertEqual(len(manifests), 4)
            self.assertEqual({row['config']['selector_settings']['queue_validation_spread'] for row in manifests}, {False, True})
            nonpublic = [row for row in manifests if not row['config']['public_contract']]
            self.assertEqual({row['config']['selector_settings']['future_coverage'] for row in nonpublic}, {.5, .9})

    def test_selection_dates_are_disjoint_and_legacy_keeps_all_january(self):
        dates = ('2030-12-31', '2031-01-01', '2031-01-15', '2031-01-16', '2031-01-31', '2031-02-01')
        files = {date: Path(date) for date in dates}
        with patch.object(self.learning, 'file_index', return_value=files):
            take = lambda period, split: self.learning.period_files(Path('/release'), 'P', period, selection_split=split)
            self.assertEqual(take('select', True), [files[d] for d in dates[1:3]])
            self.assertEqual(take('calibrate', True), [files[d] for d in dates[3:5]])
            self.assertEqual(take('select', False), [files[d] for d in dates[1:5]])
            self.assertEqual(take('holdout', True), [files[dates[-1]]])

    def test_validation_origins_are_independent_of_training_seed_and_epoch(self):
        args = types.SimpleNamespace(seed=42, queue_validation_seed=17, queue_public_contract=False,
            queue_validation_spread=True, queue_condition_gap=True, queue_resample=True,
            window_gap_minutes=0, normal_windows=0, windows_per_day=3)
        candidates = dict(onset=list(range(12, 42)), ongoing=list(range(42, 72)), normal=[])
        choose = lambda epoch: self.learning.choose_queue_origins(candidates, args, 'P', Path('day'), 'select', epoch)
        before = choose(0)
        args.seed = 43
        self.assertEqual(before, choose(9))

    def test_old_resume_retains_legacy_recipe_and_new_resume_rejects_changes(self):
        args = types.SimpleNamespace()
        self.learning.queue_recipe(args, dict(training_args={}))
        self.assertEqual(args.queue_iou_target, 'train')
        self.assertEqual(args.queue_clear_fraction, 0.)
        self.assertFalse(args.queue_selection_split)
        saved = dict(queue_iou_target='hard', queue_clear_fraction=.5, queue_selection_split=True,
                     queue_validation_seed=17, queue_policy_scope='global')
        restored = types.SimpleNamespace()
        self.learning.queue_recipe(restored, dict(training_args=saved))
        self.assertEqual({k: getattr(restored, k) for k in saved}, saved)
        self.assertFalse(restored.queue_prior_gate)
        self.assertEqual(restored.queue_transition_weight, 0.)
        with self.assertRaisesRegex(ValueError, 'Resume must retain'):
            self.learning.queue_recipe(types.SimpleNamespace(queue_clear_fraction=0.), dict(training_args=saved))

    def test_model_independent_window_manifest_detects_label_or_link_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = np.zeros((6, 2), bool)
            row = dict(panel='P', origin='2031-01-16T01:00:00Z', condition='ongoing', truth=truth,
                       target_mask=np.ones_like(truth), valid=np.ones_like(truth), link_ids=np.array(['a', 'b']))
            manifest = dict(config=dict(checkpoint_sha256='A', period='calibrate', seed=17, windows=10))
            self.learning.check_queue_window_manifest(root / 'release', [row], manifest)
            manifest['config']['checkpoint_sha256'] = 'B'
            self.learning.check_queue_window_manifest(root / 'release', [row], manifest)
            self.assertEqual(len(list((root / 'processed_v2').glob('queue-windows-*.json'))), 1)
            truth[0, 0] = True
            with self.assertRaisesRegex(ValueError, '검증 창'):
                self.learning.check_queue_window_manifest(root / 'release', [row], manifest)
            truth[0, 0] = False
            row['link_ids'] = np.array(['b', 'a'])
            with self.assertRaisesRegex(ValueError, '검증 창'):
                self.learning.check_queue_window_manifest(root / 'release', [row], manifest)

    def test_global_policy_excludes_conditional_and_force_nonempty(self):
        truth = np.tile([True, False], (6, 1))
        rows = [dict(panel='P', date='2031-01-16', origin='T', condition=kind,
                     probability=np.tile([.8, .2], (6, 1)), truth=truth, valid=np.ones_like(truth),
                     target_mask=np.ones_like(truth), baseline=np.zeros_like(truth)) for kind in ('onset', 'ongoing')]
        candidates = self.learning.queue_policy_candidates(rows, [.3, .7], .6, 'global')
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            policy = candidate['policy']
            self.assertFalse(policy['force_nonempty'])
            self.assertEqual(policy['thresholds']['onset'], policy['thresholds']['ongoing'])
        with patch.object(self.learning.torch, 'load', return_value=dict(training_args=dict(queue_selection_split=True))):
            with self.assertRaisesRegex(ValueError, '16~31'):
                self.learning.calibrate_queue(rows, dict(config=dict(period='select')), Path('/model'), Path('/out'), [.7])

    def test_error_categories_partition_false_positives_and_mask_transitions(self):
        truth = np.array([[0, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 0]], bool)
        prediction = np.ones_like(truth)
        valid = np.ones_like(truth)
        errors = self.learning.queue_error_masks(prediction, truth, valid)
        self.assertEqual([errors[k].sum() for k in ('fp_new_link', 'fp_start_early', 'fp_end_late', 'fp_inside')], [6, 1, 3, 0])
        self.assertEqual(errors['onset_transitions'].sum(), 1)
        self.assertEqual(errors['clear_transitions'].sum(), 1)
        self.assertEqual(errors['clear_transitions_detected'].sum(), 0)
        valid[3, 0] = False
        self.assertEqual(self.learning.queue_error_masks(prediction, truth, valid)['clear_transitions'].sum(), 0)

    def test_notebook_task2_only_needs_no_state_checkpoint_and_resume_omits_new_recipe(self):
        for notebook in (NOTEBOOK, KAGGLE_NOTEBOOK):
            with self.subTest(notebook=notebook), tempfile.TemporaryDirectory() as directory:
                root, calls, ns = Path(directory), [], {}
                exec(tagged('configuration', notebook), ns)
                ns.update(TASK2_ONLY=True, MAGINET_DIR=root / 'missing-state', RUN_MODE='train',
                          QUEUE_DIR=root, QUEUE_CHECKPOINT=root / 'queue.pt', RELEASE=root / 'release',
                          EFFECTIVE_BACKEND='scan', QUEUE_PRECISION='fp32', panel_args=[], queue_eval_args=[],
                          training_action=lambda *args: 'resume', call_queue=calls.append,
                          call_maginet=lambda *args: self.fail('Task 2 must not call MagiNet'))
                ns['QUEUE_CHECKPOINT'].write_bytes(b'checkpoint')
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(tagged('train-state', notebook), ns)
                    exec(tagged('evaluate', notebook), ns)
                    exec(tagged('train-queue', notebook), ns)
                self.assertEqual(len(calls), 1)
                self.assertIn('--resume', calls[0])
                self.assertNotIn('--queue-label-mode', calls[0])
                self.assertNotIn('--queue-selection-split', calls[0])
                def predict(checkpoint, release, split, path, selection):
                    path.write_text('queue_pred\n0\n')
                ns.update(learning=types.SimpleNamespace(predict_queue_calibrated=predict), ACTIVE_QUEUE_SELECTION=None,
                          runner=types.SimpleNamespace(STATE_KEYS=[], QUEUE_KEYS=[], ODME_KEYS=[]))
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(tagged('predict', notebook), ns)
                    exec(tagged('merge', notebook), ns)
                self.assertEqual(ns['STATE_FILES'], [])
                self.assertEqual(len(ns['QUEUE_FILES']), 2)
                self.assertTrue(all(path.is_file() for path in ns['QUEUE_FILES']))
                tree = ast.parse(tagged('predict', notebook))
                prediction_guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                                        and ast.unparse(node.test) == 'not TASK2_ONLY')
                self.assertTrue(prediction_guard.body)


if __name__ == '__main__':
    unittest.main()
