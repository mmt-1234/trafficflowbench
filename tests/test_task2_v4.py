"""Task 2 training contracts on synthetic inputs, shared by both notebooks."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from test_task123_colab import NOTEBOOK, KAGGLE_NOTEBOOK, embedded, tagged
from test_task123_kaggle import release_fixture


@contextlib.contextmanager
def synthetic_parquet():
    # Exercise the learning pipeline without requiring the optional Parquet codec.
    def read(path, columns=None, **kwargs):
        frame = pd.read_pickle(path)
        return frame if columns is None else frame[columns]
    def write(frame, path, **kwargs):
        frame.to_pickle(path)
    arrow = types.ModuleType('pyarrow')
    parquet = types.ModuleType('pyarrow.parquet')
    parquet.read_schema = lambda path: types.SimpleNamespace(names=list(pd.read_pickle(path).columns))
    arrow.parquet = parquet
    replacements = {'pyarrow': arrow, 'pyarrow.parquet': parquet}
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        with patch.object(pd, 'read_parquet', read), patch.object(pd.DataFrame, 'to_parquet', write):
            yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class Task2V4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = dict(ast=ast, hashlib=hashlib, types=types, sys=sys, WORK_ROOT=Path('/tmp/task2-contract'))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(tagged('module-loader'), ns)
            for name in ('data', 'fast_model', 'learning', 'run'):
                exec(tagged(name), ns)
        cls.data, cls.model, cls.learning, cls.runner = (ns['MODULES'][name] for name in
                                                       ('data', 'fast_model', 'learning', 'run'))
        torch.set_num_threads(1)

    def args(self, release, checkpoint, *extra):
        captured = []
        command = ['train-queue', '--release-root', str(release), '--checkpoint', str(checkpoint),
                   '--backend', 'scan', '--width', '8', '--heads', '2', '--blocks', '1', '--dropout', '0',
                   '--cpu-threads', '1', '--workers', '0', '--epochs', '1', '--batch-size', '2',
                   '--update-windows', '4', '--max-days', '2', '--windows-per-day', '1',
                   '--queue-condition-gap', '--queue-changes', '--graph', *map(str, extra)]
        with patch.object(sys, 'argv', ['task2-test', *command]), patch.object(self.runner, 'fit_queue', captured.append):
            self.runner.main()
        return captured[0]

    def test_notebooks_share_the_same_task2_implementation(self):
        for name in ('data', 'fast_model', 'learning', 'run'):
            self.assertEqual(embedded(name, NOTEBOOK), embedded(name, KAGGLE_NOTEBOOK))

    def test_removed_state_entry_points_and_checkpoints_are_rejected(self):
        for command in ('train-state', 'submit'):
            with self.subTest(command=command), patch.object(sys, 'argv', ['queue-test', command]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.runner.main()
        with self.assertRaisesRegex(ValueError, 'Queue'):
            self.learning.model_from_checkpoint(dict(format_version=2, task='state'), 'cpu')

    def test_removed_experiment_options_and_structures_are_rejected(self):
        removed = (('--label-temperature', '1'), ('--uncertainty-margin', '5'),
                   ('--missing-aware',), ('--adaptive-graph-k', '2'), ('--graph-mode', 'reliability'))
        for options in removed:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(Path('/unused'), Path('/unused/queue.pt'), *options)
        config = dict(features=44, node_count=2, width=8, blocks=1, heads=2, backend='scan', graph=True)
        for change in (dict(missing_aware=True), dict(adaptive_graph_k=2), dict(graph_mode='reliability')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, '제거된 Queue 구조'):
                self.model.TrafficQueueModel(**config, **change)
        # Stored defaults from an existing Queue checkpoint remain readable.
        self.model.TrafficQueueModel(**config, missing_aware=False, adaptive_graph_k=0, graph_mode='mean')
        self.assertFalse(hasattr(self.runner, 'validate_odme'))
        self.assertFalse(hasattr(self.runner, 'odme_metrics'))
        self.assertFalse(hasattr(self.runner, 'odme_structure'))
        self.assertTrue(callable(self.runner.predict_odme))

    def test_removed_label_correction_cannot_silently_change_resume_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key in ('label_temperature', 'uncertainty_margin'):
                checkpoint = root / 'legacy.pt'
                torch.save(dict(task='queue', training_args={key: 1.}), checkpoint)
                before = checkpoint.read_bytes()
                args = self.args(root / 'unused', root / 'new.pt', '--resume', checkpoint)
                with self.subTest(key=key), patch('torch.cuda.is_available', return_value=False), \
                        self.assertRaisesRegex(ValueError, '제거된 라벨 보정'):
                    self.learning.train(args, 'queue')
                self.assertEqual(checkpoint.read_bytes(), before)

    def test_wavelet_inputs_levels_gradients_and_checkpoint_reload(self):
        torch.manual_seed(7)
        x = torch.rand(2, 13, 2, 44)
        x[..., 10:13] = torch.randint(0, 2, (2, 13, 2, 3)).float()
        future = torch.rand(2, 6, 2, 7)
        call = (x, torch.tensor([1, 2]), ((0, 2),), torch.tensor([[0], [1]]))
        options = dict(condition=torch.tensor([0, 1]), future=future)
        for wavelet_input in ('signals', 'embedded'):
            for levels in (0, 1, 2, 3):
                with self.subTest(input=wavelet_input, levels=levels):
                    config = dict(features=44, node_count=2, width=8, blocks=1, heads=2,
                                  dropout=0., backend='scan', graph=True, queue_head='conditional',
                                  queue_changes=True, queue_future_prior=True,
                                  wavelet_levels=levels, wavelet_input=wavelet_input)
                    model = self.model.TrafficQueueModel(**config).eval()
                    logits, speeds = model(*call, **options)
                    self.assertEqual(logits.shape, (2, 6, 2))
                    (logits.square().mean() + speeds.square().mean()).backward()
                    if levels:
                        gradient = model.encoder.wavelet.gain.grad
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(gradient.abs().item(), 0)
                    else:
                        self.assertIsNone(model.encoder.wavelet)
                    checkpoint = dict(format_version=2, task='queue', model_config=config,
                                      feature_names=list(self.data.queue_feature_names(True)),
                                      queue_future_names=list(self.data.QUEUE_FUTURE_NAMES), weights=model.state_dict())
                    stream = io.BytesIO()
                    torch.save(checkpoint, stream)
                    stream.seek(0)
                    restored = self.learning.model_from_checkpoint(torch.load(stream, weights_only=True), 'cpu').eval()
                    with torch.no_grad():
                        expected = model(*call, **options)
                        actual = restored(*call, **options)
                    for first, second in zip(expected, actual):
                        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_probability_and_future_calendar_boundaries(self):
        p = self.learning.queue_label_probability(np.array([30, 59, 60, 61, 90]), 60, 5)
        np.testing.assert_allclose(p, [1, .6, .5, .4, 0])
        for margin in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                self.learning.queue_label_probability(np.array([59]), 60, margin)
        panel = types.SimpleNamespace(links=['l0', 'l1'])
        profile = np.broadcast_to(np.arange(2016)[:, None, None], (2016, 2, 3)).astype(np.float32).copy()
        future = self.data.queue_future_features('2030-06-02T23:55:00Z', panel, profile)
        self.assertEqual(future.shape, (6, 2, 7))
        np.testing.assert_array_equal(future[:, 0, 0], np.arange(6))  # Sunday -> Monday
        np.testing.assert_allclose(future[:, 0, -1], np.arange(1, 7) / 6)
        self.assertTrue(np.isfinite(future).all())

    def test_warm_start_preserves_predictions_and_routes_gradients(self):
        config = dict(features=44, node_count=2, width=8, heads=2, blocks=1, backend='scan', dropout=0)
        old = self.model.TrafficQueueModel(queue_head='pooled-mlp', **config).eval()
        new = self.model.TrafficQueueModel(queue_head='conditional', queue_future_prior=True, **config).eval()
        self.learning.initialize_queue(new, dict(format_version=2, task='queue', model_config={'queue_head': 'pooled-mlp'},
                                                weights=old.state_dict()))
        x, nodes = torch.randn(2, 12, 2, 44), torch.tensor([1, 2])
        future = torch.randn(2, 6, 2, 7)
        edges = torch.tensor([[0, 1], [1, 0]])
        with torch.no_grad():
            expected = old(x, nodes, [(0, 2)], edges)
            actual = new(x, nodes, [(0, 2)], edges, condition=torch.tensor([0, 1]), future=future)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        logits, speed = new(x, nodes, [(0, 2)], edges, condition=torch.zeros(2, dtype=torch.long), future=future)
        (logits.sum() + speed.sum()).backward()
        self.assertGreater(new.head[0][-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(new.head[1][-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(new.future_head[-1].weight.grad.abs().sum().item(), 0)

    def test_bce_and_iou_share_sampled_labels_and_ignore_invalid_cells(self):
        logits = torch.tensor([[[.3, -.5, 2.]]], requires_grad=True)
        label = torch.tensor([[[1., 0., 1.]]])
        mask = torch.tensor([[[True, True, False]]])
        batch = dict(truth=1 - label, soft=1 - label, train_target=label, mask=mask,
                     weight=torch.ones_like(label), speed=torch.zeros_like(label))
        args = types.SimpleNamespace(soft_iou_weight=.2, speed_weight=0, normal_weight=.2, queue_balanced=True)
        got = self.learning.queue_loss(logits, batch['speed'], batch, ['onset'], args)
        probability = logits.sigmoid() * mask
        truth = label * mask
        expected = F.binary_cross_entropy_with_logits(logits[..., :2], label[..., :2])
        expected += .2 * (1 - ((probability * truth).sum() + 1e-6) /
                          ((probability + truth - probability * truth).sum() + 1e-6))
        torch.testing.assert_close(got, expected)
        got.backward()
        self.assertEqual(logits.grad[..., -1].item(), 0)
        accumulated = self.learning.queue_loss(logits, batch['speed'], batch, ['onset'], args, {'onset': 4})
        torch.testing.assert_close(accumulated, got / 4)

    def test_sampled_labels_are_reproducible_and_evaluation_stays_hard(self):
        with tempfile.TemporaryDirectory() as directory, synthetic_parquet():
            root = Path(directory)
            release_fixture(root / 'release')
            args = self.args(root / 'release', root / 'queue.pt', '--queue-label-mode', 'sample', '--queue-future-prior')
            args.queue_sampling_version = 4
            with contextlib.redirect_stdout(io.StringIO()):
                context = self.learning.contexts(args, 'queue')['P_N']
            path = context['fit_paths'][0]
            frame = pd.read_parquet(path)
            # Keep a fixed eligible ongoing window, but place its future near the cutoff.
            future_times = sorted(frame.timestamp.unique())[17:23]
            frame.loc[frame.timestamp.isin(future_times), 'speed_kmh'] = 59
            frame.to_parquet(path, index=False)
            def samples(period, epoch):
                return self.learning.queue_examples(context, path, args, period, epoch, selected=[('ongoing', 16)])[0]
            a, b = samples('fit', 2), samples('fit', 2)
            np.testing.assert_array_equal(a['train_target'], b['train_target'])
            self.assertEqual(a['branch'].item(), 1)
            self.assertTrue(set(np.unique(a['train_target'])) <= {0, 1})
            self.assertTrue(any(not np.array_equal(a['train_target'], samples('fit', i)['train_target']) for i in range(10)))
            before = samples('select', 2)
            after = samples('select', 9)
            np.testing.assert_array_equal(before['train_target'], before['truth'])
            np.testing.assert_array_equal(before['train_target'], after['train_target'])
            frame.loc[frame.timestamp.isin(future_times), 'speed_kmh'] = 30
            frame.to_parquet(path, index=False)
            changed = samples('fit', 2)
            for field in ('x', 'future', 'branch'):
                np.testing.assert_array_equal(a[field], changed[field])

    def test_repeated_windows_build_features_once_and_keep_multiplicity(self):
        with tempfile.TemporaryDirectory() as directory, synthetic_parquet():
            root = Path(directory)
            release_fixture(root / 'release')
            args = self.args(root / 'release', root / 'queue.pt', '--queue-future-prior', '--queue-label-mode', 'sample')
            args.queue_sampling_version = 4
            with contextlib.redirect_stdout(io.StringIO()):
                context = self.learning.contexts(args, 'queue')['P_N']
            path = context['fit_paths'][0]
            with patch.object(self.learning, 'queue_features', wraps=self.learning.queue_features) as past, \
                    patch.object(self.learning, 'queue_future_features', wraps=self.learning.queue_future_features) as future:
                samples = self.learning.queue_examples(context, path, args, 'fit', 1,
                                                       selected=[('onset', 12)] * 48)
            self.assertEqual(len(samples), 48)
            self.assertEqual(past.call_count, 1)
            self.assertEqual(future.call_count, 1)
            self.assertTrue(all(sample is samples[0] for sample in samples))

    def test_initial_checkpoint_cannot_alias_best_or_last_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            best = root / 'queue.pt'
            last = root / 'queue.last.pt'
            for path in (best, last):
                torch.save({'format_version': 2, 'task': 'queue'}, path)
            alias = root / 'initial.pt'
            alias.symlink_to(last)
            before = {path: path.read_bytes() for path in (best, last)}
            for initial in (best, last, alias):
                with self.subTest(initial=initial.name), patch('torch.cuda.is_available', return_value=False):
                    args = self.args(root / 'release', best, '--init-queue', initial)
                    with self.assertRaisesRegex(ValueError, 'best·last 출력 경로'):
                        self.learning.train(args, 'queue')
                    self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_one_update_is_not_rescaled_by_another_direction(self):
        class Queue(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(()))
            def forward(self, x, *args, **kwargs):
                logits = self.bias.expand(len(x), 6, 2)
                return logits, torch.zeros_like(logits)
        sample = dict(x=np.zeros((12, 2, 44), np.float32), truth=np.ones((6, 2), np.float32),
                      soft=np.ones((6, 2), np.float32), speed=np.zeros((6, 2), np.float32),
                      mask=np.ones((6, 2), bool), weight=np.ones((6, 2), np.float32), condition='onset')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for names in (('P_N',), ('P_N', 'P_S')):
                args = self.args(root, root / 'queue.pt', '--queue-balanced')
                args.model_config, args.use_bf16 = {}, False
                args.update_windows, args.soft_iou_weight, args.speed_weight = 1, 0, 0
                context = {name: dict(panel=types.SimpleNamespace(links=['l0', 'l1'], segments=((0, 2),),
                                                                 edges=np.array([[0], [1]]))) for name in names}
                model = Queue()
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                with patch.object(self.learning, 'balanced_queue_items', return_value=[('P_N', root)]), \
                        patch.object(self.learning, 'prepared_days', return_value=[('P_N', root, [sample], 0.)]):
                    self.learning.run_period(model, context, {}, args, 'queue', 'fit', 0, 'cpu', optimizer)
                self.assertAlmostEqual(model.bias.item(), .05, places=7)

    def test_epoch_selection_matches_final_global_and_conditional_policies(self):
        class Queue(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.register_buffer('logits', torch.logit(torch.tensor(values, dtype=torch.float32)))
            def forward(self, x, *args, **kwargs):
                logits = self.logits[:, None, :].expand(len(x), 6, 4)
                return logits, torch.zeros_like(logits)
        truth = np.tile(np.array([1, 1, 0, 0], np.float32), (6, 1))
        samples = [dict(x=np.zeros((12, 4, 44), np.float32), truth=truth, soft=truth,
                        speed=np.zeros_like(truth), mask=np.ones_like(truth, bool), weight=np.ones_like(truth),
                        target_mask=np.ones_like(truth, bool), baseline=np.zeros_like(truth, bool),
                        condition=kind, date='2031-01-01', origin=12,
                        forecast_origin='2031-01-01T01:00:00Z') for kind in ('onset', 'ongoing')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root, root / 'queue.pt')
            args.model_config, args.use_bf16 = {}, False
            context = {'P_N': dict(panel=types.SimpleNamespace(links=['a', 'b', 'c', 'd'],
                                    segments=((0, 4),), edges=np.empty((2, 0), np.int64)))}
            scores = []
            for values in (([.64, .56, .57, .10],) * 2, ([.85, .80, .75, .70],) * 2,
                           ([.85, .80, .75, .70], [.25, .20, .15, .10])):
                with patch.object(self.learning, 'schedule', return_value=[('P_N', root)]), \
                        patch.object(self.learning, 'prepared_days', return_value=[('P_N', root, samples, 0.)]):
                    report, _ = self.learning.run_period(Queue(values), context, {}, args, 'queue', 'select', 0, 'cpu')
                scores.append(report['proxy_IoU'])
                records = report['daily_records']
                self.assertAlmostEqual(report['proxy_IoU'], self.learning.queue_metrics(records, 'policy')['proxy_IoU'])
                checkpoint = root / 'queue.pt'
                torch.save(dict(threshold=report['threshold']), checkpoint)
                manifest = dict(config=dict(period='select', checkpoint_sha256=self.learning.queue_file_digest(checkpoint)),
                                window_signature='fixed-window-test')
                rows = [dict(panel='P_N', condition=sample['condition'], date=sample['date'],
                             origin=sample['forecast_origin'], probability=np.tile(values[i], (6, 1)),
                             truth=truth.astype(bool), valid=sample['mask'], target_mask=sample['target_mask'],
                             baseline=sample['baseline']) for i, sample in enumerate(samples)]
                selected = self.learning.calibrate_queue(rows, manifest, checkpoint, root / 'calibration', args.queue_thresholds)
                self.assertEqual(selected['policy'], report['policy'])
                self.assertEqual(selected['metrics']['proxy_IoU'], report['proxy_IoU'])
                args.fixed_threshold, args.fixed_policy = report['threshold'], report['policy']
                with patch.object(self.learning, 'schedule', return_value=[('P_N', root)]), \
                        patch.object(self.learning, 'prepared_days', return_value=[('P_N', root, samples, 0.)]), \
                        patch.object(self.learning, 'queue_policy_candidates', side_effect=AssertionError('holdout must not tune')):
                    holdout, _ = self.learning.run_period(Queue(values), context, {}, args, 'queue', 'holdout', 0, 'cpu')
                self.assertEqual(holdout['policy'], report['policy'])
                self.assertEqual(holdout['proxy_IoU'], report['proxy_IoU'])
                del args.fixed_policy
            np.testing.assert_allclose(scores, [2 / 3, 1., 1.])
            self.assertEqual(report['policy']['thresholds'], dict(onset=.8, ongoing=.2))

    def test_balanced_training_save_resume_and_prediction(self):
        with tempfile.TemporaryDirectory() as directory, synthetic_parquet():
            root = Path(directory)
            release = root / 'release'
            release_fixture(release)
            old_path, new_path = root / 'baseline.pt', root / 'v4/queue.pt'
            wavelet_flags = ('--wavelet-levels', '2', '--wavelet-input', 'signals')
            flags = ('--queue-head', 'conditional', '--queue-future-prior', '--queue-label-mode', 'sample',
                     '--queue-balanced', '--queue-resample', *wavelet_flags)
            with contextlib.redirect_stdout(io.StringIO()), patch('torch.cuda.is_available', return_value=False):
                self.learning.train(self.args(release, old_path, '--queue-head', 'pooled-mlp', *wavelet_flags), 'queue')
                before = old_path.read_bytes()
                args = self.args(release, new_path, *flags, '--init-queue', old_path, '--lr', '.0001')
                self.learning.train(args, 'queue')
                self.assertEqual(before, old_path.read_bytes())
                report = json.loads(new_path.with_suffix('.epoch1.json').read_text())
                counts = report['training_group_counts']
                self.assertEqual(set(counts), {'P_N:onset', 'P_N:ongoing'})
                self.assertEqual(len(set(counts.values())), 1)
                saved = torch.load(new_path, map_location='cpu', weights_only=True)
                self.assertIn('policy', saved['metrics'])
                self.assertEqual(saved['metrics']['selection_thresholds'], args.queue_thresholds)
                self.assertEqual(saved['training_args']['queue_label_mode'], 'sample')
                self.assertEqual(saved['model_config']['wavelet_levels'], 2)
                self.assertEqual(saved['model_config']['wavelet_input'], 'signals')
                self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], .0001)
                context = self.learning.contexts(args, 'queue', saved)
                with patch.object(self.learning, 'queue_examples', side_effect=AssertionError('cached candidates must be reused')):
                    items = self.learning.balanced_queue_items(context, args, 1)
                self.assertTrue(items)
                last = new_path.with_name('queue.last.pt')
                self.learning.train(self.args(release, new_path, *flags, '--resume', last, '--epochs', '2'), 'queue')
                saved = torch.load(last, map_location='cpu', weights_only=True)
                self.assertEqual(saved['epoch'], 2)
                self.assertEqual(saved['training_args']['queue_label_mode'], 'sample')
                model = self.learning.model_from_checkpoint(saved, 'cpu').eval()
                for split in ('validation', 'private'):
                    output = root / f'{split}.csv'
                    self.learning.predict_queue_v2(model, saved, release, split, output)
                    frame = pd.read_csv(output)
                    self.assertEqual(len(frame), 12)
                    self.assertTrue(frame.queue_pred.isin([0, 1]).all())
                    original = output.read_bytes()
                    self.learning.predict_queue_v2(model, saved, release, split, output)
                    self.assertEqual(output.read_bytes(), original)
                forced = {**saved, 'threshold': .99, 'metrics': {**saved['metrics'], 'policy': dict(
                    name='fixed-test', thresholds=dict(onset=.99, ongoing=.99), force_nonempty=True)}}
                self.learning.predict_queue_v2(model, forced, release, 'validation', root / 'forced.csv')
                self.assertEqual(int(pd.read_csv(root / 'forced.csv').queue_pred.sum()), 1)
                forced['metrics'].pop('policy')
                self.learning.predict_queue_v2(model, forced, release, 'validation', root / 'legacy.csv')
                self.assertEqual(int(pd.read_csv(root / 'legacy.csv').queue_pred.sum()), 0)
                # An old best score must be measured again before a new epoch is compared with it.
                previous = torch.load(new_path, map_location='cpu', weights_only=True)
                previous['metrics'].pop('policy')
                previous['metrics'].pop('selection_thresholds')
                torch.save(previous, new_path)
                with patch.object(self.learning, 'run_period', wraps=self.learning.run_period) as run:
                    self.learning.train(self.args(release, new_path, *flags, '--resume', last, '--epochs', '3'), 'queue')
                self.assertEqual([call.args[5] for call in run.call_args_list], ['select', 'fit', 'select'])
                self.assertIn('policy', torch.load(new_path, weights_only=True)['metrics'])


if __name__ == '__main__':
    unittest.main()
