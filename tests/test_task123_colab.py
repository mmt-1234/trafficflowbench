"""Focused regressions for the standalone Task123 notebooks; no real data or GPU."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/task123_colab.ipynb'
KAGGLE_NOTEBOOK = NOTEBOOK.with_name('task123_kaggle.ipynb')


def tagged(tag, path=NOTEBOOK):
    notebook = json.loads(path.read_text())
    return ''.join(next(cell['source'] for cell in notebook['cells']
                        if tag in cell.get('metadata', {}).get('tags', [])))


def embedded(name, path=NOTEBOOK):
    call = ast.parse(tagged(name, path)).body[0].value
    return ast.literal_eval(call.args[1])


def functions(source, names, namespace):
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook-functions>', 'exec'), namespace)
    return namespace


class Task123NotebookRegressions:
    def test_cells_and_embedded_source_hashes(self):
        notebook = json.loads(self.notebook.read_text())
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                tree = ast.parse(''.join(cell['source']), filename=f'cell-{index}')
                if 'embedded-source' in cell.get('metadata', {}).get('tags', []):
                    name, source = (ast.literal_eval(arg) for arg in tree.body[0].value.args)
                    self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                                     notebook['metadata']['trafficflow']['source_sha256'][name])
                    compile(ast.parse(source), name, 'exec')

    def test_notebook_has_only_train_and_predict_modes(self):
        notebook = json.loads(self.notebook.read_text())
        self.assertNotIn('smoke', ''.join(''.join(cell['source']) for cell in notebook['cells']).lower())
        namespace = {}
        exec(tagged('configuration', self.notebook), namespace)
        self.assertEqual(namespace['RUN_MODE'], 'train')
        self.assertIn('--queue-balanced', namespace['QUEUE_FLAGS'])
        validation = next(node for node in ast.parse(tagged('environment', self.notebook)).body
                          if isinstance(node, ast.If) and ast.unparse(node.test).startswith('RUN_MODE not in'))
        validation = compile(ast.Module(body=[validation], type_ignores=[]), '<mode-validation>', 'exec')
        for mode in ('train', 'predict'):
            exec(validation, {'RUN_MODE': mode})
        with self.assertRaises(ValueError):
            exec(validation, {'RUN_MODE': 'smoke'})

    def test_train_cells_preserve_configured_budgets_and_queue_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = {}
            exec(tagged('configuration', self.notebook), namespace)
            calls = {}
            def record(task, arguments):
                calls[task] = arguments
                checkpoint = namespace['QUEUE_CHECKPOINT' if task == 'queue' else 'MAGINET_CHECKPOINT']
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b'test checkpoint')
            namespace.update(RELEASE=root / 'release', CACHE_DIR=root / 'cache', MAGINET_DIR=root / 'state',
                             QUEUE_CHECKPOINT=root / 'queue/queue.pt', QUEUE_INIT_CHECKPOINT=root / 'old/queue.pt',
                             EFFECTIVE_BACKEND='scan', QUEUE_PRECISION='fp32', QUEUE_EPOCHS=3, MAGINET_EPOCHS=4,
                             queue_eval_args=['--validation-windows', '5'], panel_args=[],
                             training_action=lambda *args: 'new',
                             call_queue=lambda args: record('queue', args),
                             call_maginet=lambda args: record('state', args))
            for tag in ('train-state', 'train-queue'):
                exec(tagged(tag, self.notebook), namespace)
            for task, budget in (('queue', 3), ('state', 4)):
                arguments = calls[task]
                self.assertEqual(arguments[arguments.index('--epochs') + 1], budget)
                self.assertNotIn('--max-days', arguments)
                self.assertNotIn('--days-per-regime', arguments)
                self.assertNotIn('--width', arguments)
            self.assertIn('--queue-balanced', calls['queue'])
            self.assertIn('--queue-public-contract', calls['queue'])
            self.assertEqual(calls['queue'][calls['queue'].index('--init-queue') + 1], root / 'old/queue.pt')

    def test_nonempty_adds_only_one_target_and_keeps_existing_positives(self):
        decide = functions(embedded('learning', self.notebook), ['queue_decisions'], {'np': np})['queue_decisions']
        probability = np.array([[.1, .4], [.3, .99]])
        valid = np.array([[True, True], [True, False]])
        predicted = decide(probability, .5, valid, True)
        self.assertEqual(int(predicted.sum()), 1)
        self.assertTrue(predicted[0, 1])
        expected = probability >= .2
        np.testing.assert_array_equal(decide(probability, .2, np.ones_like(valid), True), expected)
        self.assertFalse(decide(probability, .5, np.zeros_like(valid), True).any())

    def test_bf16_resume_rejects_unsupported_device_and_inference_uses_fp32(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: False))
        helper = functions(embedded('learning', self.notebook), ['checkpoint_bf16'], {'torch': fake_torch})['checkpoint_bf16']
        checkpoint = {'training_args': {'use_bf16': True}}
        with self.assertRaises(ValueError):
            helper(checkpoint, 'cuda', strict=True)
        self.assertFalse(helper(checkpoint, 'cuda'))
        self.assertFalse(helper(checkpoint, 'cpu'))

    def test_public_queue_coverage_does_not_count_unobserved_connectors(self):
        count = 36
        panel = SimpleNamespace(name='P_N', links=['observed', 'connector1', 'connector2'],
                                cutoffs=np.full(3, 60.))
        speed = np.full((count, 1), 90. / 25)
        speed[20:] = 40. / 25
        day = SimpleNamespace(timestamps=[f'2031-01-01T{slot // 12:02d}:{slot % 12 * 5:02d}:00Z'
                                          for slot in range(count)], station_to_link=np.array([0]),
                              speed=speed, eligible=np.ones((count, 1), bool))
        args = SimpleNamespace(queue_public_contract=True, history_coverage=.5, future_coverage=.5,
                               queue_resample=False, queue_validation_spread=True, windows_per_day=3,
                               normal_windows=1, window_gap_minutes=360, seed=42, queue_origin=False,
                               queue_changes=False, label_temperature=0., uncertainty_margin=0.)
        contract = dict(minimum_history_coverage=.7, maximum_persistence_iou=.9, minimum_spacing_minutes=360)
        namespace = dict(np=np, hashlib=hashlib, QUEUE_PUBLIC_CONTRACT=contract,
                         StationDay=SimpleNamespace(load=lambda path, p: day),
                         aggregate_channels=lambda *a: (np.zeros((count, 3, 3)), None, None),
                         ramp_features=lambda *a: np.zeros((count, 3, 4)),
                         slice_day=lambda *a: None, queue_features=lambda *a: np.zeros((12, 3, 44)))
        functions(embedded('learning', self.notebook), ['queue_examples', 'choose_queue_origins', 'stable_seed', 'iou'], namespace)
        context = dict(panel=panel, release=Path('/unused'), profile=None)
        samples = namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select')
        self.assertTrue(samples)
        self.assertTrue(all(sample['condition'] != 'normal' for sample in samples))
        self.assertEqual([sample['origin'] for sample in samples], [sample['origin'] for sample in
                         namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select')])
        args.queue_sampling_version = 3
        fit = namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'fit')
        self.assertTrue(fit)
        self.assertTrue(all(not sample['mask'][:, 1:].any() for sample in fit))
        # A checkpoint without the version field keeps its original fit coverage.
        del args.queue_sampling_version
        self.assertEqual(namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'fit'), [])
        day.speed[:] = 40. / 25
        self.assertEqual(namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select'), [])

    def test_fit_queue_conditions_use_target_history_and_include_recurrence(self):
        count = 36
        panel = SimpleNamespace(name='P_N', links=['target', 'other', 'connector'], cutoffs=np.full(3, 60.))
        day = SimpleNamespace(timestamps=[f'2031-01-01T{slot // 12:02d}:{slot % 12 * 5:02d}:00Z'
                                          for slot in range(count)], station_to_link=np.array([0, 1]),
                              speed=np.full((count, 2), 90. / 25), eligible=np.ones((count, 2), bool))
        day.speed[20:, 0] = 40. / 25
        day.speed[:, 1] = 40. / 25
        args = SimpleNamespace(queue_sampling_version=3, queue_public_contract=True,
                               history_coverage=.5, future_coverage=.5, queue_resample=False,
                               queue_validation_spread=True, queue_condition_gap=True, windows_per_day=count,
                               normal_windows=0, window_gap_minutes=0, seed=42, queue_origin=False,
                               queue_changes=False, label_temperature=0., uncertainty_margin=0.)
        namespace = dict(np=np, hashlib=hashlib,
                         QUEUE_PUBLIC_CONTRACT=dict(minimum_history_coverage=.7, maximum_persistence_iou=.9,
                                                    minimum_spacing_minutes=360),
                         StationDay=SimpleNamespace(load=lambda path, p: day),
                         aggregate_channels=lambda *a: (np.zeros((count, 3, 3)), None, None),
                         ramp_features=lambda *a: np.zeros((count, 3, 4)),
                         slice_day=lambda *a: None, queue_features=lambda *a: np.zeros((12, 3, 44)))
        functions(embedded('learning', self.notebook), ['queue_examples', 'choose_queue_origins', 'stable_seed', 'iou'], namespace)
        context = dict(panel=panel, release=Path('/unused'), profile=None, queue_target_nodes=np.array([0]))
        path = Path('states_2031-01-01.parquet')
        samples = namespace['queue_examples'](context, path, args, 'fit')
        self.assertTrue(any(sample['condition'] == 'onset' for sample in samples))
        self.assertTrue(all(sample['mask'][:, 0].all() and not sample['mask'][:, 1:].any() for sample in samples))

        day.speed[:] = 90. / 25
        day.speed[:2, 0] = 40. / 25
        day.speed[16:, 0] = 40. / 25
        samples = namespace['queue_examples'](context, path, args, 'fit')
        recurrence = next(sample for sample in samples if sample['origin'] == 12)
        self.assertEqual(recurrence['condition'], 'ongoing')
        public = namespace['queue_examples'](context, path, args, 'select')
        self.assertTrue(public)
        self.assertTrue(any(sample['condition'] == 'ongoing' and sample['origin'] in (12, 13) for sample in public))
        args.queue_sampling_version = 2
        self.assertNotIn(12, [sample['origin'] for sample in namespace['queue_examples'](context, path, args, 'fit')])

    def test_holdout_cannot_select_thresholds(self):
        helper = functions(embedded('learning', self.notebook), ['calibrate_queue'], {})['calibrate_queue']
        with self.assertRaisesRegex(ValueError, '1월'):
            helper([], {'config': {'period': 'holdout'}}, None, None, [.5])

    def test_nonempty_policy_cannot_use_future_eligibility_to_choose_a_cell(self):
        import pandas as pd
        namespace = dict(np=np, pd=pd)
        functions(embedded('learning', self.notebook), ['queue_decisions', 'queue_policy_metrics', 'queue_metrics',
                                         'family_mean', 'iou'], namespace)
        row = dict(panel='P_N', condition='onset', date='2031-01-01', origin='2031-01-01T01:00:00Z',
                   probability=np.array([[.2, .9]] * 6), truth=np.array([[True, False]] * 6),
                   valid=np.array([[True, False]] * 6), target_mask=np.ones((6, 2), bool),
                   baseline=np.zeros((6, 2), bool))
        metrics, details, _ = namespace['queue_policy_metrics'](
            [row], dict(thresholds={'onset': .5}, force_nonempty=True))
        self.assertEqual(metrics['proxy_IoU'], 0.)
        self.assertFalse(details.empty_before.iloc[0])

    def test_current003_is_fixed_without_weight_or_gradient_options(self):
        namespace = {}
        exec(tagged('configuration', self.notebook), namespace)
        self.assertNotIn('PHYSICS_PRESET', namespace)
        self.assertNotIn('PHYSICS_PRESETS', namespace)
        self.assertNotIn('GRADIENT_EVERY', namespace)
        code = embedded('maginet', self.notebook)
        assignment = next(node for node in ast.parse(code).body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'LOSS_WEIGHTS' for target in node.targets))
        values = {}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<physics-weights>', 'exec'), values)
        self.assertEqual(values['LOSS_WEIGHTS'], dict(N=.05, dN=.03, FD=0.))
        for removed in ('--n-weight', '--dn-weight', '--gradient-every', 'loss_gradient_diagnostics'):
            self.assertNotIn(removed, code)
        self.assertIn('-current003', tagged('environment', self.notebook))

    def test_obsolete_comparisons_and_fast_state_pipeline_are_removed(self):
        namespace = {}
        exec(tagged('configuration', self.notebook), namespace)
        self.assertNotIn('QUEUE_REFERENCE_SUBMISSION', namespace)
        self.assertNotIn('--experiment', tagged('train-state', self.notebook))
        definitions = {}
        for name in ('data', 'fast_model', 'learning', 'run', 'maginet'):
            definitions[name] = {node.name for node in ast.parse(embedded(name, self.notebook)).body
                                 if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        self.assertTrue({'train_files', 'day_masks', 'queue_windows', 'mean_iou', 'fit_state',
                         'predict_state', 'make_submission'}.isdisjoint(definitions['run']))
        self.assertTrue({'StateModel', 'TrafficStateModel', 'FASTEncoder'}.isdisjoint(definitions['fast_model']))
        self.assertTrue({'state_examples', 'state_loss', 'state_metrics', 'predict_state_v2'}.isdisjoint(definitions['learning']))
        self.assertTrue({'EXPERIMENTS', 'LearnedSpatialConv', 'resample_fit_mask'}.isdisjoint(definitions['maginet']))
        self.assertNotIn('load_best_maginet', definitions['maginet'])
        self.assertNotIn('AdaptiveGraphResidual', definitions['fast_model'])
        self.assertIn('WaveletResidual', definitions['fast_model'])
        self.assertIn('TrafficQueueModel', definitions['fast_model'])
        self.assertIn('queue_examples', definitions['learning'])
        code = embedded('maginet', self.notebook)
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'save_checkpoint':
                self.assertNotIn('best_state_maginet.pt', ast.unparse(node))

    def test_checkpoint_restore_accepts_only_the_current_maginet_architecture(self):
        restore = functions(embedded('maginet', self.notebook), ['restore_settings'], {'SETTING_NAMES': {}})['restore_settings']
        improvements = dict(temporal_direct=True, learned_graph=True, mask_resample=None, spatial_mode='channel')
        restore(dict(architecture='maginet-task13-residual-v2', improvements=improvements))
        for change in (dict(temporal_direct=False), dict(learned_graph=False),
                       dict(mask_resample='fit-link-v2'), dict(spatial_mode='legacy'), dict(spatial_mode='context')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'channel_graph'):
                restore(dict(architecture='maginet-task13-residual-v2', improvements={**improvements, **change}))


class Task123ColabTests(Task123NotebookRegressions, unittest.TestCase):
    notebook = NOTEBOOK

    def test_interrupted_restore_can_retry_without_partial_reports(self):
        namespace = functions(tagged('environment'), ['restore_reports'],
                              dict(Path=Path, shutil=shutil, tempfile=tempfile))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'previous/reports', root / 'current/reports'
            source.mkdir(parents=True)
            (source / 'model.pt').write_bytes(b'complete checkpoint')

            def interrupt(previous, pending, **kwargs):
                (pending / 'model.pt').write_bytes(b'partial')
                raise InterruptedError('interrupted copy')

            with patch.object(shutil, 'copytree', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    namespace['restore_reports'](source, target)
            self.assertFalse(target.exists())
            namespace['restore_reports'](source, target)
            self.assertEqual((target / 'model.pt').read_bytes(), b'complete checkpoint')
            with self.assertRaises(FileExistsError):
                namespace['restore_reports'](source, target)


class Task123KaggleTests(Task123NotebookRegressions, unittest.TestCase):
    notebook = KAGGLE_NOTEBOOK

    def test_task1_profile_is_read_from_previous_cache_without_recomputing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / 'previous', root / 'current'
            panel = SimpleNamespace(name='P_N', links=['l0'])
            targets = SimpleNamespace(stations=['s0'], links=['l0'])
            dates = ['2030-06-01']
            metadata = dict(format='maginet-input-v1', panel='P_N', links=['l0'], stations=['s0'],
                            station_links=['l0'], fit_dates=dates, aux_names=['quality'], scale=[25., 600.])
            key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
            folder = previous / 'metadata/profiles/P_N' / key
            folder.mkdir(parents=True)
            profile = np.ones((2016, 1, 3), np.float32)
            path = folder / 'fit_profile.npy'
            np.save(path, profile)
            (folder / 'cache_metadata.json').write_text(json.dumps(metadata))
            marker = path.with_suffix('.sources.json')
            marker.write_text(json.dumps('same-input'))
            before = {p: p.read_bytes() for p in folder.iterdir()}
            namespace = dict(np=np, json=json, hashlib=hashlib, shutil=shutil,
                             CACHE_METADATA_DIR=current / 'metadata', CACHE_ROOT=current,
                             CACHE_CONTAINER=current, CACHE_PROFILE_DIR=current / 'legacy_profiles',
                             CACHE_SOURCE_DIR=previous, CACHE_CHECKPOINT=None, CACHE_MODE='reuse', AUX_NAMES=('quality',),
                             SCALE=np.array([25., 600.]), profile_source_signature=lambda *a: 'same-input',
                             fit_prior=lambda *a: self.fail('Previous profile must not be recomputed'))
            functions(embedded('maginet', self.notebook), ['load_fit_profile', 'cache_is_current'], namespace)
            np.testing.assert_array_equal(namespace['load_fit_profile'](panel, targets, dates, {}), profile)
            self.assertEqual(before, {p: p.read_bytes() for p in folder.iterdir()})
            self.assertFalse(list(current.rglob('*.npy')))
            marker.write_text(json.dumps('different-input'))
            with self.assertRaisesRegex(ValueError, '재사용 가능한 과거 평균'):
                namespace['load_fit_profile'](panel, targets, dates, {})
            marker.write_text(json.dumps('same-input'))
            (folder / 'cache_metadata.json').write_text(json.dumps({**metadata, 'fit_dates': ['2031-01-01']}))
            with self.assertRaisesRegex(ValueError, '학습 날짜'):
                namespace['load_fit_profile'](panel, targets, dates, {})

    def test_task1_day_cache_uses_previous_npz_and_rejects_missing_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / 'previous', root / 'current'
            folder = previous / 'inputs/key/P_N/fit'
            folder.mkdir(parents=True)
            path = folder / '2030-06-01.npz'
            np.savez_compressed(path, target=np.ones((1, 1), bool))
            marker = path.with_suffix('.sources.json')
            marker.write_text(json.dumps('same-input'))
            before = path.read_bytes()
            namespace = dict(np=np, json=json, CACHE_SOURCE_DIR=previous, CACHE_CHECKPOINT=None, CACHE_MODE='reuse',
                             TRAIN_MODEL=False, MASK_RESAMPLE=False,
                             time=SimpleNamespace(perf_counter=lambda: 0.),
                             day_source_signature=lambda *a: 'same-input',
                             validate_cache_targets=lambda *a: ('R1', 1),
                             make_day=lambda *a: self.fail('Previous NPZ must not be regenerated'))
            functions(embedded('maginet', self.notebook), ['cache_context', 'cache_is_current'], namespace)
            context = dict(panel=SimpleNamespace(name='P_N'), dates={'fit': ['2030-06-01']}, cache_key='key')
            entries = namespace['cache_context'](context, current, periods=('fit',))
            self.assertEqual(entries[0]['path'], str(path))
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(current.rglob('*.npz')))
            marker.unlink()
            with self.assertRaisesRegex(FileNotFoundError, '출처'):
                namespace['cache_context'](context, current, periods=('fit',))

    def test_legacy_native_cache_recovers_profile_and_verifies_checkpoint_input_hash(self):
        import re
        import torch
        import zipfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / 'previous', root / 'current'
            dates = dict(fit=['2030-06-01'], early=['2031-01-01'], blend=['2031-01-19'])
            panel = SimpleNamespace(name='P_N', links=['l0'])
            targets = SimpleNamespace(stations=['s0'], links=['l0'],
                archive={date: np.array([0]) for days in dates.values() for date in days},
                regimes={date: 'R1' for days in dates.values() for date in days})
            profile = np.ones((2016, 1, 3), np.float32)
            packed = dict(network={'links': panel.links}, stations=targets.stations,
                          station_links=targets.links, dates=dates, profile=torch.from_numpy(profile),
                          cache_key='2b519739ec38ca9ff744')
            entries = []
            for period, days in dates.items():
                path = previous / 'inputs' / packed['cache_key'] / 'P_N' / period / f'{days[0]}.npz'
                path.parent.mkdir(parents=True)
                np.savez_compressed(path, values=np.ones((1, 1, 3)), visible=np.ones((1, 1, 3), bool),
                    aux=np.ones((1, 1, 21), np.float32), slot=np.array([0]), weekday=np.array([0]),
                    base=np.ones((1, 1, 2)), truth=np.ones((1, 1, 2)), target=np.ones((1, 1), bool),
                    active=np.ones((1, 1), bool), observed=np.ones((1, 1, 2)),
                    timestamps=np.array([f'{days[0]}T00:00:00Z']), ti=np.array([0]), si=np.array([0]),
                    regime=np.array(0), ramp_valid=np.ones((1, 1), bool), consecutive=np.zeros(0, bool))
                entries.append(dict(panel='P_N', period=period, date=days[0], path=str(path)))
            settings = dict(release='old-location', architecture='maginet-task13-residual-v2',
                            aux_names=['quality'], output_scale=[25., 600.],
                            panels={'P_N': dict(stations=targets.stations, links=targets.links, dates=dates,
                                               profile_hash=hashlib.sha256(profile.tobytes()).hexdigest())})
            namespace = dict(np=np, json=json, hashlib=hashlib, torch=torch, re=re, zipfile=zipfile,
                             CACHE_CHECKPOINT=None, CACHE_SOURCE_DIR=previous, CACHE_MODE='reuse',
                             AUX_NAMES=('quality',), SCALE=np.array([25., 600.]),
                             REGIMES=('R1', 'R2', 'R3'),
                             BATCH_POLICY='fixed', PANEL_BATCH={}, DAYS_PER_REGIME=None,
                             notebook_signature=lambda *a: (None, {**settings}),
                             network_hashes=lambda *a: {'P_N': 'same-network'},
                             TRAIN_MODEL=False, MASK_RESAMPLE=False,
                             time=SimpleNamespace(perf_counter=lambda: 0.),
                             day_source_signature=lambda *a: self.fail('Legacy reuse must not rescan raw days'),
                             fit_prior=lambda *a: self.fail('Stored profile must not be recomputed'),
                             make_day=lambda *a: self.fail('Stored NPZ must not be regenerated'))
            functions(embedded('maginet', self.notebook),
                      ['load_cache_checkpoint', 'checkpoint_fit_profile', 'load_fit_profile',
                       'cache_context', 'cache_is_current', 'run_signature', 'validate_cache_targets'], namespace)
            signature, saved_settings = namespace['run_signature']({}, entries, {})
            checkpoint = root / 'best_maginet.pt'
            payload = dict(format_version=3, settings=saved_settings, signature=signature, contexts={'P_N': packed})
            torch.save(payload, checkpoint)
            before = {p: p.read_bytes() for p in previous.rglob('*') if p.is_file()}
            namespace['CACHE_CHECKPOINT'] = namespace['load_cache_checkpoint'](checkpoint, previous)
            np.testing.assert_array_equal(namespace['load_fit_profile'](panel, targets, dates['fit'], {}), profile)
            # A new panel subset may generate a different destination key; the saved key still locates old NPZs.
            context = dict(panel=panel, targets=targets, dates=dates, profile=profile,
                           mapping=np.array([0]), cache_key='new-key')
            with np.load(entries[0]['path'], allow_pickle=False) as data:
                archive_type = type(data)
            read_array, reads = archive_type.__getitem__, []
            def tracked_read(archive, key):
                reads.append(key)
                return read_array(archive, key)
            with patch.object(archive_type, '__getitem__', tracked_read):
                reused = namespace['cache_context'](context, current)
            self.assertIn('target', reads)
            self.assertNotIn('aux', reads)
            self.assertNotIn('weekday', reads)
            self.assertNotIn('values', reads)
            self.assertEqual([e['path'] for e in reused], [e['path'] for e in entries])
            self.assertEqual(before, {p: p.read_bytes() for p in previous.rglob('*') if p.is_file()})
            self.assertFalse(list(current.rglob('*.npz')))
            self.assertFalse(list(current.rglob('*.npy')))
            # Without the verified-snapshot fast path, the full profile check remains required.
            with np.load(entries[0]['path'], allow_pickle=False) as data:
                with self.assertRaisesRegex(ValueError, '과거 평균 특징'):
                    namespace['validate_cache_targets'](data, {**context, 'profile': profile * 2}, dates['fit'][0])
            targets.archive[dates['fit'][0]] = np.array([1])
            with self.assertRaisesRegex(ValueError, '대상 셀'):
                namespace['cache_context'](context, current)
            targets.archive[dates['fit'][0]] = np.array([0])
            with self.assertRaisesRegex(ValueError, 'fit 날짜'):
                namespace['load_fit_profile'](panel, targets, ['2030-06-02'], {})
            bad_profile = namespace['CACHE_CHECKPOINT']['contexts']['P_N']['profile']
            bad_profile[0, 0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, '과거 평균'):
                namespace['load_fit_profile'](panel, targets, dates['fit'], {})
            bad_profile[0, 0, 0] = 1.
            namespace['network_hashes'] = lambda *a: {'P_N': 'different-network'}
            with self.assertRaisesRegex(ValueError, '도로 구성'):
                namespace['cache_context'](context, current)
            namespace['network_hashes'] = lambda *a: {'P_N': 'same-network'}
            # Template validation must still reject an incompatible cache after snapshot verification.
            def reject_targets(*args):
                raise ValueError('잘못된 템플릿')
            namespace['validate_cache_targets'] = reject_targets
            with self.assertRaisesRegex(ValueError, '템플릿'):
                namespace['cache_context'](context, current)
            path = Path(entries[0]['path'])
            np.savez_compressed(path, target=np.zeros((1, 1), bool))
            with self.assertRaisesRegex(ValueError, '입력 hash'):
                namespace['load_cache_checkpoint'](checkpoint, previous)
            path.unlink()
            with self.assertRaisesRegex(FileNotFoundError, '입력 캐시'):
                namespace['load_cache_checkpoint'](checkpoint, previous)
            payload['signature'] = 'invalid'
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, '설정 서명'):
                namespace['load_cache_checkpoint'](checkpoint, previous)


if __name__ == '__main__':
    unittest.main()
