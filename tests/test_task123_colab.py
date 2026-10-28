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


class QueueNotebookRegressions:
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
        functions(embedded('learning', self.notebook), ['queue_examples', 'stable_seed', 'iou'], namespace)
        context = dict(panel=panel, release=Path('/unused'), profile=None)
        samples = namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select')
        self.assertTrue(samples)
        self.assertTrue(all(sample['condition'] != 'normal' for sample in samples))
        self.assertEqual([sample['origin'] for sample in samples], [sample['origin'] for sample in
                         namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select')])
        day.speed[:] = 40. / 25
        self.assertEqual(namespace['queue_examples'](context, Path('states_2031-01-01.parquet'), args, 'select'), [])

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


class Task123ColabTests(QueueNotebookRegressions, unittest.TestCase):
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

    def test_gradient_probe_does_not_accumulate_parameter_gradients(self):
        import torch
        namespace = dict(torch=torch, np=np, LOSS_WEIGHTS={'N': .05, 'dN': .2})
        probe = functions(embedded('maginet'), ['loss_gradient_diagnostics'], namespace)['loss_gradient_diagnostics']
        parameter = torch.tensor([2.], requires_grad=True)
        losses = dict(State=parameter.square().sum(), N=-parameter.sum(), dN=2 * parameter.sum())
        result = probe(losses, [parameter], .5)
        self.assertIsNone(parameter.grad)
        self.assertAlmostEqual(result['cos_State_N'], -1.)
        self.assertAlmostEqual(result['cos_State_dN'], 1.)
        self.assertAlmostEqual(result['weighted_grad_physics'], .175, places=6)
        (losses['State'] + .5 * (.05 * losses['N'] + .2 * losses['dN'])).backward()
        self.assertAlmostEqual(float(parameter.grad[0]), 4.175, places=5)


class Task123KaggleTests(QueueNotebookRegressions, unittest.TestCase):
    notebook = KAGGLE_NOTEBOOK


if __name__ == '__main__':
    unittest.main()
