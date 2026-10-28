"""Contracts for the frozen Task 1/3 corrector, without real release data."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/task123_colab.ipynb'


def embedded(name):
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(c for c in notebook['cells'] if name in c.get('metadata', {}).get('tags', []))
    return ast.literal_eval(ast.parse(''.join(cell['source'])).body[0].value.args[1])


def definitions(source, namespace, names=None):
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names):
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                ('VERSION', 'GAMMAS', 'ENDPOINT_NAMES', 'FEATURES') for t in node.targets):
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<corrector-contract>', 'exec'), namespace)
    return namespace


def fixture():
    size, stations = 7, 2
    visible = np.ones((size, stations, 3), bool)
    target = np.zeros((size, stations), bool)
    target[2:5, 0] = True
    visible[target, :2] = False
    values = np.ones((size, stations, 3), np.float32)
    values[..., 0] = np.arange(size)[:, None] / 20 + 2
    values[~visible] = 0
    aux = np.ones((size, stations, 21), np.float32)
    d = dict(values=values, visible=visible, target=target, aux=aux,
             base=np.full((size, stations, 2), 2., np.float32),
             timestamps=pd.date_range('2030-06-01', periods=size, freq='5min', tz='UTC').astype(str).tolist(),
             truth=np.full((size, stations, 2), 100.), observed=np.full((size, stations, 2), 200.),
             active=np.zeros((size, stations), bool))
    panel = SimpleNamespace(links=['l0'], lengths=np.array([1.]), free_speed=np.array([100.]),
                            capacity=np.array([6000.]), lanes=np.array([2.]))
    context = dict(panel=panel, mapping=np.array([0, 0]), lanes=np.array([2., 3.]))
    return d, context, np.full((size, stations, 2), 2.5, np.float32)


class PhysicsCorrectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        helpers = definitions(embedded('maginet'), dict(np=np, pd=pd),
                              {'link_mean_numpy', 'consecutive_five_minutes', 'final_state_output'})
        cls.m = SimpleNamespace(**{name: helpers[name] for name in
                                  ('link_mean_numpy', 'consecutive_five_minutes', 'final_state_output')})
        cls.code = definitions(embedded('physics_corrector'),
                               dict(np=np, pd=pd, torch=torch, nn=nn, m=cls.m, Path=Path,
                                    json=json, hashlib=hashlib))

    def test_features_ignore_labels_and_hidden_placeholders(self):
        d, context, base = fixture()
        build = self.code['correction_features']
        before, _ = build(d, context, base)
        d['truth'][:] = np.nan
        d['observed'][:] = -999
        d['active'][:] = True
        d['values'][~d['visible']] = 99999
        after, _ = build(d, context, base)
        np.testing.assert_array_equal(before, after)
        self.assertTrue(np.isfinite(after).all())
        self.assertEqual(after.shape[-1], len(self.code['FEATURES']))
        # Link N must be ratio-of-means using total station flow, not mean of ratios.
        t = 3
        speed = np.mean([base[t, 0, 0] * 25, d['values'][t, 1, 0] * 25])
        flow = np.mean([base[t, 0, 1] * 600 * 2, d['values'][t, 1, 1] * 600 * 3])
        index = self.code['FEATURES'].index('log_N')
        self.assertAlmostEqual(float(after[t, 0, index]), float(np.log1p(flow / speed)), places=5)

    def test_endpoints_and_convolution_do_not_cross_a_timestamp_gap(self):
        d, context, base = fixture()
        d['timestamps'][4:] = pd.date_range('2030-06-01T02:00Z', periods=3, freq='5min').astype(str).tolist()
        x, consecutive = self.code['correction_features'](d, context, base)
        right = self.code['FEATURES'].index('speed_right_valid')
        self.assertEqual(x[3, 0, right], 0)
        features = len(self.code['FEATURES'])
        model = self.code['FrozenStateCorrector'](np.zeros(features), np.ones(features), [.1, .1])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.001)
        before = model(torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive)
        changed = torch.tensor(x)
        changed[4:] += 100
        after = model(changed, torch.tensor(base), torch.tensor(d['target']), consecutive)
        torch.testing.assert_close(before[:4], after[:4], rtol=0, atol=0)

    def test_zero_initialization_bounds_and_target_mask(self):
        d, context, base = fixture()
        x, consecutive = self.code['correction_features'](d, context, base)
        width = len(self.code['FEATURES'])
        model = self.code['FrozenStateCorrector'](np.zeros(width), np.ones(width), [.1, .2])
        args = torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive
        torch.testing.assert_close(model(*args), args[1], rtol=0, atol=0)
        (model(*args)[args[2]] - 1).square().mean().backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.net[-1].bias.fill_(10)
        output = model(*args)
        torch.testing.assert_close(output[~args[2]], args[1][~args[2]], rtol=0, atol=0)
        self.assertTrue(torch.all((output - args[1]).abs() <= model.bounds + 1e-6))
        torch.testing.assert_close(model(*args, gamma=0), args[1], rtol=0, atol=0)

    def test_correction_sees_the_final_blend_exactly_once(self):
        d, _, _ = fixture()
        neural = np.full_like(d['base'], 6.)
        calls = []

        def correct(day, mixed):
            calls.append(mixed.copy())
            mixed[day['target']] += .1
            return mixed

        result = self.m.final_state_output(d, neural, .25, correct)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], 3.)
        np.testing.assert_allclose(result[d['target']], 3.1)
        np.testing.assert_allclose(result[~d['target']], 3.)
        with self.assertRaises(ValueError):
            self.m.final_state_output(d, neural, .25, lambda day, mixed: mixed + 1)

    def test_selection_rejects_state_or_fd_regressions_and_can_keep_baseline(self):
        rows = pd.DataFrame([
            dict(candidate='baseline', gamma=0., State=.9, FD_proxy=.9, LWR_proxy=.7),
            dict(candidate='bad_state', gamma=1., State=.89, FD_proxy=.95, LWR_proxy=.9),
            dict(candidate='bad_fd', gamma=1., State=.91, FD_proxy=.89, LWR_proxy=.9),
            dict(candidate='valid', gamma=.25, State=.9, FD_proxy=.9, LWR_proxy=.71),
        ])
        choose = self.code['choose_candidate']
        self.assertEqual(choose(rows)[0].candidate, 'valid')
        self.assertEqual(choose(rows.iloc[:3])[0].candidate, 'baseline')

    def test_february_cannot_enter_training_or_selection_periods(self):
        validate = self.code['validate_period']
        for period in ('fit', 'early', 'blend'):
            with self.assertRaises(ValueError):
                validate([dict(period=period, date='2031-02-01')], period)
        validate([dict(period='fit', date='2030-12-31')], 'fit')


if __name__ == '__main__':
    unittest.main()
