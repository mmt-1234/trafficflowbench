"""Contracts for the frozen Task 1/3 corrector, without real release data."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
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
    d['regime'] = np.array(0, np.int32)
    panel = SimpleNamespace(name='P_N', links=['l0'], lengths=np.array([1.]), free_speed=np.array([100.]),
                            capacity=np.array([6000.]), lanes=np.array([2.]))
    context = dict(panel=panel, mapping=np.array([0, 0]), lanes=np.array([2., 3.]),
                   station_graph=np.ones((stations, stations), bool))
    return d, context, np.full((size, stations, 2), 2.5, np.float32)


def statistics(width, bounds=(.1, .2), variance=(.01, .04), n_bound=.2, n_variance=.01):
    return dict(mean=np.zeros(width).tolist(), std=np.ones(width).tolist(), fitted_period='fit',
                projection={'P_N/0': dict(bounds=list(bounds), channel_mse=list(variance),
                                         log_n_bound=n_bound, log_n_mse=n_variance)})


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
        model = self.code['FrozenStateCorrector'](statistics(features))
        projection = self.code['projection_inputs'](d, context, 'cpu')
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(.001)
        before = model(torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive, projection)
        changed = torch.tensor(x)
        changed[4:] += 100
        after = model(changed, torch.tensor(base), torch.tensor(d['target']), consecutive, projection)
        torch.testing.assert_close(before[:4], after[:4], rtol=0, atol=0)

    def test_zero_initialization_bounds_and_target_mask(self):
        d, context, base = fixture()
        x, consecutive = self.code['correction_features'](d, context, base)
        width = len(self.code['FEATURES'])
        model = self.code['FrozenStateCorrector'](statistics(width))
        args = (torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                self.code['projection_inputs'](d, context, 'cpu'))
        torch.testing.assert_close(model(*args), args[1], rtol=0, atol=0)
        (model(*args)[args[2]] - 1).square().mean().backward()
        self.assertGreater(float(model.net[-1].weight.grad.abs().sum()), 0)
        with torch.no_grad():
            model.net[-1].bias.fill_(10)
        output = model(*args)
        torch.testing.assert_close(output[~args[2]], args[1][~args[2]], rtol=0, atol=0)
        self.assertTrue(torch.all((output - args[1]).abs() <= model.bounds[0] + 1e-6))
        torch.testing.assert_close(model(*args, gamma=0), args[1], rtol=0, atol=0)

    def test_occupancy_and_other_link_features_use_only_public_measurements(self):
        d, context, base = fixture()
        context['panel'].links = ['l0', 'l1']
        for name in ('lengths', 'free_speed', 'capacity', 'lanes'):
            setattr(context['panel'], name, np.repeat(getattr(context['panel'], name), 2))
        context['mapping'] = np.array([0, 1])
        names = self.code['FEATURES']
        before, _ = self.code['correction_features'](d, context, base)
        np.testing.assert_allclose(before[:, 0, names.index('neighbor_speed')], d['values'][:, 1, 0])
        self.assertTrue(before[:, 0, names.index('neighbor_speed_valid')].all())
        d['values'][3, 0, 2] = .7
        d['visible'][:, 1, 0] = False
        d['values'][:, 1, 0] = 99999
        after, _ = self.code['correction_features'](d, context, base)
        self.assertAlmostEqual(after[3, 0, names.index('occupancy')], .7)
        self.assertFalse(after[:, 0, names.index('neighbor_speed_valid')].any())
        self.assertFalse(after[:, 0, names.index('neighbor_speed')].any())
        d['visible'][3, 0, 2] = False
        masked, _ = self.code['correction_features'](d, context, base)
        self.assertEqual(masked[3, 0, names.index('occupancy')], 0.)

    def test_link_projection_preserves_observed_stations_and_is_minimum_weighted_step(self):
        d, context, base = fixture()
        config = statistics(len(self.code['FEATURES']), bounds=(100., 100.), n_bound=.1, n_variance=0.)
        model = self.code['FrozenStateCorrector'](config)
        with torch.no_grad():
            model.net[-1].bias.fill_(1.)
        x, consecutive = self.code['correction_features'](d, context, base)
        output = model(torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                       self.code['projection_inputs'](d, context, 'cpu'))
        torch.testing.assert_close(output[~torch.tensor(d['target'])], torch.tensor(base)[~torch.tensor(d['target'])], rtol=0, atol=0)
        before, _, _ = self.code['public_link_fields'](d, context, base)
        after, _, _ = self.code['public_link_fields'](d, context, output.detach().numpy())
        n0 = before[3, 0, 1] / before[3, 0, 0]
        expected = (n0 + 1) * np.exp(.1 * np.tanh(1.)) - 1
        self.assertAlmostEqual(after[3, 0, 1] / after[3, 0, 0], expected, places=4)
        step = output[3, 0].detach().numpy() - base[3, 0]
        covariance = np.array(config['projection']['P_N/0']['channel_mse']) / [.54, .46]
        tangent = np.array([600 * context['lanes'][0], 25 * expected])
        tangent /= np.linalg.norm(tangent)
        cost = lambda delta: np.sum(delta**2 / covariance)
        for amount in (-.1, .1):
            self.assertGreater(cost(step + amount * tangent), cost(step))
        torch.testing.assert_close(torch.tensor(self.code['correct_day'](model, d, context, base, 1.)),
                                   output.detach(), rtol=0, atol=0)

    def test_reliability_protects_the_more_accurate_channel(self):
        d, context, base = fixture()
        x, consecutive = self.code['correction_features'](d, context, base)
        args = (torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                self.code['projection_inputs'](d, context, 'cpu'))
        steps = []
        for variance in ((1e-6, 1.), (1., 1e-6)):
            model = self.code['FrozenStateCorrector'](statistics(len(self.code['FEATURES']),
                bounds=(100., 100.), variance=variance, n_bound=.1, n_variance=0.))
            with torch.no_grad():
                model.net[-1].bias.fill_(1.)
            steps.append((model(*args) - args[1]).abs()[args[2]].mean(0))
        self.assertLess(steps[0][0], steps[1][0])
        self.assertGreater(steps[0][1], steps[1][1])

    def test_uncertain_count_target_shrinks_the_projection(self):
        d, context, base = fixture()
        x, consecutive = self.code['correction_features'](d, context, base)
        args = (torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                self.code['projection_inputs'](d, context, 'cpu'))
        changes = []
        for variance in (0., 1.):
            model = self.code['FrozenStateCorrector'](statistics(len(self.code['FEATURES']),
                bounds=(100., 100.), n_variance=variance))
            with torch.no_grad():
                model.net[-1].bias.fill_(1.)
            changes.append((model(*args) - args[1]).square().sum().detach())
        self.assertGreater(changes[0], changes[1])

    def test_saved_corrector_reloads_the_same_projection_and_rejects_old_features(self):
        d, context, base = fixture()
        saved_stats = statistics(len(self.code['FEATURES']))
        model = self.code['FrozenStateCorrector'](saved_stats)
        with torch.no_grad():
            model.net[-1].bias.fill_(.5)
        payload = dict(model=model.state_dict(), features=self.code['FEATURES'],
                       statistics=saved_stats, contract='test-v2')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corrector.pt'
            torch.save(payload, path)
            loaded = self.code['load_corrector'](path, 'test-v2', 'cpu')
            np.testing.assert_array_equal(self.code['correct_day'](model, d, context, base, 1.),
                                          self.code['correct_day'](loaded, d, context, base, 1.))
            payload['features'] = ['old-v1-feature']
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                self.code['load_corrector'](path, 'test-v2', 'cpu')

    def test_training_uses_final_state_and_count_losses_then_reloads(self):
        names = {'link_mean_numpy', 'consecutive_five_minutes', 'observability_mask', 'completed_link_fields',
                 'training_transitions', 'joint_loss', 'physics_totals', 'state_totals', 'summarize', 'macro_mean'}
        namespace = definitions(embedded('maginet'), dict(np=np, pd=pd, torch=torch,
            LOSS_WEIGHTS=dict(N=.05, dN=.03, FD=0.), SCALE=np.array([25., 600.]),
            SELECT_WEIGHTS=dict(State=.7, FD_proxy=.1, LWR_proxy=.2)), names)
        physical = SimpleNamespace(**{name: namespace[name] for name in names}, DEVICE=torch.device('cpu'),
                                   REGIMES=('R1', 'R2', 'R3'), SCALE=np.array([25., 600.]),
                                   save_checkpoint=torch.save)
        days, entries = {}, []
        for regime in range(3):
            d, context, base = fixture()
            d['regime'] = np.array(regime, np.int32)
            d['truth'] = base.copy()
            d['truth'][d['target']] += [.1, -.1]
            d['active'] = d['target'] | d['visible'][..., :2].all(-1)
            d['observed'] = np.where(d['visible'][..., :2], d['values'][..., :2], 0.) * [25., 1.]
            d['observed'][..., 1] *= 600 * context['lanes']
            d['ti'], d['si'] = np.nonzero(d['target'])
            d['ramp_valid'] = np.ones((len(base), 1), bool)
            d['consecutive'] = np.ones(len(base) - 1, bool)
            days[regime] = d
            entries.append(dict(panel='P_N', regime=regime, period='fit', date=f'2030-06-0{regime+1}'))
        context.update(mode='A', topology_nodes={0})
        context['panel'].edges = np.empty((2, 0), int)
        context['panel'].jam_density = np.array([400.])
        tensors = dict(matrix=torch.ones(2, 1), lanes=torch.tensor(context['lanes'], dtype=torch.float32),
                       link_lanes=torch.tensor([2.]), lengths=torch.tensor([1.]), free=torch.tensor([100.]),
                       capacity=torch.tensor([6000.]), jam=torch.tensor([400.]))
        cache = SimpleNamespace(contexts={'P_N': context}, tensors={'P_N': tensors},
                                get=lambda entry: (days[int(entry['regime'])], base))
        original = self.code['m']
        self.code['m'] = physical
        try:
            stats = self.code['fit_statistics'](entries, cache, 42)
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                self.code['train_variant']('physics', entries, entries, cache, stats,
                    dict(seed=42, epochs=1, steps_per_group=1), 'training-test', folder)
                loaded = self.code['load_corrector'](folder / 'best_State.pt', 'training-test', 'cpu')
                self.assertGreater(float(loaded.net[-1].weight.detach().abs().sum()), 0.)
                final = self.code['correct_day'](loaded, days[0], context, base, 1.)
                self.assertTrue(np.isfinite(final).all())
                history = pd.read_csv(folder / 'history.csv')
                self.assertEqual(int(history.updates.iloc[0]), 3)
                self.assertTrue(np.isfinite(history.loss.iloc[0]))
        finally:
            self.code['m'] = original

    def test_speed_floor_zero_flow_and_zero_bounds_are_finite(self):
        for speed in (0., .5, 1., 1.01):
            for flow in (0., 1000.):
                for bounds in ((100., 100.), (0., 100.)):
                    d, context, base = fixture()
                    d['target'][:] = True
                    d['visible'][..., :2] = False
                    base[..., 0], base[..., 1] = speed / 25, flow / 600
                    model = self.code['FrozenStateCorrector'](statistics(len(self.code['FEATURES']),
                        bounds=bounds, variance=(1., 1e-6), n_bound=2., n_variance=0.))
                    with torch.no_grad():
                        model.net[-1].bias.fill_(1.)
                    x, consecutive = self.code['correction_features'](d, context, base)
                    output = model(torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                                   self.code['projection_inputs'](d, context, 'cpu'))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue((output >= 0).all())
                    self.assertTrue(((output - torch.tensor(base)).abs() <= torch.tensor(bounds) + 1e-6).all())
                    output.square().mean().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                    if speed <= 1 or bounds[0] == 0:
                        torch.testing.assert_close(output[..., 0], torch.tensor(base[..., 0]), rtol=0, atol=0)
                    else:
                        self.assertGreaterEqual(float(output[..., 0].detach().mean()) * 25, 1 - 1e-6)

    def test_projection_statistics_are_fit_only_and_use_normalized_channel_errors(self):
        d, context, base = fixture()
        d['truth'] = base.copy()
        d['truth'][d['target']] += [.2, -.1]
        cache = SimpleNamespace(contexts={'P_N': context}, get=lambda entry: (d, base))
        entry = dict(period='fit', date='2030-06-01', panel='P_N', regime=0)
        saved = self.code['fit_statistics']([entry], cache, 42)
        np.testing.assert_allclose(saved['projection']['P_N/0']['channel_mse'], [.04, .01], rtol=1e-5)
        self.assertGreater(saved['projection']['P_N/0']['log_n_bound'], 0)
        with self.assertRaises(ValueError):
            self.code['fit_statistics']([{**entry, 'date': '2031-01-01'}], cache, 42)
        with self.assertRaises(ValueError):
            self.code['FrozenStateCorrector']({**saved, 'fitted_period': 'early'})

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
            dict(candidate='bad_N', gamma=.25, State=.91, FD_proxy=.9, LWR_proxy=.9),
        ])
        rows['N_MAE_proxy'] = [1., .8, .8, .9, 1.1]
        rows['DN_MAE_proxy'] = [2., 1., 1., 1.9, 1.]
        choose = self.code['choose_candidate']
        self.assertEqual(choose(rows)[0].candidate, 'valid')
        self.assertEqual(choose(rows.iloc[:3])[0].candidate, 'baseline')
        self.assertEqual(choose(rows.iloc[[0, 4]])[0].candidate, 'baseline')
        rows['DN_MAE_proxy'] = np.nan
        self.assertEqual(choose(rows)[0].candidate, 'baseline')

    def test_february_cannot_enter_training_or_selection_periods(self):
        validate = self.code['validate_period']
        for period in ('fit', 'early', 'blend'):
            with self.assertRaises(ValueError):
                validate([dict(period=period, date='2031-02-01')], period)
        validate([dict(period='fit', date='2030-12-31')], 'fit')


if __name__ == '__main__':
    unittest.main()
