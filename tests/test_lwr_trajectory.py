"""Contracts for the Task 1/3 Delta-N trajectory, without real release data."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from torch import nn


NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/task123_colab.ipynb'


def embedded(name, path=NOTEBOOK):
    notebook = json.loads(path.read_text())
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
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<trajectory-contract>', 'exec'), namespace)
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


def statistics(width, bounds=(.1, .2), variance=(.01, .04), dn_bound=.2):
    return dict(mean=np.zeros(width).tolist(), std=np.ones(width).tolist(), fitted_period='fit',
                projection={'P_N/0': dict(bounds=list(bounds), channel_mse=list(variance),
                                         dn_bound=dn_bound)})


class LwrTrajectoryTests(unittest.TestCase):
    notebook = NOTEBOOK

    @classmethod
    def setUpClass(cls):
        helpers = definitions(embedded('maginet', cls.notebook), dict(np=np, pd=pd),
                              {'link_mean_numpy', 'consecutive_five_minutes', 'final_state_output'})
        cls.m = SimpleNamespace(**{name: helpers[name] for name in
                                  ('link_mean_numpy', 'consecutive_five_minutes', 'final_state_output')})
        cls.code = definitions(embedded('lwr_trajectory', cls.notebook),
                               dict(np=np, pd=pd, torch=torch, nn=nn, m=cls.m, Path=Path,
                                    json=json, hashlib=hashlib))

    def test_features_ignore_labels_and_hidden_placeholders(self):
        d, context, base = fixture()
        build = self.code['trajectory_features']
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
        x, consecutive = self.code['trajectory_features'](d, context, base)
        right = self.code['FEATURES'].index('speed_right_valid')
        self.assertEqual(x[3, 0, right], 0)
        features = len(self.code['FEATURES'])
        model = self.code['DeltaNTrajectory'](statistics(features))
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
        x, consecutive = self.code['trajectory_features'](d, context, base)
        width = len(self.code['FEATURES'])
        model = self.code['DeltaNTrajectory'](statistics(width))
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
        before, _ = self.code['trajectory_features'](d, context, base)
        np.testing.assert_allclose(before[:, 0, names.index('neighbor_speed')], d['values'][:, 1, 0])
        self.assertTrue(before[:, 0, names.index('neighbor_speed_valid')].all())
        d['values'][3, 0, 2] = .7
        d['visible'][:, 1, 0] = False
        d['values'][:, 1, 0] = 99999
        after, _ = self.code['trajectory_features'](d, context, base)
        self.assertAlmostEqual(after[3, 0, names.index('occupancy')], .7)
        self.assertFalse(after[:, 0, names.index('neighbor_speed_valid')].any())
        self.assertFalse(after[:, 0, names.index('neighbor_speed')].any())
        d['visible'][3, 0, 2] = False
        masked, _ = self.code['trajectory_features'](d, context, base)
        self.assertEqual(masked[3, 0, names.index('occupancy')], 0.)

    def test_link_projection_preserves_observed_stations_and_is_minimum_weighted_step(self):
        d, context, base = fixture()
        config = statistics(len(self.code['FEATURES']), bounds=(100., 100.), dn_bound=.1)
        model = self.code['DeltaNTrajectory'](config)
        with torch.no_grad():
            model.net[-1].bias.fill_(1.)
        x, consecutive = self.code['trajectory_features'](d, context, base)
        output = model(torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                       self.code['projection_inputs'](d, context, 'cpu'))
        torch.testing.assert_close(output[~torch.tensor(d['target'])], torch.tensor(base)[~torch.tensor(d['target'])], rtol=0, atol=0)
        before, _, _ = self.code['public_link_fields'](d, context, base)
        after, _, _ = self.code['public_link_fields'](d, context, output.detach().numpy())
        n0 = before[2, 0, 1] / before[2, 0, 0]
        free = torch.tensor(d['target'].any(1))[:, None]
        delta = self.code['solve_delta_residuals'](torch.full((6, 1), .1 * np.tanh(1.)),
            torch.full((6, 1), torch.sigmoid(torch.tensor(1.)).item()), free)
        expected = n0 + float(delta[2, 0])
        self.assertAlmostEqual(after[2, 0, 1] / after[2, 0, 0], expected, places=4)
        step = output[2, 0].detach().numpy() - base[2, 0]
        covariance = np.array(config['projection']['P_N/0']['channel_mse']) / [.54, .46]
        tangent = np.array([600 * context['lanes'][0], 25 * expected])
        tangent /= np.linalg.norm(tangent)
        cost = lambda delta: np.sum(delta**2 / covariance)
        for amount in (-.1, .1):
            self.assertGreater(cost(step + amount * tangent), cost(step))
        torch.testing.assert_close(torch.tensor(self.code['reconstruct_day'](model, d, context, base, 1.)),
                                   output.detach(), rtol=0, atol=0)

    def test_reliability_protects_the_more_accurate_channel(self):
        d, context, base = fixture()
        x, consecutive = self.code['trajectory_features'](d, context, base)
        args = (torch.tensor(x), torch.tensor(base), torch.tensor(d['target']), consecutive,
                self.code['projection_inputs'](d, context, 'cpu'))
        steps = []
        for variance in ((1e-6, 1.), (1., 1e-6)):
            model = self.code['DeltaNTrajectory'](statistics(len(self.code['FEATURES']),
                bounds=(100., 100.), variance=variance, dn_bound=.1))
            with torch.no_grad():
                model.net[-1].bias.fill_(1.)
            steps.append((model(*args) - args[1]).abs()[args[2]].mean(0))
        self.assertLess(steps[0][0], steps[1][0])
        self.assertGreater(steps[0][1], steps[1][1])

    def test_bridge_matches_dense_solve_has_gradients_and_respects_gaps(self):
        solve = self.code['solve_delta_residuals']
        for size in (1, 2, 7, 12):
            torch.manual_seed(size)
            residual = torch.randn(size - 1, 2, dtype=torch.float64, requires_grad=True)
            weight = torch.rand_like(residual)
            if size > 3:
                weight[size // 2] = 0
            free = torch.ones(size, 2, dtype=torch.bool)
            free[0, 0] = False
            free[-1, 0] = False
            actual = solve(residual, weight, free)
            expected = []
            for link in range(2):
                difference = torch.zeros(size - 1, size, dtype=torch.float64)
                for i in range(size - 1):
                    difference[i, i:i+2] = torch.tensor([-1., 1.])
                matrix = .1 * torch.eye(size, dtype=torch.float64) + difference.T @ torch.diag(weight[:, link]) @ difference
                rhs = difference.T @ (weight[:, link] * residual[:, link])
                indices = torch.where(free[:, link])[0]
                output = torch.zeros(size, dtype=torch.float64)
                if len(indices):
                    output = output.index_copy(0, indices, torch.linalg.solve(matrix[indices][:, indices], rhs[indices]))
                expected.append(output)
            torch.testing.assert_close(actual, torch.stack(expected, 1), rtol=1e-9, atol=1e-9)
            if size > 1:
                actual.square().sum().backward()
                self.assertTrue(torch.isfinite(residual.grad).all())
        r = torch.ones(5, 1)
        w = torch.ones_like(r)
        w[2] = 0
        free = torch.ones(6, 1, dtype=torch.bool)
        before = solve(r, w, free)
        r[3:] = 100
        torch.testing.assert_close(solve(r, w, free)[:3], before[:3])

    def test_saved_trajectory_reloads_the_same_projection_and_rejects_old_features(self):
        d, context, base = fixture()
        saved_stats = statistics(len(self.code['FEATURES']))
        model = self.code['DeltaNTrajectory'](saved_stats)
        with torch.no_grad():
            model.net[-1].bias.fill_(.5)
        payload = dict(version=self.code['VERSION'], model=model.state_dict(), features=self.code['FEATURES'],
                       statistics=saved_stats, contract='test-v3')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.pt'
            torch.save(payload, path)
            loaded = self.code['load_trajectory'](path, 'test-v3', 'cpu')
            np.testing.assert_array_equal(self.code['reconstruct_day'](model, d, context, base, 1.),
                                          self.code['reconstruct_day'](loaded, d, context, base, 1.))
            payload['features'] = ['old-v1-feature']
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                self.code['load_trajectory'](path, 'test-v3', 'cpu')

    def test_training_uses_final_state_and_count_losses_then_reloads(self):
        names = {'link_mean_numpy', 'consecutive_five_minutes', 'observability_mask', 'completed_link_fields',
                 'training_transitions', 'joint_loss', 'physics_totals', 'state_totals', 'summarize', 'macro_mean'}
        namespace = definitions(embedded('maginet', self.notebook), dict(np=np, pd=pd, torch=torch,
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
                self.code['train_trajectory'](entries, entries, cache, stats,
                    dict(seed=42, epochs=1, steps_per_group=1, n_weight=.05, dn_weight=.03), 'training-test', folder)
                loaded = self.code['load_trajectory'](folder / 'best_State.pt', 'training-test', 'cpu')
                self.assertGreater(float(loaded.net[-1].weight.detach().abs().sum()), 0.)
                final = self.code['reconstruct_day'](loaded, days[0], context, base, 1.)
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
                    model = self.code['DeltaNTrajectory'](statistics(len(self.code['FEATURES']),
                        bounds=bounds, variance=(1., 1e-6), dn_bound=2.))
                    with torch.no_grad():
                        model.net[-1].bias.fill_(1.)
                    x, consecutive = self.code['trajectory_features'](d, context, base)
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

    def test_sparse_errors_keep_positive_scale_without_changing_dense_or_zero_cases(self):
        scale = self.code['residual_scale']
        sparse = np.r_[np.zeros(95), np.full(5, -5.)]
        self.assertEqual(float(np.quantile(np.abs(sparse), .9)), 0.)
        self.assertEqual(scale(sparse), 5.)
        self.assertEqual(scale(np.zeros(20)), 0.)
        self.assertEqual(scale([]), 0.)
        dense = np.arange(-10., 10.)
        self.assertEqual(scale(dense), float(np.quantile(np.abs(dense), .9)))
        for value in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                scale([value])

    def test_projection_statistics_are_fit_only_and_use_normalized_channel_errors(self):
        d, context, base = fixture()
        d['truth'] = base.copy()
        d['truth'][d['target']] += [.2, -.1]
        cache = SimpleNamespace(contexts={'P_N': context}, get=lambda entry: (d, base))
        entry = dict(period='fit', date='2030-06-01', panel='P_N', regime=0)
        saved = self.code['fit_statistics']([entry], cache, 42)
        np.testing.assert_allclose(saved['projection']['P_N/0']['channel_mse'], [.04, .01], rtol=1e-5)
        self.assertGreater(saved['projection']['P_N/0']['dn_bound'], 0)
        with self.assertRaises(ValueError):
            self.code['fit_statistics']([{**entry, 'date': '2031-01-01'}], cache, 42)
        with self.assertRaises(ValueError):
            self.code['DeltaNTrajectory']({**saved, 'fitted_period': 'early'})

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


class KaggleLwrTrajectoryTests(LwrTrajectoryTests):
    notebook = NOTEBOOK.with_name('task123_kaggle.ipynb')

    def test_selected_correction_reaches_both_submission_splits_after_blend(self):
        callbacks = {'P_N': object()}
        base_model, contexts, tensors = object(), {'P_N': {}}, {'P_N': {}}
        cache = SimpleNamespace(alpha=.25)
        predict = Mock(side_effect=lambda *args, **kwargs: [f'{args[4]}.csv'])
        load = Mock(return_value=({}, base_model, contexts, tensors, cache, callbacks))
        with patch.dict(self.code, m=SimpleNamespace(predict_split=predict), load_selection=load):
            paths = self.code['predict_selected']('selection.json', 'base.pt', 'release', 'cache')
        self.assertEqual(paths, [Path('validation.csv'), Path('private.csv')])
        self.assertEqual(predict.call_count, 2)
        for call in predict.call_args_list:
            self.assertEqual(call.args[:4], (contexts, base_model, tensors, .25))
            self.assertEqual(call.kwargs, dict(finalprocessors=callbacks))

    def test_notebook_reconstruction_preserves_kaggle_cache_policy(self):
        notebook = json.loads(self.notebook.read_text())
        cell = next(cell for cell in notebook['cells']
                    if 'lwr-reconstruction' in cell.get('metadata', {}).get('tags', []))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'selected_alpha.json').write_text('{}')
            selection = root / 'lwr-trajectory/selection.json'
            for mode in ('off', 'train', 'predict'):
                for use_source, use_checkpoint in ((False, False), (True, False), (True, True)):
                    with self.subTest(mode=mode, source=use_source, checkpoint=use_checkpoint):
                        runtime = SimpleNamespace(load_cache_checkpoint=Mock(return_value='verified'))
                        trajectory = SimpleNamespace(train_and_select=Mock(return_value=selection),
                                                    load_selection=Mock())
                        cache_source = root / 'previous/cache' if use_source else None
                        checkpoint = root / 'previous/base.pt' if use_checkpoint else None
                        namespace = dict(Path=Path, LWR_MODE=mode, LWR_SELECTION=selection,
                            USE_RETRIEVAL=False, MAGINET_DIR=root, MAGINET_CHECKPOINT=root / 'base.pt',
                            MAGINET_CACHE_SOURCE=cache_source, TASK1_CACHE_CHECKPOINT=checkpoint,
                            RELEASE=root / 'release', CACHE_DIR=root / 'cache', CPU_THREADS=2,
                            panel_args=[], selected_panels=['P_N'], LWR_EPOCHS=10,
                            LWR_STEPS_PER_GROUP=4, SEED=42, RUN_FEBRUARY=False,
                            maginet=runtime, lwr_trajectory=trajectory, call_maginet=Mock(),
                            pd=SimpleNamespace(read_csv=Mock()), display=Mock())
                        exec(''.join(cell['source']), namespace)
                        namespace['call_maginet'].assert_not_called()
                        if mode == 'off':
                            self.assertIsNone(namespace['ACTIVE_LWR_SELECTION'])
                            runtime.load_cache_checkpoint.assert_not_called()
                            trajectory.train_and_select.assert_not_called()
                            trajectory.load_selection.assert_not_called()
                            continue
                        self.assertEqual(namespace['ACTIVE_LWR_SELECTION'], selection)
                        self.assertEqual(runtime.CACHE_SOURCE_DIR, cache_source)
                        self.assertEqual(runtime.CACHE_MODE, 'reuse' if use_source else 'build')
                        if use_checkpoint:
                            runtime.load_cache_checkpoint.assert_called_once_with(checkpoint, cache_source)
                            self.assertEqual(runtime.CACHE_CHECKPOINT, 'verified')
                        else:
                            runtime.load_cache_checkpoint.assert_not_called()
                            self.assertIsNone(runtime.CACHE_CHECKPOINT)
                        if mode == 'train':
                            trajectory.train_and_select.assert_called_once()
                            trajectory.load_selection.assert_not_called()
                        else:
                            trajectory.load_selection.assert_called_once()
                            trajectory.train_and_select.assert_not_called()


class TrajectoryCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = embedded('lwr_trajectory')
        names = {'link_mean_numpy', 'consecutive_five_minutes'}
        helpers = definitions(embedded('maginet'), dict(np=np, pd=pd), names)
        cls.helpers = SimpleNamespace(**{name: helpers[name] for name in names})
        cls.code = definitions(source, dict(np=np, pd=pd, torch=torch, nn=nn, m=cls.helpers,
                                           Path=Path, json=json, hashlib=hashlib,
                                           SimpleNamespace=SimpleNamespace))

    def test_runtime_change_invalidates_prediction_cache_and_saved_selection(self):
        model = nn.Linear(1, 1)
        runtime = SimpleNamespace(
            DEVICE=torch.device('cpu'), USE_AMP=False, AMP_DTYPE=torch.float16,
            resolve_release=Path, load_local_checkpoint=lambda *args: (model, {'P_N': {}}, {}),
            checkpoint_folder=lambda path: path.parent, prediction_alpha=lambda *args: 1.,
            file_digest=lambda path: 'base-checkpoint', tensor_context=lambda context: {},
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(self.code, m=runtime):
            checkpoint, release, cache_dir = (Path(directory) / name for name in ('base.pt', 'release', 'cache'))
            first = self.code['open_base'](checkpoint, release, cache_dir)
            first_cache, first_identity = first[3:]
            selection = Path(directory) / 'selection.json'
            selection.write_text(json.dumps(dict(
                version=self.code['VERSION'], selected_on='2031-01-16~31', base=first_identity,
                gamma=0., candidate='baseline', trajectory=None,
            )))
            self.code['load_selection'](selection, checkpoint, release, cache_dir)
            environments = [
                (torch.device('cuda'), True, torch.float16, str(torch.__version__)),
                (torch.device('cuda'), True, torch.bfloat16, str(torch.__version__)),
                (torch.device('cpu'), False, torch.float16, 'different-pytorch-version'),
            ]
            identities = {self.code['digest'](first_identity)}
            for device, amp, dtype, version in environments:
                runtime.DEVICE, runtime.USE_AMP, runtime.AMP_DTYPE = device, amp, dtype
                with patch.object(torch, '__version__', version):
                    changed = self.code['open_base'](checkpoint, release, cache_dir)
                    self.assertNotEqual(first_cache.folder, changed[3].folder)
                    identity = self.code['digest'](changed[4])
                    self.assertNotIn(identity, identities)
                    identities.add(identity)
                    with self.assertRaisesRegex(ValueError, '실행 환경'):
                        self.code['load_selection'](selection, checkpoint, release, cache_dir)

    def test_candidate_inputs_are_built_once_per_day_without_changing_predictions(self):
        d, context, base = fixture()
        d['ti'], d['si'] = np.nonzero(d['target'])
        models = [self.code['DeltaNTrajectory'](statistics(len(self.code['FEATURES'])))
                  for _ in range(4)]
        for index, model in enumerate(models):
            with torch.no_grad():
                model.net[-1].bias.fill_(.2 * (index - 1))
        candidates = [('baseline', None, 0.)] + [
            (f'model_{index}_g{gamma}', model, gamma)
            for index, model in enumerate(models) for gamma in (.25, .5, 1.)]
        expected = [self.code['reconstruct_day'](model, d, context, base, gamma)[d['target']]
                    for _, model, gamma in candidates]
        received = []

        def record_state(truth, prediction):
            received.append(prediction.copy())
            return dict(count=len(prediction))

        def summarize(records):
            frame = pd.DataFrame(records)
            return frame, frame, frame

        physical = SimpleNamespace(**vars(self.helpers), SCALE=np.array([25., 600.]),
                                   state_totals=record_state, physics_totals=lambda *args: {},
                                   summarize=summarize)
        cache = SimpleNamespace(contexts={'P_N': context}, get=lambda entry: (d, base))
        entries = [dict(panel='P_N', regime=0, date=date) for date in ('2031-01-01', '2031-01-02')]
        features = Mock(wraps=self.code['trajectory_features'])
        projection = Mock(wraps=self.code['projection_inputs'])
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                self.code, m=physical, trajectory_features=features, projection_inputs=projection):
            self.code['score_candidates'](entries, cache, candidates, Path(directory), 'early')
            self.assertEqual(features.call_count, len(entries))
            self.assertEqual(projection.call_count, len(entries))
            self.assertEqual(len(received), len(entries) * len(candidates))
            for index, prediction in enumerate(received):
                np.testing.assert_array_equal(prediction, expected[index % len(candidates)] * physical.SCALE)
            features.reset_mock()
            projection.reset_mock()
            self.code['score_candidates'](entries, cache, candidates[:1], Path(directory), 'baseline')
            features.assert_not_called()
            projection.assert_not_called()


if __name__ == '__main__':
    unittest.main()
