"""Local MagiNet migration: notebook contracts, portable inference, and resume."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import maginet as m
from historical_retrieval import HistoricalRetrieval, METHODS
from run import external_state_inputs


def release_fixture(root):
    network = root / 'corridors' / 'P_N' / 'network'
    network.mkdir(parents=True)
    pd.DataFrame(dict(link_id=['l0', 'l1'], lanes=[2., 2.], length_km=[1., 1.],
                      free_speed_kmh=[100., 100.], capacity_vph=[3600., 3600.])).to_csv(network / 'links.csv', index=False)
    pd.DataFrame(dict(link_id=['l0', 'l1'], incoming_link_ids=['', 'l0'],
                      outgoing_link_ids=['l1', ''])).to_csv(network / 'lwr_mainline_topology.csv', index=False)
    pd.DataFrame(dict(link_id=['l0', 'l1'], station_id=['001', '002'], lanes=[2., 2.],
                      free_speed_kmh=[100., 100.], capacity_vph=[3600., 3600.],
                      k_jam=[200., 200.])).to_csv(network / 'fd_parameters.csv', index=False)
    train_dates = ['2030-06-0' + str(i) for i in range(1, 7)]
    train_dates += ['2031-01-0' + str(i) for i in range(1, 4)]
    train_dates += ['2031-01-' + str(i) for i in range(16, 19)]
    train_dates += ['2031-02-0' + str(i) for i in range(1, 4)]
    for split, dates in (('train', train_dates), ('validation', ['2031-03-01']), ('private', ['2031-04-01'])):
        targets = []
        for number, date in enumerate(dates):
            regime = m.REGIMES[number % 3]
            times = pd.date_range(date, periods=12, freq='5min', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
            frame = pd.DataFrame(dict(timestamp=np.repeat(times, 2), station_id=['001', '002'] * 12,
                                      link_id=['l0', 'l1'] * 12, speed_kmh=75. + number + np.arange(24) * .1,
                                      flow_vph=1200. + np.arange(24), occupancy=.1,
                                      is_score_eligible=True, pct_observed=100.))
            chosen = (frame.station_id == '001') & frame.timestamp.isin(times[2:6])
            target = frame.loc[chosen, ['timestamp', 'station_id', 'link_id']].copy()
            target['panel'], target['mask_regime'] = 'P_N', regime
            targets.append(target[m.KEYS])
            if split == 'train':
                original = root / 'corridors' / 'P_N' / split / 'mainline_states'
                original.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(original / f'states_{date}.parquet', index=False)
            masked = root / 'corridors' / 'P_N' / split / 'mainline_states_masked' / f'mask_regime={regime}'
            masked.mkdir(parents=True, exist_ok=True)
            frame.loc[chosen, ['speed_kmh', 'flow_vph']] = np.nan
            frame.to_parquet(masked / f'states_{date}.parquet', index=False)
        template = root / 'task1' / 'P_N' / split
        template.mkdir(parents=True, exist_ok=True)
        pd.concat(targets, ignore_index=True).to_csv(template / 'sample_submission_state.csv', index=False)


class MaginetLocal(unittest.TestCase):
    def test_context_gate_uses_public_neighbors_and_handles_empty_context(self):
        with patch.dict(m.__dict__, WIDTH=4):
            visible = torch.zeros(1, 2, 3, 3, dtype=torch.bool)
            visible[0, 0, 0, :2] = True
            visible[0, 0, 2, 2] = True
            aux = torch.zeros(1, 2, 3, len(m.AUX_NAMES))
            aux[..., 4:8] = 4
            neighbors = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
            context = m.observation_context(visible, aux, neighbors)
            self.assertEqual(tuple(context.shape), (1, 3, 2, 10))
            self.assertEqual(context[0, 0, 0, 3].item(), 0.)  # self is excluded
            self.assertEqual(context[0, 1, 0, 3].item(), .5)
            self.assertEqual(context[0, 1, 0, 4].item(), 1.)
            self.assertEqual(context[0, 2, 0, 5].item(), 0.)  # no physical neighbors
            self.assertTrue(torch.equal(context[..., 6:], torch.ones_like(context[..., 6:])))
            layer = m.LearnedSpatialConv(contextual=True)
            with torch.no_grad():
                layer.project.weight.copy_(torch.eye(4))
                layer.context_gate.weight[0, 3] = 4.
            h = torch.ones(1, 3, 2, 4)
            attention = torch.full((1, 1, 3, 3), 1 / 3)
            result = layer(h, attention, visible, context)
            self.assertGreater(result[0, 1, 0, 0].item(), result[0, 0, 0, 0].item())
            self.assertTrue(torch.equal(result[:, :, 1], torch.zeros_like(result[:, :, 1])))
            result.sum().backward()
            self.assertTrue(torch.isfinite(layer.context_gate.weight.grad).all())

    def test_channel_spatial_residual_can_gate_flow_independently(self):
        with patch.dict(m.__dict__, WIDTH=4, GRAPH_DIM=8, CHEB_K=2):
            layer = m.ChannelSpatialReadout()
            layer.eval()
            h = torch.randn(1, 3, 2, 4)
            original = h.clone()
            visible = torch.ones(1, 2, 3, 3, dtype=torch.bool)
            visible[:, 1] = False
            node_embedding = torch.zeros(3, 4)
            context = torch.zeros(1, 3, 2, 10)
            with torch.no_grad():
                layer.correction.bias.fill_(1.)
                before = layer(h, visible, node_embedding, context)
                layer.gate.bias[0] = 3.
                after = layer(h, visible, node_embedding, context)
            self.assertTrue(torch.equal(h, original))
            self.assertTrue(torch.equal(before[..., 1], after[..., 1]))
            self.assertTrue((after[:, :, 0, 0] > before[:, :, 0, 0]).all())
            self.assertTrue(torch.equal(after[:, :, 1], torch.zeros_like(after[:, :, 1])))

    def test_context_and_channel_cli_restore_resume_and_retrieval(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release_fixture(root / 'release')
            common = ['--release-root', str(root / 'release'), '--cache-dir', str(root / 'cache'),
                      '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1']
            for mode in ('context', 'channel'):
                report = root / mode
                experiment = f'temporal_direct_dn003_{mode}_graph'
                m.main(['smoke', *common, '--output-dir', str(report), '--experiment', experiment,
                        '--epochs', '2', '--width', '8', '--heads', '2', '--blocks', '1'])
                checkpoint = report / 'full/best_maginet.pt'
                saved = torch.load(checkpoint, weights_only=True)
                history = pd.read_csv(report / 'full/history.csv')
                self.assertEqual(saved['settings']['improvements']['spatial_mode'], mode)
                self.assertEqual(saved['settings']['checkpoint_metric'], 'State')
                self.assertAlmostEqual(saved['score'], history.early_State.max())
                state_best = torch.load(report / 'full/best_state_maginet.pt', weights_only=True)
                for name, weight in saved['model'].items():
                    self.assertTrue(torch.equal(weight, state_best['model'][name]))
                self.assertEqual(json.loads((report / 'selected_alpha.json').read_text())['criterion'], {'State': 1.})
                model, contexts, _ = m.load_local_checkpoint(checkpoint, root / 'release')
                self.assertEqual(model.spatial_mode, mode)
                self.assertEqual(model.channel_spatial is not None, mode == 'channel')
                self.assertEqual(model.MASTdec[0].learned_spatial is None, mode == 'channel')
                day_entries = m.evaluation_inputs(contexts, 'blend')
                day = m.load_day(day_entries[0]['path'])
                tensors = m.tensor_context(contexts['P_N'])
                batch = m.window_batch(day, [0])
                with torch.inference_mode():
                    expected = model(**batch, ct=tensors)
                    batch['values'][~batch['visible']] = 1e6
                    self.assertTrue(torch.equal(expected, model(**batch, ct=tensors)))
                inference = [*common, '--checkpoint', str(checkpoint), '--cache-mode', 'reuse']
                m.main(['evaluate', *inference, '--period', 'early', '--select-retrieval'])
                selection = report / 'retrieval/selected_retrieval.json'
                m.main(['evaluate', *inference, '--period', 'blend', '--retrieval-selection', str(selection)])
                m.main(['predict', *inference, '--split', 'both', '--retrieval-selection', str(selection)])
                paths = sorted((report / 'retrieval/submissions').glob('*.csv'))
                self.assertEqual(len(paths), 2)
                external_state_inputs(root / 'release', paths)
                m.main(['train', *common, '--output-dir', str(report), '--resume', '--epochs', '3'])
                last = torch.load(report / 'full/last_maginet.pt', weights_only=True)
                self.assertEqual(last['settings'], saved['settings'])
                digest = m.file_digest(report / 'full/last_maginet.pt')
                with self.assertRaisesRegex(ValueError, '설정'):
                    m.main(['train', *common, '--output-dir', str(report), '--resume', '--epochs', '4',
                            '--checkpoint-metric', 'proxy'])
                self.assertEqual(digest, m.file_digest(report / 'full/last_maginet.pt'))

    def test_state_checkpoint_selection_does_not_follow_physics_proxy(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release_fixture(root / 'release')
            m.main(['prepare', '--release-root', str(root / 'release'), '--cache-dir', str(root / 'cache'),
                    '--output-dir', str(root / 'run'), '--experiment', 'temporal_direct_dn003_context_graph',
                    '--days-per-regime', '1', '--updates-per-group', '1', '--width', '8', '--heads', '2',
                    '--blocks', '1', '--device', 'cpu', '--cpu-threads', '1'])
            contexts, entries = m.prepare_inputs()
            tensors = {n: m.tensor_context(c) for n, c in contexts.items()}
            metrics = [pd.DataFrame([dict(State=.9, selection_proxy=.5)]),
                       pd.DataFrame([dict(State=.8, selection_proxy=.95)])]
            with patch.object(m, 'evaluate', side_effect=[(v, None, None, 0.) for v in metrics]):
                _, history = m.train_maginet(entries, contexts, tensors, root / 'run/full', epochs=2)
            best = torch.load(root / 'run/full/best_maginet.pt', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(best['score'], .9)
            self.assertEqual(history.early_selection_proxy.tolist(), [.5, .95])
            self.assertEqual(history.early_checkpoint_score.tolist(), [.9, .8])
            alphas = pd.DataFrame([dict(alpha=0., State=.92, selection_proxy=.5),
                                   dict(alpha=1., State=.91, selection_proxy=.95)])
            with patch.object(m, 'evaluate', return_value=(alphas, None, None, 0.)):
                selected = m.select_alpha(entries, contexts, None, root / 'run/full/best_maginet.pt')
            self.assertEqual(selected, 0.)

    def test_historical_retrieval_uses_only_visible_values_and_aligns_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '2030-12-01.npz'
            pattern = np.repeat(np.arange(16, dtype=np.float32)[:, None, None], 2, axis=2)
            np.savez(path, truth=pattern, active=np.ones((16, 1), bool), aux=np.zeros((16, 1, 21)))
            memory = HistoricalRetrieval([path])
            values = np.zeros((16, 1, 3), np.float32)
            values[:14, :, :2] = pattern[2:]
            visible = np.zeros_like(values, dtype=bool)
            visible[:14, :, :2] = True
            visible[6] = False
            day = dict(values=values, visible=visible, aux=np.zeros((16, 1, 21)),
                       timestamps=np.array(['2031-01-01'] * 16))
            network = np.ones((16, 1, 2), np.float32)
            candidates = memory.candidates(day, network)
            self.assertAlmostEqual(candidates['local'][6, 0, 0], pattern[8, 0, 0])
            self.assertNotAlmostEqual(candidates['global'][6, 0, 0], pattern[8, 0, 0])
            day['values'][~visible] = 1e6
            day['truth'], day['active'] = np.full_like(network, -999.), np.zeros((16, 1), bool)
            for method, value in memory.candidates(day, network).items():
                np.testing.assert_array_equal(value, candidates[method])
                self.assertTrue(np.isfinite(value).all())
                self.assertTrue((value >= 0).all())
            day['visible'][:] = False
            np.testing.assert_array_equal(memory.apply(day, network, 'local_gated', [1., 1.]), network)
            day['timestamps'][:] = '2030-12-01'
            with self.assertRaisesRegex(ValueError, 'fit'):
                memory.candidates(day, network)

    def test_retrieval_cli_selects_early_then_evaluates_and_predicts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release_fixture(root / 'release')
            common = ['--release-root', str(root / 'release'), '--cache-dir', str(root / 'cache'),
                      '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1']
            report = root / 'run'
            m.main(['smoke', *common, '--output-dir', str(report), '--epochs', '1', '--width', '8',
                    '--heads', '2', '--blocks', '1', '--experiment', 'temporal_direct_dn003_learned_graph'])
            checkpoint = report / 'full/best_maginet.pt'
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved['settings']['loss_weights']['dN'], .03)
            self.assertTrue(saved['settings']['improvements']['learned_graph'])
            protected = {p: m.file_digest(p) for p in report.glob('full/*') if p.is_file()}
            stamps = {p: p.stat().st_mtime_ns for p in (root / 'cache').rglob('*.npz')}
            inference = [*common, '--checkpoint', str(checkpoint), '--cache-mode', 'reuse']
            with patch('builtins.print') as printed:
                m.main(['evaluate', *inference, '--period', 'early', '--select-retrieval'])
            configuration = next(call.args[0] for call in printed.call_args_list
                                 if call.args and isinstance(call.args[0], dict) and 'reports' in call.args[0])
            self.assertEqual(configuration['reports'], str(report / 'retrieval'))
            selection = report / 'retrieval/selected_retrieval.json'
            settings = json.loads(selection.read_text())
            self.assertEqual(settings['selected_on'], 'early')
            self.assertIn(settings['method'], METHODS)
            self.assertEqual(len(settings['weights']), 2)
            early_weights = selection.read_bytes()
            m.main(['evaluate', *inference, '--period', 'blend', '--retrieval-selection', str(selection)])
            self.assertEqual(selection.read_bytes(), early_weights)
            self.assertTrue((report / 'retrieval/blend_macro.csv').is_file())
            m.main(['predict', *inference, '--retrieval-selection', str(selection), '--split', 'both'])
            paths = sorted((report / 'retrieval/submissions').glob('*.csv'))
            self.assertEqual(len(paths), 2)
            external_state_inputs(root / 'release', paths)
            self.assertEqual(protected, {p: m.file_digest(p) for p in protected})
            self.assertEqual(stamps, {p: p.stat().st_mtime_ns for p in stamps})
            settings['panels']['P_N']['memory_sha256'] = 'changed'
            selection.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, 'fit'):
                m.main(['predict', *inference, '--retrieval-selection', str(selection)])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                m.main(['evaluate', *inference, '--period', 'blend', '--select-retrieval'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA resume check requires a GPU')
    def test_cuda_fp32_resume_matches_continuous_training(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release = root/'release'
            release_fixture(release)
            common = ['--release-root', str(release), '--cache-dir', str(root/'cache'),
                      '--device', 'cuda', '--precision', 'fp32', '--cpu-threads', '1',
                      '--width', '8', '--heads', '2', '--blocks', '1',
                      '--experiment', 'temporal_direct_dnzero']
            continuous, resumed = root/'continuous', root/'resumed'
            m.main(['smoke', *common, '--output-dir', str(continuous), '--epochs', '3'])
            m.main(['smoke', *common, '--output-dir', str(resumed), '--epochs', '1'])
            first = torch.load(resumed/'full/last_maginet.pt', weights_only=True)
            self.assertEqual(len(first['rng_cuda']), torch.cuda.device_count())
            m.main(['train', *common, '--output-dir', str(resumed), '--resume', '--epochs', '3'])
            a = torch.load(continuous/'full/last_maginet.pt', weights_only=True)
            b = torch.load(resumed/'full/last_maginet.pt', weights_only=True)
            for key, value in a['model'].items():
                torch.testing.assert_close(value, b['model'][key], rtol=1e-5, atol=1e-6)
            for left, right in zip(a['rng_cuda'], b['rng_cuda']):
                self.assertTrue(torch.equal(left, right))

    def test_source_changes_rebuild_cache_and_reuse_rejects_stale_data(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release, cache = root / 'release', root / 'cache'
            release_fixture(release)
            m.main(['prepare', '--release-root', str(release), '--cache-dir', str(cache),
                    '--output-dir', str(root/'run'), '--days-per-regime', '1', '--device', 'cpu'])
            contexts, entries = m.prepare_inputs()
            norm = m.fit_input_normalization(entries, m.CACHE_METADATA_DIR/'normalization')
            signature, _ = m.run_signature(contexts, entries, norm)
            stamps = {e['path']: Path(e['path']).stat().st_mtime_ns for e in entries}
            # A legacy cache is verified once; identical arrays keep their bytes/mtime/signature.
            for marker in cache.rglob('*.sources.json'):
                marker.unlink()
            m.CACHE_MODE = 'reuse'
            with self.assertRaisesRegex(ValueError, 'build'):
                m.prepare_inputs()
            m.CACHE_MODE = 'build'
            contexts, entries = m.prepare_inputs()
            self.assertEqual(signature, m.run_signature(contexts, entries, norm)[0])
            self.assertEqual(stamps, {e['path']: Path(e['path']).stat().st_mtime_ns for e in entries})
            context = contexts['P_N']
            entry = next(e for e in entries if e['period']=='fit')
            old_day = m.load_day(entry['path'])
            ti, si = int(old_day['ti'][0]), int(old_day['si'][0])
            raw_path = context['files'][entry['date']]
            frame = pd.read_parquet(raw_path)
            selected = ((frame.timestamp == str(old_day['timestamps'][ti]))
                        & (frame.station_id == context['targets'].stations[si]))
            before = raw_path.stat()
            frame.loc[selected, 'speed_kmh'] += 25.
            frame.to_parquet(raw_path, index=False)
            os.utime(raw_path, ns=(before.st_atime_ns, before.st_mtime_ns))
            m.CACHE_MODE = 'reuse'
            with self.assertRaisesRegex(ValueError, 'build'):
                m.prepare_inputs()
            m.CACHE_MODE = 'build'
            contexts, changed = m.prepare_inputs()
            updated_entry = next(e for e in changed if e['date']==entry['date'] and e['period']=='fit')
            fresh = m.load_day(updated_entry['path'])
            self.assertAlmostEqual(float(fresh['truth'][ti,si,0]-old_day['truth'][ti,si,0]), 1., places=5)
            self.assertFalse(np.array_equal(context['profile'], contexts['P_N']['profile']))
            self.assertNotEqual(signature, m.run_signature(contexts, changed, norm)[0])
            # A masked observation changes only its day's cache, including after an LRU read.
            early = next(e for e in changed if e['period']=='early')
            cached = m.load_day(early['path'])
            value = float(cached['values'][0,1,0])
            path = contexts['P_N']['masked'][early['date']]
            frame = pd.read_parquet(path)
            frame.loc[(frame.timestamp==cached['timestamps'][0]) & (frame.station_id=='002'), 'speed_kmh'] += 25.
            frame.to_parquet(path, index=False)
            untouched = {e['path']:Path(e['path']).stat().st_mtime_ns for e in changed if e['date']!=early['date']}
            m.CACHE_MODE = 'reuse'
            with self.assertRaisesRegex(ValueError, 'build'):
                m.cache_context(contexts['P_N'], m.CACHE_ROOT, periods=('early',))
            m.CACHE_MODE = 'build'
            m.cache_context(contexts['P_N'], m.CACHE_ROOT, periods=('early',))
            self.assertAlmostEqual(float(m.load_day(early['path'])['values'][0,1,0])-value, 1., places=5)
            self.assertTrue(all(Path(p).stat().st_mtime_ns==stamp for p,stamp in untouched.items()))
            # Newly published ramps and target indices also participate in provenance.
            current = m.day_source_signature(contexts['P_N'], early['date'])
            ramps = release/'corridors/P_N/train/ramp_states'
            ramps.mkdir()
            pd.DataFrame(dict(timestamp=[cached['timestamps'][0]], ramp_link_id=['r0'], flow_vph=[10.],
                              pct_observed=[100.], is_score_eligible=[True])).to_parquet(ramps/f"ramp_{early['date']}.parquet")
            self.assertNotEqual(current, m.day_source_signature(contexts['P_N'], early['date']))

    def test_portable_alpha_and_prediction_batch_override(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch.dict(m.__dict__):
            root = Path(directory)
            release = root/'release'
            release_fixture(release)
            common = ['--release-root', str(release), '--cache-dir', str(root/'cache'),
                      '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1']
            m.main(['smoke', *common, '--output-dir', str(root/'run'), '--epochs', '1',
                    '--width', '8', '--heads', '2', '--blocks', '1', '--batch-policy', 'adaptive'])
            portable = root/'portable'
            portable.mkdir()
            checkpoint = portable/'best.pt'
            shutil.copyfile(root/'run/full/best_maginet.pt', checkpoint)
            m.main(['evaluate', *common, '--checkpoint', str(checkpoint), '--select-alpha'])
            self.assertTrue((portable/'selected_alpha.json').is_file())
            observed = []
            predict = m.maginet_predict
            def capture(*args, **kwargs):
                observed.append(kwargs['batch_size'])
                return predict(*args, **kwargs)
            with patch.object(m, 'maginet_predict', side_effect=capture):
                m.main(['predict', *common, '--checkpoint', str(checkpoint), '--batch-size', '3', '--alpha', '1'])
            self.assertEqual(set(observed), {3})
            with patch.object(m, 'load_local_checkpoint', wraps=m.load_local_checkpoint) as load:
                m.main(['predict', *common, '--checkpoint', str(checkpoint)])
                self.assertIsNotNone(load.call_args.kwargs['saved'])
            self.assertTrue((portable/'submissions/task1-P_N-validation.csv').is_file())
            with contextlib.redirect_stderr(io.StringIO()):
                for flag, value in (('--width', '64'), ('--lr', '.01'), ('--input-normalization', 'fixed')):
                    with self.subTest(flag=flag), self.assertRaises(SystemExit):
                        m.main(['predict', *common, '--checkpoint', str(checkpoint), flag, value])

    def test_independent_variants_panel_batches_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            release, cache = root / 'release', root / 'cache'
            release_fixture(release)
            common = ['--release-root', str(release), '--cache-dir', str(cache),
                      '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1',
                      '--width', '8', '--heads', '2', '--blocks', '1']
            with patch.dict(m.__dict__):
                report = root / 'dn003'
                m.main(['smoke', *common, '--output-dir', str(report), '--epochs', '6',
                        '--experiment', 'temporal_direct_dn003', '--batch-policy', 'adaptive'])
                last = torch.load(report / 'full/last_maginet.pt', weights_only=True)
                settings = last['settings']
                self.assertEqual(settings['loss_weights'], dict(N=.05, dN=.03, FD=0.))
                self.assertTrue(settings['improvements']['temporal_direct'])
                self.assertFalse(settings['improvements']['learned_graph'])
                self.assertEqual(settings['panel_batches'], {'P_N': 8})
                self.assertEqual(last['history'][-1]['train_windows'], 24)
                self.assertGreater(last['history'][-1]['loss_dN'], 0)
                m.main(['train', *common, '--output-dir', str(report), '--resume', '--epochs', '7'])
                resumed = torch.load(report / 'full/last_maginet.pt', weights_only=True)
                self.assertEqual(resumed['settings'], settings)
                self.assertEqual(resumed['history'][-1]['train_windows'], 24)
                resume_hash = m.file_digest(report / 'full/last_maginet.pt')
                with self.assertRaisesRegex(ValueError, '설정'):
                    m.main(['train', *common, '--output-dir', str(report), '--resume', '--epochs', '8',
                            '--panel-batch', 'P_N=4'])
                self.assertEqual(m.file_digest(report / 'full/last_maginet.pt'), resume_hash)

                graph_report = root / 'graph'
                m.main(['smoke', *common, '--output-dir', str(graph_report), '--epochs', '2',
                        '--experiment', 'temporal_direct_learned_graph', '--batch-policy', 'adaptive',
                        '--panel-batch', 'P_N=3'])
                graph = torch.load(graph_report / 'full/last_maginet.pt', weights_only=True)
                self.assertEqual(graph['settings']['panel_batches'], {'P_N': 3})
                self.assertEqual(graph['history'][-1]['train_windows'], 9)
                model, contexts, _ = m.load_local_checkpoint(graph_report / 'full/best_maginet.pt', release)
                self.assertTrue(model.MASTdec[0].temporal_direct)
                self.assertIsNotNone(model.MASTdec[0].learned_spatial)
                self.assertFalse(torch.equal(graph['model']['MASTdec.0.learned_spatial.gate'], torch.tensor(-3.)))
                self.assertEqual(m.panel_batch_size(contexts['P_N']), 3)
                m.main(['predict', '--release-root', str(release), '--checkpoint', str(graph_report / 'full/best_maginet.pt'),
                        '--output-dir', str(root / 'predict'), '--device', 'cpu', '--alpha', '1', '--split', 'both'])
                self.assertEqual(len(list((root / 'predict/submissions').glob('*.csv'))), 2)

    def test_batch_overrides_validate_and_legacy_settings_restore_defaults(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            release = Path(directory) / 'release'
            release_fixture(release)
            common = ['--release-root', str(release), '--device', 'cpu', '--output-dir', str(Path(directory)/'report')]
            with patch.dict(m.__dict__):
                for override in ('P_N=0', 'P_N=-1', 'P_N=1.5', '=2', 'P_N', 'unknown=2'):
                    with self.subTest(override=override), self.assertRaises(ValueError):
                        m.main(['prepare', *common, '--panel-batch', override])
                with self.assertRaises(ValueError):
                    m.main(['prepare', *common, '--panel-batch', 'P_N=2', '--panel-batch', 'P_N=3'])
                with self.assertRaisesRegex(ValueError, 'adaptive'):
                    m.main(['prepare', *common, '--batch-policy', 'adaptive', '--batch-size', '4'])
                m.BATCH_POLICY, m.PANEL_BATCH = 'adaptive', {'P_N': 9}
                m.restore_settings(dict(architecture='maginet-task13-residual-v2', batch=2))
                self.assertEqual(m.BATCH_POLICY, 'fixed')
                self.assertEqual(m.PANEL_BATCH, {})
                for stations, expected in ((128,8), (129,4)):
                    m.BATCH_POLICY = 'adaptive'
                    context = dict(panel=m.SimpleNamespace(name='P_N'), mapping=np.zeros(stations))
                    self.assertEqual(m.panel_batch_size(context), expected)

    def test_notebook_self_checks_run_against_local_functions(self):
        nb_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'task13_maginet_colab.ipynb'
        nb = json.loads(nb_path.read_text(encoding='utf-8'))
        function = ast.parse(''.join(nb['cells'][17]['source'])).body[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = dict(CACHE_METADATA_DIR=root / 'metadata', CACHE_ROOT=root / 'cache',
                          CACHE_CONTAINER=root / 'cache', CACHE_PROFILE_DIR=root / 'legacy',
                          REPORT_BASE=root / 'reports', DEVICE=torch.device('cpu'), USE_AMP=False,
                          DAYS_PER_REGIME=None, DATE_OVERRIDES={})
            # Upstream notebook cache checks have no raw files; provenance is covered above.
            with (patch.dict(m.__dict__, values), contextlib.redirect_stdout(io.StringIO()),
                  patch.object(m, 'profile_source_signature', return_value='notebook'),
                  patch.object(m, 'day_source_signature', return_value='notebook'),
                  patch.object(m, 'cache_is_current', side_effect=lambda path, source: path.is_file())):
                torch.set_num_threads(2)
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(nb_path), 'exec'), m.__dict__)
                m.run_self_checks()

    def test_local_cli_resume_portable_inference_and_colab_import(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            release, cache, report = root / 'release', root / 'cache', root / 'report'
            release_fixture(release)
            common = ['--release-root', str(release), '--cache-dir', str(cache), '--output-dir', str(report),
                      '--device', 'cpu', '--precision', 'fp32', '--cpu-threads', '1']
            with patch.dict(m.__dict__):
                m.main(['smoke', *common, '--epochs', '1', '--width', '8', '--heads', '2', '--blocks', '1'])
                best = report / 'full' / 'best_maginet.pt'
                self.assertTrue(best.is_file())
                cached = list((cache / 'inputs').rglob('*.npz'))
                self.assertEqual(len(cached), 9)
                stamps = {str(p): p.stat().st_mtime_ns for p in cached}
                with self.assertRaises(FileExistsError):
                    m.main(['train', *common])
                m.main(['train', *common, '--resume', '--epochs', '2'])
                self.assertEqual(pd.read_csv(report / 'full' / 'history.csv').epoch.tolist(), [1, 2])
                self.assertEqual(stamps, {str(p): p.stat().st_mtime_ns for p in cached})
                moved_release = root / 'moved_release'
                shutil.copytree(release, moved_release)
                moved = root / 'moved_run'
                shutil.copytree(report, moved)
                moved_cache = root / 'moved_cache'
                shutil.copytree(cache, moved_cache)
                m.main(['train', *common, '--resume', '--epochs', '3'])
                m.main(['train', '--resume', '--epochs', '3', '--release-root', str(moved_release),
                        '--cache-dir', str(moved_cache), '--output-dir', str(moved), '--device', 'cpu',
                        '--cpu-threads', '1'])
                original_last = torch.load(report / 'full/last_maginet.pt', weights_only=True)
                relocated_last = torch.load(moved / 'full/last_maginet.pt', weights_only=True)
                for key, value in original_last['model'].items():
                    self.assertTrue(torch.equal(value, relocated_last['model'][key]))
                prediction = root / 'prediction'
                moved_best = moved / 'full' / 'best_maginet.pt'
                with (patch.object(m, 'fit_input_normalization', side_effect=AssertionError('Fit cache was read')),
                      patch.object(m.PanelInfo, 'load', side_effect=AssertionError('Train layout was read'))):
                    m.main(['predict', '--checkpoint', str(moved_best), '--release-root', str(moved_release),
                            '--cache-dir', str(root / 'empty_cache'), '--cache-mode', 'reuse',
                            '--output-dir', str(prediction), '--split', 'both', '--device', 'cpu'])
                    m.main(['evaluate', '--checkpoint', str(moved_best), '--release-root', str(moved_release),
                            '--cache-dir', str(root / 'evaluation_cache'), '--output-dir', str(root / 'evaluation'),
                            '--period', 'blend', '--device', 'cpu'])
                paths = sorted((prediction / 'submissions').glob('*.csv'))
                result = external_state_inputs(moved_release, paths)
                for split, frame in result.items():
                    expected = pd.read_csv(moved_release / 'task1/P_N' / split / 'sample_submission_state.csv', dtype=str)
                    self.assertTrue(frame[m.KEYS].equals(expected[m.KEYS]))
                    self.assertTrue(np.isfinite(frame[['speed_kmh', 'flow_vph']]).all().all())
                missing = pd.read_csv(paths[0], dtype={k: str for k in m.KEYS}).iloc[:-1]
                missing.to_csv(paths[0], index=False)
                with self.assertRaises(ValueError):
                    external_state_inputs(moved_release, paths)

                # Produce a legacy Colab bundle from the same model and verify its original files survive import.
                model, contexts, saved = m.load_local_checkpoint(best, release)
                entries = []
                for period in ('fit', 'early', 'blend'):
                    entries.extend(m.evaluation_inputs(contexts, period))
                _, legacy_settings = m.notebook_signature(contexts, entries, saved['normalization'])
                legacy_settings['release'] = '/content/drive/MyDrive/kaggle_public'
                legacy = root / 'colab' / 'full'
                legacy.mkdir(parents=True)
                (legacy / 'config.json').write_text(json.dumps(legacy_settings), encoding='utf-8')
                for name, context in contexts.items():
                    np.save(legacy / f'fit_profile-{name}.npy', context['profile'])
                signature = m.hashlib.sha256(json.dumps(legacy_settings, sort_keys=True).encode()).hexdigest()
                old = {k: saved[k] for k in ('model', 'epoch', 'score')}
                old['signature'] = signature
                torch.save(old, legacy / 'best_maginet.pt')
                before = (legacy / 'best_maginet.pt').read_bytes()
                imported = root / 'imported_run'
                m.main(['import-colab', '--source-run', str(legacy.parent), '--release-root', str(release),
                        '--cache-dir', str(cache), '--output-dir', str(imported), '--device', 'cpu'])
                self.assertEqual(before, (legacy / 'best_maginet.pt').read_bytes())
                restored, _, _ = m.load_local_checkpoint(imported / 'full' / 'best_maginet.pt', moved_release)
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_windows_zip_paths_and_invalid_external_states_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in ('C:/outside.npz', 'panel/file:stream', '../outside.npz'):
                archive = root / 'unsafe.zip'
                with m.zipfile.ZipFile(archive, 'w') as stream:
                    stream.writestr(member, b'bad')
                with self.assertRaises(ValueError):
                    m.restore_input_cache(['P_N'], archive, root / 'restored')
            release = root / 'release'
            release_fixture(release)
            original = pd.concat([pd.read_csv(release / 'task1/P_N' / split / 'sample_submission_state.csv', dtype=str)
                                  for split in ('validation', 'private')], ignore_index=True)
            original['speed_kmh'], original['flow_vph'] = 75., 1200.
            path = root / 'states.csv'
            for case in ('duplicate', 'extra', 'negative', 'nan'):
                frame = original.copy()
                if case == 'duplicate':
                    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
                elif case == 'extra':
                    extra = frame.iloc[:1].copy()
                    extra['station_id'] = '999'
                    frame = pd.concat([frame, extra], ignore_index=True)
                else:
                    frame.loc[0, 'speed_kmh'] = -1 if case == 'negative' else np.nan
                frame.to_csv(path, index=False)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    external_state_inputs(release, [path])


if __name__ == '__main__':
    unittest.main()
