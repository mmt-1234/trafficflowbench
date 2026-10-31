"""Exercise the notebook's actual training loop with a fixed validation score."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch

from test_task123_colab import NOTEBOOK, KAGGLE_NOTEBOOK, embedded, functions


class PhysicsScheduleTests(unittest.TestCase):
    def test_physics_ramp_resume_and_early_stopping(self):
        class Model(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.value = torch.nn.Parameter(torch.zeros(()))
            def forward(self, **kwargs):
                return self.value
        for notebook in (NOTEBOOK, KAGGLE_NOTEBOOK):
            with self.subTest(notebook=notebook.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                namespace = dict(np=np, pd=pd, torch=torch, json=json, hashlib=hashlib, time=time,
                    EPOCHS=30, RESUME=False, CACHE_METADATA_DIR=root / 'cache', DEVICE=torch.device('cpu'),
                    LR=1e-3, WEIGHT_DECAY=1e-4, USE_AMP=False, AMP_DTYPE=torch.float32,
                    SEED=42, WARMUP_EPOCHS=5, PHYSICS_RAMP_EPOCHS=5, UPDATES_PER_GROUP=1,
                    WINDOW=2, REGIMES=('R1', 'R2', 'R3'), CHECKPOINT_METRIC='State', PATIENCE=8,
                    MagiNetTask13=Model, reset_seed=lambda: torch.manual_seed(42),
                    fit_input_normalization=lambda *a, **k: {}, run_signature=lambda *a: ('fixed', {}),
                    checkpoint_metadata=lambda *a: dict(format_version=3), panel_batch_size=lambda c: 1,
                    load_day=lambda p: dict(values=np.zeros((2, 1)), target=np.ones((2, 1), bool)),
                    window_starts=lambda n: [0], window_batch=lambda *a: {},
                    joint_loss=lambda output, *a: ((output - 1).square(), dict(State=0., N=0., dN=0., FD=0.)),
                    evaluate=lambda *a: (pd.DataFrame([dict(State=.9, selection_proxy=.9)]), None, None, 0.))
                functions(embedded('maginet', notebook), ['train_maginet', 'optimizer_step', 'save_checkpoint'], namespace)
                context = {'P_N': dict(station_codes=np.array([0]), panel_code=0, family_code=0)}
                entries = [dict(panel='P_N', regime=r, period='fit', targets=1, path='synthetic', date=str(r))
                           for r in range(3)]
                def train(folder, epochs, resume=False):
                    with contextlib.redirect_stdout(io.StringIO()):
                        namespace['train_maginet'](entries, context, {'P_N': {}}, folder, epochs=epochs, resume=resume)
                    return torch.load(folder / 'last_maginet.pt', weights_only=True)
                first = train(root / 'resumed', 9)
                self.assertEqual(first['optimizer']['param_groups'][0]['lr'], 1e-3)
                self.assertEqual(first['bad_epochs'], 0)
                shutil.copytree(root / 'resumed', root / 'legacy')
                old = torch.load(root / 'legacy/last_maginet.pt', weights_only=True)
                old.pop('physics_schedule')
                old.pop('phase_best')
                old['optimizer']['param_groups'][0]['lr'] = 1.25e-4
                old['bad_epochs'] = 8
                torch.save(old, root / 'legacy/last_maginet.pt')
                migrated = train(root / 'legacy', 10, resume=True)
                self.assertEqual(migrated['optimizer']['param_groups'][0]['lr'], 1e-3)
                self.assertEqual(migrated['bad_epochs'], 0)
                resumed = train(root / 'resumed', 30, resume=True)
                full = train(root / 'full', 30)
                self.assertEqual(resumed['epoch'], 18)
                self.assertEqual(resumed['bad_epochs'], 8)
                self.assertEqual(resumed['scheduler'], full['scheduler'])
                torch.testing.assert_close(resumed['model']['value'], full['model']['value'], rtol=0, atol=0)
                rows = pd.DataFrame(resumed['history']).set_index('epoch')
                self.assertEqual(rows.loc[10, 'lr'], 1e-3)
                self.assertEqual(rows.loc[13, 'lr'], 5e-4)
                self.assertEqual(torch.load(root / 'resumed/best_maginet.pt', weights_only=True)['epoch'], 1)
                namespace.update(WARMUP_EPOCHS=0, PHYSICS_RAMP_EPOCHS=0)
                self.assertEqual(train(root / 'no-ramp', 30)['epoch'], 9)


if __name__ == '__main__':
    unittest.main()
