"""Torch checks run when model dependencies are installed."""

import importlib.util
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class ModelChecks(unittest.TestCase):
    def test_parallel_scan_matches_sequential_values_and_gradients(self):
        import torch
        from torch.nn import functional as F
        from fast_model import SpatialScan

        torch.manual_seed(3)
        scan = SpatialScan(8)
        segments = ((0, 1), (1, 4), (4, 130))

        def sequential(x, reverse):
            parts = []
            for start, stop in segments:
                nodes = x[:, :, start:stop]
                if reverse:
                    nodes = nodes.flip(2)
                batch, steps, count, width = nodes.shape
                local = scan.local(nodes.reshape(batch * steps, count, width).transpose(1, 2))
                local = local.transpose(1, 2).reshape(batch, steps, count, width)
                state = torch.zeros_like(nodes[:, :, 0])
                outputs = []
                for index in range(count):
                    delta, source, readout = scan.parameters_for_step(nodes[:, :, index]).chunk(3, -1)
                    decay = torch.exp(-F.softplus(delta) * F.softplus(scan.log_decay))
                    state = decay * state + (1 - decay) * torch.tanh(source)
                    outputs.append(state * torch.sigmoid(readout))
                part = torch.stack(outputs, dim=2) * torch.sigmoid(local)
                parts.append(part.flip(2) if reverse else part)
            return scan.project(torch.cat(parts, dim=2))

        for reverse in (False, True):
            actual_input = torch.randn(2, 5, 130, 8, requires_grad=True)
            expected_input = actual_input.detach().clone().requires_grad_()
            actual = scan(actual_input, segments, reverse)
            expected = sequential(expected_input, reverse)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
            actual_grads = torch.autograd.grad(actual.square().mean(), (actual_input, *scan.parameters()))
            expected_grads = torch.autograd.grad(expected.square().mean(), (expected_input, *scan.parameters()))
            for left, right in zip(actual_grads, expected_grads):
                self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-5))

    def test_shapes_gradients_and_component_boundary(self):
        import torch
        from fast_model import QueueModel, SpatialScan, StateModel

        torch.manual_seed(1)
        links = torch.randn(1, 12, 4, 9)
        segments = ((0, 2), (2, 4))
        scan = SpatialScan(8)
        hidden = torch.randn(1, 12, 4, 8)
        changed = hidden.clone()
        changed[:, :, :2] += 100
        for reverse in (False, True):
            self.assertTrue(torch.allclose(scan(hidden, segments, reverse)[:, :, 2:],
                                               scan(changed, segments, reverse)[:, :, 2:]))

        state = StateModel(width=8, blocks=1)
        context = torch.ones(1, 12, 3, 3)
        output = state(links, context, torch.tensor([0, 1, 1]), segments)
        self.assertEqual(output.shape, (1, 12, 3, 2))
        output.sum().backward()
        self.assertIsNotNone(state.head.weight.grad)

        queue = QueueModel(width=8, blocks=1)
        logits, speeds = queue(links, segments)
        self.assertEqual(logits.shape, (1, 6, 4))
        self.assertEqual(speeds.shape, (1, 6, 4))
        (logits.sum() + speeds.sum()).backward()


if __name__ == "__main__":
    unittest.main()
