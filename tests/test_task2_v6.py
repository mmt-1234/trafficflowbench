"""Transition decoder, gated prior, and masked hazard supervision contracts."""
import types
import unittest

import torch

import test_task2_v4 as v4


class Task2V6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v4.Task2V4Tests.setUpClass()
        cls.models = v4.Task2V4Tests.model
        cls.learning = v4.Task2V4Tests.learning

    def make_model(self, **extra):
        return self.models.TrafficQueueModel(features=44, node_count=2, width=8,
            blocks=1, heads=2, dropout=0., backend='scan', graph=True,
            queue_head='transition', **extra)

    def inputs(self, steps=12):
        return (torch.randn(2, steps, 2, 44), torch.tensor([1, 2]), ((0, 2),),
                torch.tensor([[0], [1]]))

    def test_twelve_and_thirteen_step_histories_roll_out_to_six_leads_and_backpropagate(self):
        model = self.make_model()
        for steps in (12, 13):
            with self.subTest(history=steps):
                calls = []
                hook = model.decoder.register_forward_hook(lambda *args: calls.append(1))
                logits, speeds, parts = model(*self.inputs(steps), condition=torch.tensor([0, 1]), return_components=True)
                hook.remove()
                self.assertEqual(len(calls), 7 if steps == 12 else 6)
                self.assertEqual(logits.shape, (2, 6, 2))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue((speeds >= 0).all())
                self.assertEqual(parts['onset_logit'].shape, logits.shape)
                (logits.square().mean() + speeds.mean()).backward()
                self.assertGreater(model.hazards.weight.grad.abs().sum().item(), 0)

    def test_evaluation_never_uses_teacher_labels(self):
        model = self.make_model().eval()
        inputs = self.inputs()
        condition = torch.tensor([0, 1])
        first = model(*inputs, condition=condition)
        second = model(*inputs, condition=condition, teacher=torch.full((2, 6, 2), float('nan')),
                       teacher_mask=torch.ones(2, 6, 2, dtype=torch.bool), teacher_probability=1.)
        torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)

    def test_gate_has_gradients_and_prior_dropout_is_training_only(self):
        model = self.make_model(queue_future_prior=True, queue_prior_gate=True, queue_prior_dropout=.5)
        prior = torch.ones(2, 6, 2, 2, requires_grad=True)
        hidden = torch.randn(2, 2, 8)
        future = torch.randn(2, 6, 2, 7)
        model.eval()
        gated = model.apply_prior(prior, hidden, future)
        torch.testing.assert_close(gated, prior * .5)
        gated.sum().backward()
        self.assertGreater(model.prior_gate.bias.grad.abs().sum().item(), 0.)
        model.train()
        torch.manual_seed(2)
        dropped = model.apply_prior(prior, hidden, future)
        self.assertTrue(((dropped == 0) | (dropped == .5)).all())
        model.eval()
        torch.testing.assert_close(model.apply_prior(prior, hidden, future), gated)

    def test_hazard_loss_masks_missing_pairs_and_supervises_occurrence_and_clearance(self):
        truth = torch.tensor([[[0.], [1.], [0.], [1.], [0.], [0.]]])
        valid = torch.ones_like(truth, dtype=torch.bool)
        valid[:, 3] = False
        batch = dict(truth=truth, soft=truth, mask=valid, weight=torch.ones_like(truth), speed=torch.zeros_like(truth))
        args = types.SimpleNamespace(queue_iou_target='hard', soft_iou_weight=0., speed_weight=0.,
                                     queue_balanced=True, queue_transition_weight=1.)
        on = torch.zeros_like(truth, requires_grad=True)
        off = torch.zeros_like(truth, requires_grad=True)
        logits = torch.zeros_like(truth)
        loss = self.learning.queue_loss(logits, logits, batch, ['onset'], args,
                                       components=dict(onset_logit=on, clear_logit=off))
        loss.backward()
        self.assertLess(on.grad[0, 1, 0].item(), 0.)  # valid 0 -> 1
        self.assertLess(off.grad[0, 2, 0].item(), 0.)  # valid 1 -> 0
        self.assertEqual(on.grad[0, 3:5].abs().sum().item(), 0.)
        self.assertEqual(off.grad[0, 3:5].abs().sum().item(), 0.)
        self.assertEqual(on.grad[0, 0].item(), 0.)  # no label for the hidden origin gap

    def test_legacy_head_can_initialize_only_the_new_encoder_and_prior(self):
        legacy = self.models.TrafficQueueModel(features=44, node_count=2, width=8, blocks=1,
            heads=2, dropout=0., backend='scan', graph=True, queue_head='conditional')
        new = self.make_model()
        checkpoint = dict(format_version=2, task='queue', model_config=dict(queue_head='conditional'),
                          weights=legacy.state_dict())
        self.learning.initialize_queue(new, checkpoint)
        for name, value in legacy.encoder.state_dict().items():
            torch.testing.assert_close(new.encoder.state_dict()[name], value)

    def test_teacher_decay_keeps_original_budget_after_resume_extensions(self):
        original = types.SimpleNamespace(epochs=75, queue_teacher_start=.5)
        self.learning.queue_recipe(original)
        self.assertEqual(original.queue_teacher_epochs, 75)
        self.assertEqual(self.learning.queue_teacher_probability(original, 37), .25)
        saved = dict(vars(original))
        resumed = types.SimpleNamespace(epochs=150)
        self.learning.queue_recipe(resumed, dict(training_args=saved))
        self.assertEqual(resumed.queue_teacher_epochs, 75)
        for epoch in (0, 37, 74, 100):
            self.assertEqual(self.learning.queue_teacher_probability(resumed, epoch),
                             self.learning.queue_teacher_probability(original, epoch))
        self.assertEqual(self.learning.queue_teacher_probability(resumed, 74), 0.)
        again = types.SimpleNamespace(epochs=300)
        self.learning.queue_recipe(again, dict(training_args=vars(resumed)))
        self.assertEqual(again.queue_teacher_epochs, 75)
        self.assertEqual(self.learning.queue_teacher_probability(again, 150), 0.)
        # Pre-fix v6 checkpoints have only their original total epoch budget.
        saved.pop('queue_teacher_epochs')
        legacy = types.SimpleNamespace(epochs=150)
        self.learning.queue_recipe(legacy, dict(training_args=saved))
        self.assertEqual(legacy.queue_teacher_epochs, 75)
        single = types.SimpleNamespace(epochs=1, queue_teacher_start=.5)
        self.learning.queue_recipe(single)
        self.assertEqual(self.learning.queue_teacher_probability(single, 0), 0.)

    def test_resume_cannot_silently_change_v6_recipe(self):
        saved = dict(queue_prior_gate=True, queue_prior_dropout=.2,
                     queue_transition_weight=.1, queue_teacher_start=.5)
        args = types.SimpleNamespace()
        self.learning.queue_recipe(args, dict(training_args=saved))
        self.assertEqual(args.queue_prior_dropout, .2)
        with self.assertRaisesRegex(ValueError, 'Resume must retain'):
            self.learning.queue_recipe(types.SimpleNamespace(queue_prior_dropout=0.), dict(training_args=saved))


if __name__ == '__main__':
    unittest.main()
