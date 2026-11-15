"""The overlap plan stream must wait for the compute stream before draft-extend prep."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopAfterPrepare(Exception):
    pass


class _FakeEvent:
    def __init__(self, log):
        self._log = log
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True


class _FakePlanStream:
    def __init__(self, log, compute_stream):
        self._log = log
        self._compute_stream = compute_stream

    def wait_stream(self, stream):
        if stream is self._compute_stream:
            self._log.append("plan_waits_compute")

    def wait_event(self, event):
        if event.recorded:
            self._log.append("plan_waits_compute")


def _make_worker(*, plan_stream, num_front_tokens):
    worker = MultiLayerEagleDraftWorker.__new__(MultiLayerEagleDraftWorker)
    worker.device = "cuda"
    worker.speculative_num_steps = 3
    worker.speculative_num_draft_tokens = 4
    worker.draft_extend_num_front_tokens = num_front_tokens
    worker.draft_runner_list = [None]
    worker.cuda_graph_runner_for_draft_extend = None
    worker.plan_stream = plan_stream
    worker.plan_stream_ctx = nullcontext()
    return worker


def _make_batch(idle):
    return SimpleNamespace(
        seq_lens=torch.tensor([5, 7]),
        forward_mode=SimpleNamespace(is_idle=lambda: idle),
    )


def _make_batch_result():
    return SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=None),
        next_token_ids=torch.tensor([1, 2]),
        accept_lens=torch.tensor([1, 2]),
    )


class TestMultiLayerEaglePlanStreamOrder(CustomTestCase):
    def _run(self, *, num_front_tokens, idle, staged, boundary):
        log = []
        compute_stream = object()
        device_module = SimpleNamespace(
            current_stream=lambda: compute_stream,
            Event=lambda: _FakeEvent(log),
        )
        worker = _make_worker(
            plan_stream=_FakePlanStream(log, compute_stream),
            num_front_tokens=num_front_tokens,
        )
        locs = (torch.zeros(4), torch.zeros(4)) if boundary else (None, None)
        worker._compute_boundary_kv_locs_positions = lambda batch: locs

        def fake_prepare(*args, **kwargs):
            log.append("prepare")
            raise _StopAfterPrepare

        with (
            patch.object(torch, "get_device_module", return_value=device_module),
            patch(
                "sglang.srt.speculative.multi_layer_eagle_worker_v2."
                "prepare_for_draft_extend",
                side_effect=fake_prepare,
            ),
        ):
            with self.assertRaises(_StopAfterPrepare):
                worker._draft_extend_for_decode(
                    _make_batch(idle), _make_batch_result(), staged=staged
                )
        return log

    def test_no_front_tokens_waits_before_prepare(self):
        for staged in (False, True):
            with self.subTest(staged=staged):
                log = self._run(
                    num_front_tokens=0, idle=False, staged=staged, boundary=False
                )
                self.assertEqual(log, ["plan_waits_compute", "prepare"])

    def test_idle_batch_waits_before_prepare(self):
        log = self._run(num_front_tokens=2, idle=True, staged=False, boundary=False)
        self.assertEqual(log, ["plan_waits_compute", "prepare"])

    def test_boundary_locs_wait_before_prepare(self):
        log = self._run(num_front_tokens=2, idle=False, staged=False, boundary=True)
        self.assertEqual(log, ["plan_waits_compute", "prepare"])


if __name__ == "__main__":
    unittest.main()
