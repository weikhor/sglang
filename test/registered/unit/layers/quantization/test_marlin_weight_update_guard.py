"""Dense layers repacked for Marlin must refuse online weight updates."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.gpu.quantization import awq_kernels
from sglang.srt.layers.quantization import marlin_utils_fp8
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

K, N = 128, 64


def _fake_repack(b_q_weight, *args, size_k, size_n, num_bits, **kwargs):
    return torch.zeros(size_k // 16, size_n * (num_bits // 2), dtype=torch.int32)


def _fake_permute_scales(s, size_k, size_n, group_size):
    return s


class TestMarlinWeightUpdateGuard(unittest.TestCase):
    def test_fp8_marlin_layer_refuses_updates(self):
        layer = torch.nn.Module()
        layer.output_size_per_partition = N
        layer.input_size_per_partition = K
        layer.orig_dtype = torch.bfloat16
        layer.weight = torch.nn.Parameter(
            torch.zeros(K, N, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        model = torch.nn.Sequential(layer)
        self.assertIsNone(_unsupported_derived_weight_cache_error(model))

        with (
            patch.object(
                marlin_utils_fp8, "gptq_marlin_repack", _fake_repack, create=True
            ),
            patch.object(
                marlin_utils_fp8, "marlin_permute_scales", _fake_permute_scales
            ),
            patch.object(
                marlin_utils_fp8,
                "marlin_make_workspace",
                lambda device: torch.zeros(1),
            ),
        ):
            marlin_utils_fp8.prepare_fp8_layer_for_marlin(layer)

        self.assertFalse(hasattr(layer, "weight_scale_inv"))
        self.assertIn("Marlin", _unsupported_derived_weight_cache_error(model))

    def test_awq_marlin_layer_refuses_updates(self):
        group_size, num_bits = 32, 4
        layer = torch.nn.Module()
        layer.input_size_per_partition = K
        layer.output_size_per_partition = N
        layer.num_groups = K // group_size
        layer.qweight = torch.nn.Parameter(
            torch.zeros(K, N // 8, dtype=torch.int32), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros(K // group_size, N // 8, dtype=torch.int32),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones(K // group_size, N, dtype=torch.float16), requires_grad=False
        )
        model = torch.nn.Sequential(layer)
        quant_config = SimpleNamespace(
            quant_type=SimpleNamespace(size_bits=num_bits), group_size=group_size
        )
        kernel = awq_kernels.AWQMarlinLinearKernel(quant_config)

        with (
            patch.object(awq_kernels, "awq_marlin_repack", _fake_repack),
            patch.object(awq_kernels, "marlin_permute_scales", _fake_permute_scales),
            patch.object(awq_kernels, "awq_to_marlin_zero_points", lambda q, **kw: q),
            patch.object(
                awq_kernels, "marlin_make_workspace", lambda device: torch.zeros(1)
            ),
        ):
            kernel.process_weights_after_loading(layer)

        self.assertIn("Marlin", _unsupported_derived_weight_cache_error(model))


if __name__ == "__main__":
    unittest.main()
