# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, call

import pytest

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.v1.metrics.loggers import (
    CUDAGRAPH_RUNTIME_MODES,
    PrometheusStatLogger,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _logger_with_cudagraph_counters() -> PrometheusStatLogger:
    logger = PrometheusStatLogger.__new__(PrometheusStatLogger)
    logger.counter_cudagraph_iterations = {
        mode: {0: MagicMock()} for mode in CUDAGRAPH_RUNTIME_MODES
    }
    logger.counter_cudagraph_unpadded_tokens = {
        mode: {0: MagicMock()} for mode in CUDAGRAPH_RUNTIME_MODES
    }
    logger.counter_cudagraph_padded_tokens = {
        mode: {0: MagicMock()} for mode in CUDAGRAPH_RUNTIME_MODES
    }
    return logger


@pytest.mark.parametrize("runtime_mode", CUDAGRAPH_RUNTIME_MODES)
def test_cudagraph_metrics_are_cumulative_and_mode_bounded(runtime_mode: str):
    logger = _logger_with_cudagraph_counters()
    stats = CUDAGraphStat(
        num_unpadded_tokens=13,
        num_padded_tokens=16,
        num_paddings=3,
        runtime_mode=runtime_mode,
    )

    logger._record_cudagraph_metrics(stats, engine_idx=0)
    logger._record_cudagraph_metrics(stats, engine_idx=0)

    logger.counter_cudagraph_iterations[runtime_mode][0].inc.assert_has_calls(
        [call(), call()]
    )
    logger.counter_cudagraph_unpadded_tokens[runtime_mode][0].inc.assert_has_calls(
        [call(13), call(13)]
    )
    logger.counter_cudagraph_padded_tokens[runtime_mode][0].inc.assert_has_calls(
        [call(16), call(16)]
    )
    assert set(logger.counter_cudagraph_iterations) == {
        "NONE",
        "PIECEWISE",
        "FULL",
    }


def test_cudagraph_metrics_drop_unknown_mode_without_new_series():
    logger = _logger_with_cudagraph_counters()
    stats = CUDAGraphStat(
        num_unpadded_tokens=13,
        num_padded_tokens=16,
        num_paddings=3,
        runtime_mode="FULL_AND_PIECEWISE",
    )

    logger._record_cudagraph_metrics(stats, engine_idx=0)

    assert set(logger.counter_cudagraph_iterations) == set(CUDAGRAPH_RUNTIME_MODES)
    for counters_by_mode in (
        logger.counter_cudagraph_iterations,
        logger.counter_cudagraph_unpadded_tokens,
        logger.counter_cudagraph_padded_tokens,
    ):
        for counters_by_engine in counters_by_mode.values():
            counters_by_engine[0].inc.assert_not_called()
