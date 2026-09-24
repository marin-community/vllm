# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.grug_reasoning_parser import GrugReasoningParser

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"


class GrugTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {START_THINK: 128002, END_THINK: 128003}


@pytest.fixture
def parser() -> GrugReasoningParser:
    return GrugReasoningParser(GrugTokenizer())


def test_grug_reasoning_parser_splits_final_answer(parser: GrugReasoningParser):
    request = ChatCompletionRequest(model="grug", messages=[])

    reasoning, content = parser.extract_reasoning(
        f"{START_THINK}The answer is 63.{END_THINK}\\boxed{{63}}", request
    )

    assert reasoning == "The answer is 63."
    assert content == "\\boxed{63}"


def test_grug_reasoning_parser_keeps_unfinished_thinking_out_of_content(
    parser: GrugReasoningParser,
):
    request = ChatCompletionRequest(model="grug", messages=[])

    reasoning, content = parser.extract_reasoning(
        f"{START_THINK}The answer might be", request
    )

    assert reasoning == "The answer might be"
    assert content is None


def test_grug_reasoning_parser_preserves_nothink_answer(parser: GrugReasoningParser):
    request = ChatCompletionRequest(
        model="grug", messages=[], chat_template_kwargs={"enable_thinking": False}
    )

    reasoning, content = parser.extract_reasoning("\\boxed{63}", request)

    assert reasoning is None
    assert content == "\\boxed{63}"


def test_grug_reasoning_parser_splits_streaming_answer(parser: GrugReasoningParser):
    start_id, end_id = 128002, 128003
    reasoning = parser.extract_reasoning_streaming(
        "", f"{START_THINK}The answer is 63.", f"{START_THINK}The answer is 63.", [],
        [start_id, 1], [start_id, 1],
    )
    answer = parser.extract_reasoning_streaming(
        f"{START_THINK}The answer is 63.",
        f"{START_THINK}The answer is 63.{END_THINK}\\boxed{{63}}",
        f"{END_THINK}\\boxed{{63}}",
        [start_id, 1], [start_id, 1, end_id, 2], [end_id, 2],
    )

    assert reasoning.reasoning == "The answer is 63."
    assert answer.content == "\\boxed{63}"


def test_grug_reasoning_parser_streams_nothink_answer_as_content(
    parser: GrugReasoningParser,
):
    delta = parser.extract_reasoning_streaming(
        "", "\\boxed{63}", "\\boxed{63}", [], [1], [1]
    )

    assert delta.content == "\\boxed{63}"
    assert delta.reasoning is None


def test_grug_reasoning_parser_preserves_special_tokens(parser: GrugReasoningParser):
    request = ChatCompletionRequest(model="grug", messages=[])

    parser.adjust_request(request)

    assert request.skip_special_tokens is False


def test_grug_reasoning_parser_nothink_allows_content_grammar():
    parser = GrugReasoningParser(
        GrugTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )

    assert parser.is_reasoning_end([])
    assert parser.extract_content_ids([1, 2]) == [1, 2]
