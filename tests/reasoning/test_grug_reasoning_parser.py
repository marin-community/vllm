# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import NamedTuple

import pytest

from vllm.entrypoints.generate.base.protocol import DeltaToolCall
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.abstract_parser import DelegatingParser
from vllm.reasoning.grug_reasoning_parser import GrugReasoningParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
START_THINK_ID = 128002
END_THINK_ID = 128003
START_HEADER_ID = 128006
END_HEADER_ID = 128007


class StreamResult(NamedTuple):
    reasoning: str
    content: str
    calls: list[DeltaToolCall]


class GrugWithTools(DelegatingParser):
    reasoning_parser_cls = GrugReasoningParser
    tool_parser_cls = Hermes2ProToolParser


class GrugTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {
            START_THINK: START_THINK_ID,
            END_THINK: END_THINK_ID,
            "<|start_header_id|>": START_HEADER_ID,
            "<|end_header_id|>": END_HEADER_ID,
        }


def stream_grug(chunks, prompt_token_ids=None, *, tools=False) -> StreamResult:
    parser = GrugWithTools(GrugTokenizer())
    request = ChatCompletionRequest(
        model="grug",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        if tools
        else None,
        tool_choice="auto" if tools else "none",
    )
    output = []
    for index, (text, token_ids) in enumerate(chunks):
        output.append(
            parser.parse_delta(
                text,
                token_ids,
                request,
                prompt_token_ids=prompt_token_ids if index == 0 else None,
                finished=index == len(chunks) - 1,
            )
        )
    reasoning = "".join(
        delta.reasoning for delta in output if delta and delta.reasoning
    )
    content = "".join(delta.content for delta in output if delta and delta.content)
    calls = [
        call
        for delta in output
        if delta and delta.tool_calls
        for call in delta.tool_calls
    ]
    return StreamResult(reasoning, content, calls)


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
    start_id, end_id = START_THINK_ID, END_THINK_ID
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


def test_grug_reasoning_parser_streams_nothink_answer_as_content():
    parser = GrugReasoningParser(
        GrugTokenizer(), chat_template_kwargs={"enable_thinking": False}
    )
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


def test_grug_streaming_ignores_previous_turn_closing_marker():
    prompt_ids = [
        START_HEADER_ID,
        10,
        END_HEADER_ID,
        START_THINK_ID,
        11,
        END_THINK_ID,
        START_HEADER_ID,
        10,
        END_HEADER_ID,
    ]
    result = stream_grug(
        [
            (f"{START_THINK}Check.", [START_THINK_ID, 1]),
            (f"{END_THINK}42", [END_THINK_ID, 2]),
        ],
        prompt_ids,
    )

    assert result.reasoning == "Check."
    assert result.content == "42"


def test_grug_streaming_holds_unmarked_reasoning_until_closing_marker():
    result = stream_grug(
        [("Check.", [1]), (f"{END_THINK}4", [END_THINK_ID, 2])], []
    )

    assert result.reasoning == "Check."
    assert result.content == "4"


def test_grug_streaming_hands_marker_free_tool_call_to_hermes():
    result = stream_grug(
        [
            ("<tool_", [1]),
            ('call>{"name":"calculator","arguments":{"x":4}}</tool_call>', [2]),
        ],
        [],
        tools=True,
    )

    assert result.reasoning == ""
    assert result.content == ""
    assert result.calls[0].function.name == "calculator"
    arguments = "".join(call.function.arguments or "" for call in result.calls)
    assert json.loads(arguments) == {"x": 4}


def test_grug_streaming_preserves_marker_free_plain_answer():
    result = stream_grug(
        [("Plain", [1]), (" answer", [2])], []
    )

    assert result.reasoning == ""
    assert result.content == "Plain answer"
    assert result.calls == []
