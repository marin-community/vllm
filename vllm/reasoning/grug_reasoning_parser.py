# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import Any

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class GrugReasoningParser(BaseThinkingReasoningParser):
    """Separate Grug's special-token reasoning block from its final answer."""

    def __init__(self, tokenizer: TokenizerLike, *args: Any, **kwargs: Any):
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = chat_kwargs.get("enable_thinking") is not False
        super().__init__(tokenizer, *args, **kwargs)

    @property
    def start_token(self) -> str:
        return "<|start_think|>"

    @property
    def end_token(self) -> str:
        return "<|end_think|>"

    def _nothink_content(self, input_ids: Sequence[int]) -> bool:
        return not self._thinking_enabled and self.start_token_id not in input_ids

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._nothink_content(input_ids):
            return True
        return super().is_reasoning_end(input_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._nothink_content(input_ids):
            return input_ids
        return super().extract_content_ids(input_ids)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        delta = super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
        if (
            delta is not None
            and delta.reasoning is not None
            and self.start_token_id in delta_token_ids
        ):
            # The base parser includes a start marker when it shares a text delta.
            reasoning = delta.reasoning.replace(self.start_token, "", 1)
            return delta.model_copy(update={"reasoning": reasoning})
        return delta
