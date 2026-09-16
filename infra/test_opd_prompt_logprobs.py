"""CPU contracts for exact selected-ID prompt scoring in the Marin vLLM fork."""

from types import SimpleNamespace

import pytest
import torch

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.sample.sampler import Sampler


def test_selected_prompt_scores_match_independent_full_vocabulary_reference():
    logits = torch.tensor([[0.0, 3.0, 1.0, -2.0], [2.0, -1.0, 0.5, 4.0]])
    prompt_ids = torch.tensor([2, 3], dtype=torch.long)
    candidate_ids = torch.tensor([[0, 3], [1, 2]], dtype=torch.long)

    result = Sampler.gather_prompt_logprobs_for_token_ids(
        logits.log_softmax(dim=-1), prompt_ids, candidate_ids
    )
    expected_ids = torch.tensor([[2, 0, 3], [3, 1, 2]], dtype=torch.int32)
    expected_scores = torch.stack(
        [logits[row].log_softmax(dim=-1)[ids.long()] for row, ids in enumerate(expected_ids)]
    )

    assert torch.equal(result.logprob_token_ids, expected_ids)
    torch.testing.assert_close(result.logprobs, expected_scores, rtol=0, atol=0)
    assert result.selected_token_ranks.tolist() == [2, 1]


def test_selected_prompt_ids_reject_out_of_vocabulary_request():
    params = SamplingParams(prompt_logprobs=2, prompt_logprob_token_ids=[[1, 10]])
    model = SimpleNamespace(max_logprobs=20, logits_processors=None, get_vocab_size=lambda: 10)

    with pytest.raises(VLLMValidationError, match="out-of-vocabulary"):
        params.verify(model, speculative_config=None, structured_outputs_config=None, tokenizer=None)
