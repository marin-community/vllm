"""Compare selected-ID prompt scores with full-vocabulary scores on one GPU."""

from __future__ import annotations

import json
import math

from vllm import LLM, SamplingParams, TokensPrompt


def main() -> None:
    model = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=64,
        max_logprobs=-1,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    prompt_ids = model.get_tokenizer().encode("Answer briefly: 2 + 2 =", add_special_tokens=False)
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    candidate_ids = [17, 29]
    selected = model.generate(
        [prompt],
        SamplingParams(
            max_tokens=1,
            prompt_logprobs=len(candidate_ids),
            prompt_logprob_token_ids=[candidate_ids[:] for _ in prompt_ids],
        ),
        use_tqdm=False,
    )[0].prompt_logprobs
    reference = model.generate(
        [prompt], SamplingParams(max_tokens=1, prompt_logprobs=-1), use_tqdm=False
    )[0].prompt_logprobs

    assert selected is not None and reference is not None
    assert len(selected) == len(reference) == len(prompt_ids)
    differences = []
    for selected_row, reference_row in zip(selected[1:], reference[1:], strict=True):
        assert selected_row is not None and reference_row is not None
        for token_id in candidate_ids:
            assert token_id in selected_row and token_id in reference_row
            differences.append(abs(selected_row[token_id].logprob - reference_row[token_id].logprob))
    max_difference = max(differences)
    assert math.isfinite(max_difference) and max_difference <= 1e-5
    print(json.dumps({"positions": len(prompt_ids) - 1, "candidates": len(candidate_ids), "max_abs_diff": max_difference}))


if __name__ == "__main__":
    main()
