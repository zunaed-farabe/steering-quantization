import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from steerquant.steering import capture, intervene


@pytest.fixture
def tiny():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    return LlamaForCausalLM(LlamaConfig(vocab_size=24, hidden_size=16, intermediate_size=24,
                                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)).eval()


def test_capture_matches_block_output_not_embedding_index(tiny):
    ids = torch.tensor([[1, 3, 5]])
    with capture(tiny.model.layers, [0]) as values, torch.inference_mode():
        result = tiny(ids, output_hidden_states=True, use_cache=False)
    assert torch.equal(values[0], result.hidden_states[1][0, -1].float())
    assert not tiny.model.layers[0]._forward_hooks


def test_intervention_changes_only_last_position_and_resets(tiny):
    ids = torch.tensor([[1, 3, 5]])
    vector = torch.arange(16).float() / 16
    with torch.inference_mode():
        baseline = tiny(ids, use_cache=False).logits.clone()
        with intervene(tiny.model.layers[0], vector, 0):
            assert torch.equal(baseline, tiny(ids, use_cache=False).logits)
        with intervene(tiny.model.layers[0], vector, 0.4):
            changed = tiny(ids, use_cache=False).logits
        assert torch.equal(changed[:, :-1], baseline[:, :-1])
        assert not torch.allclose(changed[:, -1], baseline[:, -1])
        assert torch.equal(baseline, tiny(ids, use_cache=False).logits)
    assert not tiny.model.layers[0]._forward_hooks


def test_hook_cleanup_on_exception(tiny):
    with pytest.raises(RuntimeError):
        with intervene(tiny.model.layers[0], torch.ones(16), 1):
            raise RuntimeError("interrupted experiment")
    assert not tiny.model.layers[0]._forward_hooks


def test_cached_decode_matches_full_sequence_intervention(tiny):
    ids = torch.tensor([[1, 3, 5]])
    v = torch.arange(16).float() / 16
    # For the first answer, cached prefill must include the final prompt state.
    with intervene(tiny.model.layers[0], v, 0.1), torch.inference_mode():
        cached = tiny(ids, use_cache=True)
        uncached = tiny(ids, use_cache=False)
    assert torch.allclose(cached.logits[:, -1], uncached.logits[:, -1])


def test_generation_hook_covers_prefill_and_each_cached_step(tiny):
    observed = []
    observer = tiny.model.layers[0].register_forward_hook(lambda module, inputs, output: observed.append(output.shape[1]))
    try:
        with intervene(tiny.model.layers[0], torch.arange(16).float(), 0.01), torch.inference_mode():
            tiny.generate(torch.tensor([[1, 3, 5]]), do_sample=False, max_new_tokens=3,
                          min_new_tokens=3, pad_token_id=0, use_cache=True)
    finally:
        observer.remove()
    assert observed == [3, 1, 1]
    assert not tiny.model.layers[0]._forward_hooks
