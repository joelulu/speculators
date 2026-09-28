"""Checks for H-Spec target-context isolation and first-block training alignment."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.models.hspec import HSpecReference, HSpecReferenceConfig, TargetContext
from speculators.models.hspec.reference import TargetKVAttention
from speculators.models.hspec.training import ce_tv_loss, training_block


@pytest.fixture
def config():
    return HSpecReferenceConfig(
        hidden_size=16,
        vocab_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        target_hidden_layers=(1, 2, 3, 4, 5),
        target_kv_layers=(3, 4, 5),
        mamba_heads=2,
        mamba_head_dim=8,
        mamba_groups=1,
        sliding_window=4,
        markov_rank=4,
    )


def test_target_context_views_exclude_future_kv_and_align_labels(config):
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    keys = [torch.randn(1, 1, 8, 8) for _ in range(5)]
    values = [torch.randn(1, 1, 8, 8) for _ in range(5)]
    output = SimpleNamespace(
        hidden_states=tuple(torch.randn(1, 8, 16) for _ in range(6)),
        past_key_values=tuple(zip(keys, values, strict=True)),
        logits=torch.randn(1, 8, 32),
    )
    block, context, previous, teacher = training_block(
        tokens, output, 4, 31, config, draft_tokens=3
    )
    assert block.tolist() == [[4, 31, 31, 31]]
    assert previous.tolist() == [[4, 4, 5, 6]]
    assert torch.equal(teacher, output.logits[:, 3:6])
    assert context.prefix_kv[0][0].data_ptr() == keys[2].data_ptr()
    assert context.prefix_kv[0][0].shape[-2] == 4
    assert torch.equal(context.last_hidden[0, 0], output.hidden_states[1][0, 3])


def test_real_qwen3_cache_matches_layer_mapping(config):
    target_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    target = Qwen3ForCausalLM(target_config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        output = target(tokens, use_cache=True, output_hidden_states=True)
    block, context, previous, teacher = training_block(
        tokens, output, 4, 31, config, draft_tokens=3
    )
    key = output.past_key_values.layers[2].keys
    assert context.prefix_kv[0][0].data_ptr() == key.data_ptr()
    assert context.prefix_kv[0][0].shape == (1, 1, 4, 8)
    assert torch.equal(context.last_hidden[0, 0], output.hidden_states[1][0, 3])
    assert torch.equal(teacher, output.logits[:, 3:6])
    assert block.shape == previous.shape == (1, 4)
    drafter = HSpecReference(
        config,
        target.get_input_embeddings().weight,
        target.get_output_embeddings().weight,
    )
    loss = ce_tv_loss(drafter(block, context, previous), teacher, tokens[:, 4:7])
    loss.backward()
    assert torch.isfinite(loss)
    assert drafter.state_proj.weight.grad is not None
    assert target.model.layers[2].self_attn.k_proj.weight.grad is None


def test_reference_model_receives_gradients_without_target_kv_gradients(config):
    torch.manual_seed(3)
    model = HSpecReference(config)
    prefix = tuple((torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)) for _ in range(3))
    context = TargetContext(torch.randn(1, 5, 16), prefix)
    logits = model(torch.tensor([[2, 31, 31]]), context, torch.tensor([[2, 2, 3]]))
    loss = ce_tv_loss(logits, torch.randn(1, 2, 32), torch.tensor([[3, 4]]))
    loss.backward()
    assert logits.shape == (1, 3, 32)
    assert model.state_proj.weight.grad is not None
    assert model.layers[0].attn.q_proj.weight.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert all(key.grad is None for key, _ in prefix)
    ids, proposals = model.draft_block(torch.tensor([[2, 31, 31]]), context)
    assert ids.shape == (1, 2)
    assert proposals.shape == (1, 2, 32)
    assert torch.equal(ids, proposals.argmax(-1))


def test_attention_cannot_see_future_block_positions(config):
    torch.manual_seed(4)
    attention = TargetKVAttention(config).eval()
    prefix = (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8))
    block = torch.randn(1, 3, 16)
    changed = block.clone()
    changed[:, 2] += 100
    with torch.no_grad():
        first = attention(block, prefix)
        second = attention(changed, prefix)
    torch.testing.assert_close(first[:, :2], second[:, :2])
    assert not torch.allclose(first[:, 2], second[:, 2])


def test_kv_layout_rejected(config):
    model = HSpecReference(config)
    wrong = tuple((torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8)) for _ in range(3))
    with pytest.raises(ValueError, match="target KV shape"):
        model(torch.tensor([[1, 31]]), TargetContext(torch.zeros(1, 5, 16), wrong))
