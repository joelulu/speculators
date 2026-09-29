"""Checks for H-Spec target-context isolation and first-block training alignment."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional
from transformers import Qwen3Config, Qwen3ForCausalLM

from speculators.models.hspec import (
    HSpecReference,
    HSpecReferenceConfig,
    TargetContext,
    context_from_vllm_pages,
)
from speculators.models.hspec.reference import (
    TargetKVAttention,
    _parallel_state_scan,
)
from speculators.models.hspec.training import ce_tv_loss, training_block

_infer_path = Path(__file__).resolve().parents[3] / "examples/hspec/reference_infer.py"
_infer_spec = importlib.util.spec_from_file_location(
    "hspec_reference_infer", _infer_path
)
assert _infer_spec is not None
assert _infer_spec.loader is not None
_infer_module = importlib.util.module_from_spec(_infer_spec)
_infer_spec.loader.exec_module(_infer_module)
greedy_verify_block = _infer_module.greedy_verify_block


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


def test_bfloat16_attention_uses_target_kv_dtype(config):
    attention = TargetKVAttention(config).to(dtype=torch.bfloat16)
    block = torch.randn(1, 3, config.hidden_size, dtype=torch.bfloat16)
    prefix = (
        torch.randn(1, 1, 4, config.head_dim, dtype=torch.bfloat16),
        torch.randn(1, 1, 4, config.head_dim, dtype=torch.bfloat16),
    )
    output = attention(block, prefix)
    assert output.dtype == torch.bfloat16
    output.float().sum().backward()
    assert attention.q_proj.weight.grad is not None


def test_kv_layout_rejected(config):
    model = HSpecReference(config)
    wrong = tuple((torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8)) for _ in range(3))
    with pytest.raises(ValueError, match="target KV shape"):
        model(torch.tensor([[1, 31]]), TargetContext(torch.zeros(1, 5, 16), wrong))


@pytest.mark.parametrize("length", [1, 2, 3, 7, 8, 16])
def test_parallel_scan_matches_serial_forward_and_gradients(length):
    torch.manual_seed(length)
    decay = torch.rand(2, length, 3, dtype=torch.double) * 0.8 + 0.1
    updates = torch.randn(2, length, 3, 2, 4, dtype=torch.double)
    initial = torch.randn(2, 3, 2, 4, dtype=torch.double)
    weights = torch.randn_like(updates)

    def serial(a, u, state):
        result = []
        for index in range(length):
            state = a[:, index, :, None, None] * state + u[:, index]
            result.append(state)
        return torch.stack(result, dim=1)

    def run(scan):
        inputs = [
            tensor.clone().requires_grad_() for tensor in (decay, updates, initial)
        ]
        output = scan(*inputs)
        gradients = torch.autograd.grad((output * weights).sum(), inputs)
        return output, gradients

    parallel_output, parallel_grads = run(_parallel_state_scan)
    serial_output, serial_grads = run(serial)
    torch.testing.assert_close(parallel_output, serial_output, atol=1e-12, rtol=1e-12)
    for parallel, expected in zip(parallel_grads, serial_grads, strict=True):
        torch.testing.assert_close(parallel, expected, atol=1e-12, rtol=1e-12)


def test_ce_tv_loss_uses_dflash_position_weights():
    draft = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]]])
    teacher = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    labels = torch.tensor([[0, 1]])
    ce = functional.cross_entropy(
        draft[:, 1:].transpose(1, 2), labels, reduction="none"
    )
    tv = (draft[:, 1:].softmax(-1) - teacher.softmax(-1)).abs().sum(-1) / 2
    weights = torch.exp(-torch.arange(2).float() / 4)
    expected = ((0.1 * ce + 0.9 * tv) * weights).mean()
    torch.testing.assert_close(ce_tv_loss(draft, teacher, labels), expected)


def test_borrowed_window_keeps_absolute_rope_positions(config):
    torch.manual_seed(11)
    attention = TargetKVAttention(config).eval()
    prefix = (torch.randn(1, 1, 40, 8), torch.randn(1, 1, 40, 8))
    block = torch.randn(1, 3, 16)
    with torch.no_grad():
        whole = attention(block, prefix, absolute_prefix_length=40)
        window = attention(
            block,
            (prefix[0][:, :, -4:], prefix[1][:, :, -4:]),
            absolute_prefix_length=40,
        )
    torch.testing.assert_close(whole, window)
    with pytest.raises(ValueError, match="exceeds the confirmed prefix"):
        attention(block, prefix, absolute_prefix_length=2)


def test_vllm_paged_bridge_matches_dense_window(config):
    pytest.importorskip("vllm.v1.spec_decode.hspec_kv")
    torch.manual_seed(19)
    names = [f"model.layers.{i - 1}.self_attn.attn" for i in (3, 4, 5)]
    caches = {name: torch.randn(4, 2, 1, 16) for name in names}
    before = {name: cache.clone() for name, cache in caches.items()}
    table = {name: torch.tensor([2, 0, 3, 1]) for name in names}
    last_hidden = torch.randn(1, 5, 16)
    borrowed = context_from_vllm_pages(last_hidden, caches, table, 7, config)
    dense_kv = []
    for name in names:
        logical = torch.cat([caches[name][page] for page in table[name]], dim=0)[:7]
        key, value = logical.split(8, dim=-1)
        dense_kv.append((key.transpose(0, 1)[None], value.transpose(0, 1)[None]))
        torch.testing.assert_close(caches[name], before[name])
    model = HSpecReference(config).eval()
    block = torch.tensor([[2, 31, 31]])
    with torch.no_grad():
        expected = model(block, TargetContext(last_hidden, tuple(dense_kv), 7))
        actual = model(block, borrowed)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("proposals", "expected", "accepted", "limit"),
    [
        ([4, 5, 6], [3], 0, 4),
        ([3, 5, 6], [3, 7], 1, 4),
        ([3, 7, 8], [3, 7, 8, 9], 3, 4),
        ([3, 7, 8], [3, 7], 2, 2),
    ],
)
def test_greedy_verifier_rejects_first_mismatch_and_bounds_extra_token(
    proposals, expected, accepted, limit
):
    logits = torch.full((1, 4, 10), -1.0)
    for index, token in enumerate([3, 7, 8, 9]):
        logits[0, index, token] = 1.0
    output, count = greedy_verify_block(logits, torch.tensor([proposals]), limit=limit)
    assert output.tolist() == [expected]
    assert count == accepted
