#!/usr/bin/env python3
"""Generate deterministic Megatron model architecture manifests for Explorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _refs import ref  # noqa: E402


FALLBACKS: dict[str, dict] = {
    "qwen3-0_6b": {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
    },
    "qwen3-8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
    },
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 6144,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 1000000.0,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "num_shared_experts": 0,
    },
    "deepseek-v3": {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim": 192,
        "intermediate_size": 18432,
        "vocab_size": 129280,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "rope_theta": 10000.0,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "num_shared_experts": 1,
        "first_k_dense_replace": 3,
    },
}

MODEL_LIST = [
    ("qwen3-0_6b", "Qwen3-0.6B", "dense-qknorm"),
    ("qwen3-8b", "Qwen3-8B", "dense-qknorm"),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B", "moe-qknorm"),
    ("deepseek-v3", "DeepSeek-V3", "mla-moe"),
]

MEGATRON_REFS = {
    "embed": ("megatron/core/models/gpt/gpt_model.py", "self.embedding"),
    "transformer": ("megatron/core/transformer/transformer_block.py", "class TransformerBlock"),
    "layer": ("megatron/core/transformer/transformer_layer.py", "class TransformerLayer"),
    "attention": ("megatron/core/transformer/attention.py", "class Attention"),
    "mlp": ("megatron/core/transformer/mlp.py", "class MLP"),
    "moe": ("megatron/core/transformer/moe/moe_layer.py", "class MoELayer"),
    "router": ("megatron/core/transformer/moe/router.py", "class Router"),
    "shared": ("megatron/core/transformer/moe/shared_experts.py", "class SharedExpertMLP"),
    "mla": ("megatron/core/transformer/multi_latent_attention.py", "class MultiLatentAttention"),
    "head": ("megatron/core/models/gpt/gpt_model.py", "self.output_layer"),
}

BlockDef = tuple[str, str, str, str, str, str, str, str, str | None]


def _src(key: str, symbol: str | None = None) -> tuple[str, str]:
    file, default_symbol = MEGATRON_REFS[key]
    return file, symbol or default_symbol


def _b(
    id: str,
    btype: str,
    kind: str,
    label: str,
    display_symbol: str,
    source: tuple[str, str],
    desc: str,
    note: str | None = None,
) -> BlockDef:
    source_file, grep_symbol = source
    return (id, btype, kind, label, display_symbol, grep_symbol, source_file, desc, note)


GPT_FILE = "megatron/core/models/gpt/gpt_model.py"
ATTN_FILE = "megatron/core/transformer/attention.py"
MLA_FILE = "megatron/core/transformer/multi_latent_attention.py"
LAYER_FILE = "megatron/core/transformer/transformer_layer.py"
BLOCK_FILE = "megatron/core/transformer/transformer_block.py"
MLP_FILE = "megatron/core/transformer/mlp.py"
MOE_FILE = "megatron/core/transformer/moe/moe_layer.py"

PRELUDE = [
    _b(
        "embed",
        "embed",
        "embed",
        "Token embeddings",
        "GPTModel.embedding",
        _src("embed", "self.embedding = LanguageModelEmbedding"),
        "Looks up token ids through Megatron's language-model embedding stage.",
        "tie_word_embeddings reuses this matrix as the output layer.",
    ),
]

ATT_PRENORM = _b(
    "input_norm",
    "rmsnorm",
    "norm",
    "Input RMSNorm",
    "TransformerLayer.input_layernorm",
    _src("layer", "self.input_layernorm = submodules.input_layernorm"),
    "Pre-attention normalization over hidden_size.",
)

DENSE_ATT_STEPS = [
    _b(
        "qkv_proj",
        "qkv_linear",
        "proj",
        "QKV projection",
        "SelfAttention.linear_qkv",
        _src("attention", "self.linear_qkv = submodules.linear_qkv"),
        "Fused projection producing query, key, and value tensors; supports grouped-query attention.",
    ),
    _b(
        "q_norm",
        "qk_norm",
        "norm",
        "Q head-norm",
        "SelfAttention.q_layernorm",
        _src("attention", "self.q_layernorm ="),
        "Optional per-head query normalization used when qk_layernorm is enabled.",
        "Qwen3-style QK-Norm stabilizes attention logits.",
    ),
    _b(
        "k_norm",
        "qk_norm",
        "norm",
        "K head-norm",
        "SelfAttention.k_layernorm",
        _src("attention", "self.k_layernorm ="),
        "Optional per-head key normalization paired with query normalization.",
        "Qwen3-style QK-Norm applies before attention scores.",
    ),
    _b(
        "rope",
        "rope",
        "rope",
        "RoPE (Q, K)",
        "GPTModel.rotary_pos_emb",
        (GPT_FILE, "self.rotary_pos_emb = RotaryEmbedding"),
        "Rotary position embeddings are prepared by GPTModel and applied inside attention.",
        "Qwen3 uses rope_theta=1e6.",
    ),
    _b(
        "attn",
        "attention_gqa",
        "attn",
        "Attention (GQA)",
        "Attention.core_attention",
        _src("attention", "self.core_attention = submodules.core_attention"),
        "Megatron core attention computes scaled dot-product attention over query, key, and value tensors.",
    ),
    _b(
        "o_proj",
        "o_proj",
        "proj",
        "Output projection",
        "Attention.linear_proj",
        _src("attention", "self.linear_proj = submodules.linear_proj"),
        "Projects attention output back to hidden_size.",
    ),
]

MLP_PRENORM = _b(
    "post_norm",
    "rmsnorm",
    "norm",
    "Post-attn RMSNorm",
    "TransformerLayer.pre_mlp_layernorm",
    _src("layer", "self.pre_mlp_layernorm = submodules.pre_mlp_layernorm"),
    "Pre-MLP normalization after attention and residual handling.",
)

DENSE_MLP_STEPS = [
    _b(
        "gate_up",
        "gate_up",
        "mlp",
        "Gate+Up projection",
        "MLP.linear_fc1",
        _src("mlp", "self.linear_fc1 = submodules.linear_fc1"),
        "First MLP projection; gated linear units double the intermediate width.",
    ),
    _b(
        "silu",
        "activation",
        "act",
        "SiLU + gating",
        "MLP.activation_func",
        _src("mlp", "self.activation_func = self.config.activation_func"),
        "SwiGLU activation multiplies the SiLU gate branch by the up branch.",
    ),
    _b(
        "down",
        "down",
        "mlp",
        "Down projection",
        "MLP.linear_fc2",
        _src("mlp", "self.linear_fc2 = submodules.linear_fc2"),
        "Projects intermediate activations back to hidden_size.",
    ),
]

MOE_STEPS = [
    _b(
        "router",
        "moe_router",
        "router",
        "MoE router",
        "MoELayer.router",
        _src("moe", "self.router = self.submodules.router"),
        "Top-k router scores experts and selects the active experts per token.",
    ),
    _b(
        "experts",
        "moe_experts",
        "moe",
        "Sparse expert FFNs",
        "MoELayer.experts",
        _src("moe", "self.experts = self.submodules.experts"),
        "Routed expert FFNs run sparsely; only top-k experts are active per token.",
    ),
]

MLA_ATT_STEPS = [
    _b(
        "q_a",
        "q_a_linear",
        "latent",
        "Q down-proj",
        "MultiLatentAttention.linear_q_down_proj",
        _src("mla", "self.linear_q_down_proj = build_module"),
        "Query down-projection compresses hidden states into q_lora_rank latent space.",
    ),
    _b(
        "q_a_norm",
        "mla_q_norm",
        "norm",
        "Q latent RMSNorm",
        "MultiLatentAttention.q_layernorm",
        _src("mla", "self.q_layernorm = submodules.q_layernorm"),
        "Normalizes compressed query latents before up-projection.",
    ),
    _b(
        "q_b",
        "q_b_linear",
        "latent",
        "Q up-proj",
        "MultiLatentAttention.linear_q_up_proj",
        _src("mla", "self.linear_q_up_proj = build_module"),
        "Query up-projection reconstructs full attention-head query dimensions.",
    ),
    _b(
        "kv_a",
        "kv_a_linear",
        "latent",
        "KV down-proj",
        "MultiLatentAttention.linear_kv_down_proj",
        _src("mla", "self.linear_kv_down_proj = build_module"),
        "KV down-projection creates latent KV plus the decoupled RoPE key component.",
    ),
    _b(
        "kv_a_norm",
        "mla_kv_norm",
        "norm",
        "KV latent RMSNorm",
        "MultiLatentAttention.kv_layernorm",
        _src("mla", "self.kv_layernorm = submodules.kv_layernorm"),
        "Normalizes compressed KV latents before up-projection.",
    ),
    _b(
        "kv_b",
        "kv_b_linear",
        "latent",
        "KV up-proj",
        "MultiLatentAttention.linear_kv_up_proj",
        _src("mla", "self.linear_kv_up_proj = build_module"),
        "KV up-projection reconstructs key/value heads from compressed latent KV.",
    ),
    _b(
        "rope",
        "rope",
        "rope",
        "RoPE (Q, K)",
        "MultiLatentAttention.rotary_pos_emb",
        _src("mla", "self.rotary_pos_emb = RotaryEmbedding"),
        "MLA applies rotary embeddings to decoupled positional query/key dimensions.",
    ),
    _b(
        "attn",
        "attention_mla",
        "attn",
        "MLA attention",
        "MultiLatentAttention.core_attention",
        _src("mla", "self.core_attention = build_module"),
        "Core attention runs over reconstructed MLA query, key, and value tensors.",
        "KV cache stores compressed latent KV plus RoPE key dimensions.",
    ),
    _b(
        "o_proj",
        "mla_o_proj",
        "proj",
        "Output projection",
        "MultiLatentAttention.linear_proj",
        _src("mla", "self.linear_proj = submodules.linear_proj"),
        "Projects MLA attention output back to hidden_size.",
    ),
]

MLA_MOE_STEPS = MOE_STEPS + [
    _b(
        "shared_expert",
        "shared_expert",
        "moe",
        "Shared expert FFN",
        "SharedExpertMLP",
        _src("shared", "class SharedExpertMLP"),
        "Always-active shared expert path contributes to every token.",
    )
]

HEAD = [
    _b(
        "final_norm",
        "rmsnorm",
        "norm",
        "Final RMSNorm",
        "TransformerBlock.final_layernorm",
        _src("transformer", "self.final_layernorm ="),
        "Normalizes the final decoder hidden states before logits.",
    ),
    _b(
        "lm_head",
        "lm_head",
        "head",
        "LM head",
        "GPTModel.output_layer",
        _src("head", "self.output_layer = tensor_parallel.ColumnParallelLinear"),
        "Projects hidden_size to vocabulary logits.",
        "Can share weights with embeddings when tie_word_embeddings is true.",
    ),
    _b(
        "logits",
        "logits",
        "head",
        "Logits",
        "GPTModel.output_layer",
        _src("head", "logits, _ = self.output_layer"),
        "Produces final per-token vocabulary logits.",
    ),
]


def _symbol_exists(repo_root: Path, file: str, symbol: str) -> bool:
    path = repo_root / file
    if not path.is_file():
        return False
    return any(symbol in line for line in path.read_text(encoding="utf-8").splitlines())


def make_block(bdef: BlockDef, repo_root: Path) -> dict:
    bid, btype, kind, label, display_symbol, grep_symbol, source_file, desc, note = bdef
    if not _symbol_exists(repo_root, source_file, grep_symbol):
        raise ValueError(f"missing symbol {grep_symbol!r} in {source_file}")
    block = {
        "id": bid,
        "type": btype,
        "label": label,
        "symbol": display_symbol,
        "ref": ref(repo_root, source_file, grep_symbol),
        "kind": kind,
        "desc": desc,
    }
    if note:
        block["note"] = note
    return block


def build_manifest(slug: str, label: str, family: str, repo_root: Path) -> dict:
    cfg = dict(FALLBACKS[slug])
    mb = lambda bdef: make_block(bdef, repo_root)
    prelude = [mb(b) for b in PRELUDE]
    head = [mb(b) for b in HEAD]

    if family == "dense-qknorm":
        layers = [
            {
                "repeat": cfg["num_hidden_layers"],
                "label": "decoder layer",
                "branches": [
                    {
                        "name": "attn",
                        "accent": "#10b981",
                        "preNorm": mb(ATT_PRENORM),
                        "steps": [mb(b) for b in DENSE_ATT_STEPS],
                    },
                    {
                        "name": "mlp",
                        "accent": "#6366f1",
                        "preNorm": mb(MLP_PRENORM),
                        "steps": [mb(b) for b in DENSE_MLP_STEPS],
                    },
                ],
            }
        ]
    elif family == "moe-qknorm":
        layers = [
            {
                "repeat": cfg["num_hidden_layers"],
                "label": "MoE decoder layer",
                "branches": [
                    {
                        "name": "attn",
                        "accent": "#10b981",
                        "preNorm": mb(ATT_PRENORM),
                        "steps": [mb(b) for b in DENSE_ATT_STEPS],
                    },
                    {
                        "name": "moe",
                        "accent": "#f472b6",
                        "preNorm": mb(MLP_PRENORM),
                        "steps": [mb(b) for b in MOE_STEPS],
                    },
                ],
            }
        ]
    elif family == "mla-moe":
        first_dense = cfg.get("first_k_dense_replace", 0)
        moe_layers = cfg["num_hidden_layers"] - first_dense
        attn_branch = {
            "name": "attn",
            "accent": "#10b981",
            "preNorm": mb(ATT_PRENORM),
            "steps": [mb(b) for b in MLA_ATT_STEPS],
        }
        layers = [
            {
                "repeat": first_dense,
                "label": f"dense layer x first {first_dense}",
                "branches": [
                    attn_branch,
                    {
                        "name": "mlp",
                        "accent": "#6366f1",
                        "preNorm": mb(MLP_PRENORM),
                        "steps": [mb(b) for b in DENSE_MLP_STEPS],
                    },
                ],
            },
            {
                "repeat": moe_layers,
                "label": f"MoE layer x {moe_layers}",
                "branches": [
                    attn_branch,
                    {
                        "name": "moe",
                        "accent": "#f472b6",
                        "preNorm": mb(MLP_PRENORM),
                        "steps": [mb(b) for b in MLA_MOE_STEPS],
                    },
                ],
            },
        ]
    else:
        raise ValueError(f"unknown model family: {family}")

    return {
        "model": label,
        "slug": slug,
        "family": family,
        "source": "fallback",
        "config": cfg,
        "prelude": prelude,
        "layers": layers,
        "head": head,
    }


def compute_total_params(cfg: dict) -> int:
    d = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    vocab = cfg["vocab_size"]
    tied = cfg.get("tie_word_embeddings", False)

    embed = vocab * d
    final_norm = d
    lm_head = 0 if tied else vocab * d

    if cfg.get("q_lora_rank") is not None:
        ql = cfg["q_lora_rank"]
        kvl = cfg["kv_lora_rank"]
        qn = cfg["qk_nope_head_dim"]
        qr = cfg["qk_rope_head_dim"]
        vd = cfg["v_head_dim"]
        mla_attn = (
            d * ql
            + ql
            + ql * heads * (qn + qr)
            + d * (kvl + qr)
            + kvl
            + kvl * heads * (qn + vd)
            + heads * vd * d
            + 2 * d
        )
        first_dense = cfg.get("first_k_dense_replace", 0)
        moe_layers = layers - first_dense
        dense_mlp = 3 * d * cfg["intermediate_size"]
        experts = cfg["num_experts"] * 3 * d * cfg["moe_intermediate_size"]
        router = d * cfg["num_experts"]
        shared = cfg.get("num_shared_experts", 0) * 3 * d * cfg["moe_intermediate_size"]
        return (
            embed
            + first_dense * (mla_attn + dense_mlp)
            + moe_layers * (mla_attn + router + experts + shared)
            + final_norm
            + lm_head
        )

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    attn = d * (q_dim + 2 * kv_dim) + head_dim + head_dim + q_dim * d + 2 * d

    if cfg.get("num_experts"):
        ffn = d * cfg["num_experts"] + cfg["num_experts"] * 3 * d * cfg["moe_intermediate_size"]
    else:
        ffn = 3 * d * cfg["intermediate_size"]

    return embed + layers * (attn + ffn) + final_norm + lm_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Path to Megatron-LM repo root")
    parser.add_argument(
        "--out-dir",
        default="explorer/public/data/models",
        help="Directory to write model manifests",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for slug, label, family in MODEL_LIST:
        manifest = build_manifest(slug, label, family, repo_root)
        total_params = compute_total_params(manifest["config"])
        (out_dir / f"{slug}.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        index.append(
            {
                "slug": slug,
                "label": label,
                "family": family,
                "totalParams": total_params,
            }
        )
        print(f"wrote {out_dir / f'{slug}.json'} : {label} totalParams={total_params:,}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'index.json'} : {len(index)} models")


if __name__ == "__main__":
    main()
