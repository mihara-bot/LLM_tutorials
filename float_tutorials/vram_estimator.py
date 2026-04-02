#!/usr/bin/env python3
"""
LLM Pre-training / SFT VRAM estimator.

This script estimates per-GPU peak memory for:
- Pre-training
- Full SFT
- LoRA SFT
- QLoRA SFT

It supports:
- Loading experiment/model/tokenizer YAMLs from local config files
- Manual architecture arguments
- Total-parameter-only mode with manual activation override
- LoRA trainable parameter estimation from rank + target modules
- Simple sharding approximations for DDP / ZeRO / FSDP

The estimator is intentionally approximate. It is designed for capacity
planning and "will this fit?" reasoning rather than exact profiler parity.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DTYPE_BYTES = {
    "fp32": 4.0,
    "float32": 4.0,
    "f32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "f16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
    "4bit": 0.5,
}

OPTIMIZER_STATE_BYTES = {
    "adamw": 8.0,
    "adam": 8.0,
    "sgd": 0.0,
    "momentum": 4.0,
    "nesterov": 4.0,
    # Approximation. Real Muon setups can be hybrid and slightly larger.
    "muon": 4.0,
}

LORA_TARGET_ALIASES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "all-linear": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


@dataclass
class ModelSpec:
    total_params: int
    hidden_size: Optional[int]
    intermediate_size: Optional[int]
    num_hidden_layers: Optional[int]
    num_attention_heads: Optional[int]
    num_key_value_heads: Optional[int]
    vocab_size: Optional[int]
    tie_word_embeddings: bool
    gated_mlp: bool
    attention_bias: bool
    mlp_bias: bool
    kv_hidden_size: Optional[int]
    per_layer_params: Optional[int]
    embedding_params: Optional[int]
    lm_head_params: Optional[int]
    final_norm_params: Optional[int]
    source: str


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(
        f"--{name}",
        dest=dest,
        action="store_true",
        help=f"{help_text} Default: {default}.",
    )
    parser.add_argument(
        f"--no-{name}",
        dest=dest,
        action="store_false",
        help=f"Disable: {help_text.lower()}",
    )
    parser.set_defaults(**{dest: default})


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def parse_scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_yaml_subset(path: str) -> Dict[str, Any]:
    lines: List[Tuple[int, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            without_comment = strip_inline_comment(raw_line).rstrip()
            if not without_comment.strip():
                continue
            indent = len(without_comment) - len(without_comment.lstrip(" "))
            if "\t" in without_comment[:indent]:
                raise ValueError(f"Tabs are not supported in YAML fallback loader: {path}")
            lines.append((indent, without_comment.lstrip(" ")))

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for idx, (indent, content) in enumerate(lines):
        if content.startswith("- "):
            raise ValueError(
                "Fallback YAML loader only supports mappings and inline lists. "
                f"Unsupported block list in {path}: {content!r}"
            )

        while indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        key, sep, remainder = content.partition(":")
        if sep != ":":
            raise ValueError(f"Invalid YAML line in {path}: {content!r}")

        key = key.strip()
        remainder = remainder.strip()

        if remainder:
            parent[key] = parse_scalar(remainder)
            continue

        next_is_nested = idx + 1 < len(lines) and lines[idx + 1][0] > indent
        if not next_is_nested:
            parent[key] = {}
            continue

        child: Dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))

    return root


def load_yaml_or_die(path: str) -> Dict[str, Any]:
    try:
        return load_yaml_subset(path)
    except Exception as exc:
        raise SystemExit(f"Failed to parse YAML {path!r}: {exc}") from exc


def resolve_path(base_path: str, maybe_relative: str) -> str:
    candidate = Path(maybe_relative)
    if candidate.is_absolute():
        return str(candidate)
    base_dir = Path(base_path).resolve().parent
    search_roots = [base_dir, *base_dir.parents]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((base_dir / candidate).resolve())


def get_first(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return default


def normalize_dtype_name(name: str) -> str:
    lowered = name.strip().lower()
    aliases = {
        "float": "fp32",
        "float32": "fp32",
        "torch.float32": "fp32",
        "torch.float": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "torch.float16": "fp16",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
    }
    return aliases.get(lowered, lowered)


def dtype_to_bytes(dtype_name: str) -> float:
    normalized = normalize_dtype_name(dtype_name)
    if normalized not in DTYPE_BYTES:
        raise SystemExit(f"Unsupported dtype {dtype_name!r}. Supported: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[normalized]


def infer_gated_mlp(cfg: Dict[str, Any], force_gated: Optional[bool]) -> bool:
    if force_gated is not None:
        return force_gated

    family = str(cfg.get("family", "")).lower()
    model_type = str(cfg.get("model_type", cfg.get("hf_model_type", ""))).lower()
    hidden_act = str(cfg.get("hidden_act", "")).lower()

    if family in {"llama_like", "mistral_like", "qwen_like"}:
        return True
    if model_type in {"llama", "mistral", "qwen2", "qwen3", "gemma"}:
        return True
    if "swiglu" in hidden_act or hidden_act == "silu":
        return True
    return False


def build_model_spec_from_cfg(
    cfg: Dict[str, Any],
    source: str,
    force_gated: Optional[bool],
    total_params_override: Optional[int],
) -> ModelSpec:
    hidden_size = get_first(cfg, "hidden_size", "n_embd", "d_model")
    intermediate_size = get_first(cfg, "intermediate_size", "n_inner")
    num_hidden_layers = get_first(cfg, "num_hidden_layers", "n_layer", "num_layers")
    num_attention_heads = get_first(cfg, "num_attention_heads", "n_head")
    num_key_value_heads = get_first(cfg, "num_key_value_heads", default=num_attention_heads)
    vocab_size = get_first(cfg, "vocab_size")
    tie_word_embeddings = bool(cfg.get("tie_word_embeddings", True))
    attention_bias = bool(cfg.get("attention_bias", False))
    mlp_bias = bool(cfg.get("mlp_bias", False))
    gated_mlp = infer_gated_mlp(cfg, force_gated)

    if hidden_size is None or num_hidden_layers is None:
        raise SystemExit(
            f"Model config {source!r} is missing hidden_size/num_hidden_layers-style fields."
        )

    hidden_size = int(hidden_size)
    num_hidden_layers = int(num_hidden_layers)

    if num_attention_heads is not None:
        num_attention_heads = int(num_attention_heads)
    if num_key_value_heads is not None:
        num_key_value_heads = int(num_key_value_heads)
    if vocab_size is not None:
        vocab_size = int(vocab_size)

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    intermediate_size = int(intermediate_size)

    kv_hidden_size: Optional[int] = None
    if num_attention_heads is not None and num_key_value_heads is not None:
        kv_hidden_size = hidden_size * num_key_value_heads // num_attention_heads

    per_layer_params: Optional[int] = None
    embedding_params: Optional[int] = None
    lm_head_params: Optional[int] = None
    final_norm_params: Optional[int] = None

    if kv_hidden_size is not None and vocab_size is not None:
        attention_weights = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_hidden_size
        attention_bias_params = 2 * hidden_size + 2 * kv_hidden_size if attention_bias else 0
        if gated_mlp:
            mlp_weights = 3 * hidden_size * intermediate_size
            mlp_bias_params = (2 * intermediate_size + hidden_size) if mlp_bias else 0
        else:
            mlp_weights = 2 * hidden_size * intermediate_size
            mlp_bias_params = (intermediate_size + hidden_size) if mlp_bias else 0

        norm_params = 2 * hidden_size
        per_layer_params = attention_weights + attention_bias_params + mlp_weights + mlp_bias_params + norm_params
        embedding_params = vocab_size * hidden_size
        lm_head_params = 0 if tie_word_embeddings else vocab_size * hidden_size
        final_norm_params = hidden_size
        total_params = num_hidden_layers * per_layer_params + embedding_params + lm_head_params + final_norm_params
    else:
        if total_params_override is None:
            raise SystemExit(
                "Could not compute total parameters from model config because vocab_size or attention heads "
                "are missing. Pass --total-params to override."
            )
        total_params = int(total_params_override)

    if total_params_override is not None:
        total_params = int(total_params_override)

    return ModelSpec(
        total_params=total_params,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        tie_word_embeddings=tie_word_embeddings,
        gated_mlp=gated_mlp,
        attention_bias=attention_bias,
        mlp_bias=mlp_bias,
        kv_hidden_size=kv_hidden_size,
        per_layer_params=per_layer_params,
        embedding_params=embedding_params,
        lm_head_params=lm_head_params,
        final_norm_params=final_norm_params,
        source=source,
    )


def build_model_spec_from_args(args: argparse.Namespace) -> ModelSpec:
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_hidden_layers = args.num_hidden_layers
    num_attention_heads = args.num_attention_heads
    num_key_value_heads = args.num_key_value_heads or args.num_attention_heads
    vocab_size = args.vocab_size
    tie_word_embeddings = args.tie_word_embeddings
    gated_mlp = args.gated_mlp
    attention_bias = args.attention_bias
    mlp_bias = args.mlp_bias

    if hidden_size is None or num_hidden_layers is None:
        if args.total_params is None:
            raise SystemExit(
                "Provide either --exp-config / --model-config, or manual architecture args "
                "like --hidden-size and --num-hidden-layers, or at least --total-params."
            )
        return ModelSpec(
            total_params=int(args.total_params),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
            gated_mlp=gated_mlp,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            kv_hidden_size=None,
            per_layer_params=None,
            embedding_params=None,
            lm_head_params=None,
            final_norm_params=None,
            source="manual",
        )

    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    cfg = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "tie_word_embeddings": tie_word_embeddings,
        "attention_bias": attention_bias,
        "mlp_bias": mlp_bias,
        "hidden_act": "silu" if gated_mlp else "gelu",
    }
    return build_model_spec_from_cfg(cfg, "manual", gated_mlp, args.total_params)


def build_model_spec(args: argparse.Namespace) -> Tuple[ModelSpec, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    exp_cfg: Dict[str, Any] = {}
    model_cfg: Dict[str, Any] = {}
    tokenize_cfg: Dict[str, Any] = {}
    optim_cfg: Dict[str, Any] = {}
    fsdp_cfg: Dict[str, Any] = {}

    if args.exp_config:
        exp_cfg = load_yaml_or_die(args.exp_config)
        if "model_config" in exp_cfg:
            model_cfg = load_yaml_or_die(resolve_path(args.exp_config, str(exp_cfg["model_config"])))
        if "tokenize_config" in exp_cfg:
            tokenize_cfg = load_yaml_or_die(resolve_path(args.exp_config, str(exp_cfg["tokenize_config"])))
        if "optim_config" in exp_cfg:
            optim_cfg = load_yaml_or_die(resolve_path(args.exp_config, str(exp_cfg["optim_config"])))
        if "fsdp_config" in exp_cfg:
            fsdp_cfg = load_yaml_or_die(resolve_path(args.exp_config, str(exp_cfg["fsdp_config"])))

    if args.model_config:
        model_cfg = load_yaml_or_die(args.model_config)

    if model_cfg:
        if args.model_config:
            source = args.model_config
        elif args.exp_config and "model_config" in exp_cfg:
            source = resolve_path(args.exp_config, str(exp_cfg["model_config"]))
        else:
            source = "model_config"
        model_spec = build_model_spec_from_cfg(model_cfg, str(source), args.gated_mlp_override, args.total_params)
    else:
        model_spec = build_model_spec_from_args(args)

    return model_spec, exp_cfg, model_cfg, tokenize_cfg, optim_cfg, fsdp_cfg


def optimizer_state_bytes(optimizer_name: str, override: Optional[float]) -> Tuple[str, float, str]:
    if override is not None:
        return optimizer_name, float(override), "manual override"

    normalized = optimizer_name.lower()
    if normalized == "auto":
        normalized = "adamw"

    if normalized not in OPTIMIZER_STATE_BYTES:
        raise SystemExit(
            f"Unsupported optimizer {optimizer_name!r}. Pass --optimizer-state-bytes if you want a custom estimate."
        )
    note = "built-in heuristic"
    if normalized == "muon":
        note = "approximate Muon momentum-only state; real hybrid setups can be slightly larger"
    return normalized, OPTIMIZER_STATE_BYTES[normalized], note


def infer_optimizer_name(args: argparse.Namespace, optim_cfg: Dict[str, Any]) -> str:
    if args.optimizer != "auto":
        return args.optimizer
    if optim_cfg and "name" in optim_cfg:
        return str(optim_cfg["name"]).lower()
    return "adamw"


def infer_precision(args: argparse.Namespace, model_cfg: Dict[str, Any], fsdp_cfg: Dict[str, Any]) -> str:
    if args.param_dtype != "auto":
        return args.param_dtype
    fsdp_precision = fsdp_cfg.get("mixed_precision")
    if fsdp_precision:
        return normalize_dtype_name(str(fsdp_precision))
    torch_dtype = model_cfg.get("torch_dtype")
    if torch_dtype:
        return normalize_dtype_name(str(torch_dtype))
    return "bf16"


def default_activation_coeff(activation_checkpointing: bool, flash_attn: bool) -> float:
    if activation_checkpointing:
        return 12.0
    if flash_attn:
        return 20.0
    return 20.0


def get_seq_len(args: argparse.Namespace, tokenize_cfg: Dict[str, Any]) -> Optional[int]:
    if args.seq_len is not None:
        return args.seq_len
    for key in ("seq_len", "max_length", "block_size", "model_max_length"):
        if key in tokenize_cfg and tokenize_cfg[key] is not None:
            return int(tokenize_cfg[key])
    return None


def get_micro_batch_size(args: argparse.Namespace, exp_cfg: Dict[str, Any]) -> Optional[int]:
    if args.micro_batch_size is not None:
        return args.micro_batch_size
    train_cfg = exp_cfg.get("train")
    if isinstance(train_cfg, dict) and train_cfg.get("micro_batch_size") is not None:
        return int(train_cfg["micro_batch_size"])
    return None


def shard_divisor(sharding: str, world_size: int, category: str) -> int:
    if world_size <= 1 or sharding in {"none", "ddp"}:
        return 1
    if sharding == "zero1":
        return world_size if category == "optimizer" else 1
    if sharding == "zero2":
        return world_size if category in {"optimizer", "grad"} else 1
    if sharding in {"zero3", "fsdp"}:
        return world_size if category in {"frozen_weight", "trainable_weight", "grad", "optimizer", "master_weight"} else 1
    raise SystemExit(f"Unsupported sharding mode {sharding!r}")


def expand_lora_targets(raw_targets: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for target in raw_targets:
        cleaned = target.strip()
        if not cleaned:
            continue
        if cleaned in LORA_TARGET_ALIASES:
            expanded.extend(LORA_TARGET_ALIASES[cleaned])
        else:
            expanded.append(cleaned)
    deduped: List[str] = []
    seen = set()
    for item in expanded:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def lora_param_count(
    model: ModelSpec,
    rank: int,
    targets: List[str],
    lora_num_layers: Optional[int],
    extra_trainable_params: int,
) -> int:
    if model.hidden_size is None or model.num_hidden_layers is None or model.intermediate_size is None:
        raise SystemExit(
            "Automatic LoRA parameter estimation requires hidden_size / intermediate_size / num_hidden_layers. "
            "Pass --lora-params directly if you only know the model size."
        )

    if model.kv_hidden_size is None:
        raise SystemExit(
            "Automatic LoRA parameter estimation requires num_attention_heads and num_key_value_heads. "
            "Pass --lora-params directly if those fields are unavailable."
        )

    if lora_num_layers is None:
        lora_num_layers = model.num_hidden_layers

    h = model.hidden_size
    i = model.intermediate_size
    kv = model.kv_hidden_size
    v = model.vocab_size

    per_target = {
        "q_proj": rank * (h + h),
        "k_proj": rank * (h + kv),
        "v_proj": rank * (h + kv),
        "o_proj": rank * (h + h),
        "gate_proj": rank * (h + i),
        "up_proj": rank * (h + i),
        "down_proj": rank * (i + h),
    }

    global_targets = {}
    if v is not None:
        global_targets = {
            "embed_tokens": rank * (v + h),
            "lm_head": rank * (v + h),
        }

    total = 0
    for target in targets:
        if target in per_target:
            total += per_target[target] * lora_num_layers
        elif target in global_targets:
            total += global_targets[target]
        else:
            raise SystemExit(
                f"Unsupported LoRA target {target!r}. Supported: "
                f"{sorted(list(per_target.keys()) + list(global_targets.keys()) + list(LORA_TARGET_ALIASES.keys()))}"
            )

    return total + extra_trainable_params


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value:,} ({value / 1_000_000_000:.3f}B)"
    if value >= 1_000_000:
        return f"{value:,} ({value / 1_000_000:.3f}M)"
    if value >= 1_000:
        return f"{value:,} ({value / 1_000:.3f}K)"
    return f"{value:,}"


def bytes_to_gib(value: float) -> float:
    return value / (1024 ** 3)


def bytes_to_gb(value: float) -> float:
    return value / (1000 ** 3)


def format_bytes(value: float) -> str:
    return f"{bytes_to_gib(value):.3f} GiB ({bytes_to_gb(value):.3f} GB)"


def compute_estimate(args: argparse.Namespace) -> Dict[str, Any]:
    model, exp_cfg, model_cfg, tokenize_cfg, optim_cfg, fsdp_cfg = build_model_spec(args)

    mode = args.mode
    precision_name = infer_precision(args, model_cfg, fsdp_cfg)
    trainable_weight_dtype = normalize_dtype_name(precision_name)
    grad_dtype = normalize_dtype_name(args.grad_dtype if args.grad_dtype != "auto" else trainable_weight_dtype)
    activation_dtype = normalize_dtype_name(args.activation_dtype if args.activation_dtype != "auto" else trainable_weight_dtype)

    optimizer_name = infer_optimizer_name(args, optim_cfg)
    optimizer_name, optimizer_bytes_per_param, optimizer_note = optimizer_state_bytes(
        optimizer_name,
        args.optimizer_state_bytes,
    )

    trainable_weight_bytes_per_param = dtype_to_bytes(trainable_weight_dtype)
    grad_bytes_per_param = dtype_to_bytes(grad_dtype)
    activation_bytes_per_element = dtype_to_bytes(activation_dtype)

    master_weight_bytes_per_param = 4.0 if args.master_weights else 0.0
    qlora_base_bytes_per_param = args.qlora_base_bytes_per_param

    if mode in {"pretrain", "full_sft"}:
        trainable_params = model.total_params + args.extra_trainable_params
        frozen_params = 0
        frozen_weight_bytes_per_param = 0.0
        lora_targets = []
        lora_rank = None
    else:
        if args.lora_params is not None:
            trainable_params = int(args.lora_params) + args.extra_trainable_params
        else:
            if args.lora_rank is None:
                raise SystemExit(
                    "LoRA / QLoRA mode requires either --lora-params or --lora-rank."
                )
            raw_targets = args.lora_targets.split(",")
            lora_targets = expand_lora_targets(raw_targets)
            if not lora_targets:
                raise SystemExit("LoRA / QLoRA mode requires at least one LoRA target module.")
            trainable_params = lora_param_count(
                model,
                rank=args.lora_rank,
                targets=lora_targets,
                lora_num_layers=args.lora_num_layers,
                extra_trainable_params=args.extra_trainable_params,
            )
        frozen_params = model.total_params
        frozen_weight_bytes_per_param = qlora_base_bytes_per_param if mode == "qlora_sft" else trainable_weight_bytes_per_param
        lora_rank = args.lora_rank
        if args.lora_params is not None:
            lora_targets = []

    frozen_weight_bytes = frozen_params * frozen_weight_bytes_per_param
    trainable_weight_bytes = trainable_params * trainable_weight_bytes_per_param
    grad_bytes = trainable_params * grad_bytes_per_param
    optimizer_state_total_bytes = trainable_params * optimizer_bytes_per_param
    master_weight_bytes = trainable_params * master_weight_bytes_per_param

    sharding = args.sharding
    world_size = args.world_size

    frozen_weight_bytes_sharded = frozen_weight_bytes / shard_divisor(sharding, world_size, "frozen_weight")
    trainable_weight_bytes_sharded = trainable_weight_bytes / shard_divisor(sharding, world_size, "trainable_weight")
    grad_bytes_sharded = grad_bytes / shard_divisor(sharding, world_size, "grad")
    optimizer_state_bytes_sharded = optimizer_state_total_bytes / shard_divisor(sharding, world_size, "optimizer")
    master_weight_bytes_sharded = master_weight_bytes / shard_divisor(sharding, world_size, "master_weight")

    model_states_bytes = (
        frozen_weight_bytes_sharded
        + trainable_weight_bytes_sharded
        + grad_bytes_sharded
        + optimizer_state_bytes_sharded
        + master_weight_bytes_sharded
    )

    seq_len = get_seq_len(args, tokenize_cfg)
    micro_batch_size = get_micro_batch_size(args, exp_cfg)

    if args.activation_gib is not None:
        linear_activation_bytes = 0.0
        attention_score_bytes = 0.0
        activation_total_bytes = float(args.activation_gib) * (1024 ** 3)
        activation_note = "manual override via --activation-gib"
        activation_coeff = None
    else:
        if micro_batch_size is None or seq_len is None:
            raise SystemExit(
                "Activation estimation requires --micro-batch-size and --seq-len, or an --exp-config "
                "with those fields, or a direct --activation-gib override."
            )
        if model.hidden_size is None or model.num_hidden_layers is None:
            raise SystemExit(
                "Activation estimation requires hidden_size and num_hidden_layers. "
                "Pass a model config / manual architecture fields, or override with --activation-gib."
            )

        activation_coeff = args.activation_coeff
        if activation_coeff is None:
            activation_coeff = default_activation_coeff(args.activation_checkpointing, args.flash_attn)

        linear_activation_bytes = (
            activation_coeff
            * model.num_hidden_layers
            * micro_batch_size
            * seq_len
            * model.hidden_size
            * activation_bytes_per_element
        )

        if args.include_attention_scores is None:
            include_attention_scores = not args.flash_attn
        else:
            include_attention_scores = args.include_attention_scores

        if include_attention_scores:
            if model.num_attention_heads is None:
                raise SystemExit(
                    "Attention-score estimation requires num_attention_heads. "
                    "Pass a model config or disable it with --no-include-attention-scores."
                )
            attention_score_bytes = (
                args.attention_score_factor
                * model.num_hidden_layers
                * micro_batch_size
                * model.num_attention_heads
                * seq_len
                * seq_len
                * activation_bytes_per_element
            )
        else:
            attention_score_bytes = 0.0

        activation_total_bytes = linear_activation_bytes + attention_score_bytes
        activation_note = "formula-based estimate"

    extra_workspace_bytes = args.extra_workspace_gib * (1024 ** 3)
    subtotal_bytes = model_states_bytes + activation_total_bytes + extra_workspace_bytes
    safety_margin_bytes = subtotal_bytes * args.safety_margin
    total_peak_bytes = subtotal_bytes + safety_margin_bytes

    return {
        "inputs": {
            "mode": mode,
            "model_source": model.source,
            "optimizer": optimizer_name,
            "optimizer_state_bytes_per_trainable_param": optimizer_bytes_per_param,
            "optimizer_note": optimizer_note,
            "param_dtype": trainable_weight_dtype,
            "grad_dtype": grad_dtype,
            "activation_dtype": activation_dtype,
            "master_weights": args.master_weights,
            "activation_checkpointing": args.activation_checkpointing,
            "flash_attn": args.flash_attn,
            "sharding": sharding,
            "world_size": world_size,
            "seq_len": seq_len,
            "micro_batch_size": micro_batch_size,
            "safety_margin": args.safety_margin,
        },
        "model": {
            "total_params": model.total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "num_hidden_layers": model.num_hidden_layers,
            "num_attention_heads": model.num_attention_heads,
            "num_key_value_heads": model.num_key_value_heads,
            "vocab_size": model.vocab_size,
            "gated_mlp": model.gated_mlp,
            "tie_word_embeddings": model.tie_word_embeddings,
            "per_layer_params": model.per_layer_params,
            "embedding_params": model.embedding_params,
            "lm_head_params": model.lm_head_params,
            "final_norm_params": model.final_norm_params,
            "kv_hidden_size": model.kv_hidden_size,
        },
        "lora": {
            "rank": lora_rank,
            "targets": lora_targets,
            "lora_num_layers": args.lora_num_layers or model.num_hidden_layers,
            "extra_trainable_params": args.extra_trainable_params,
        },
        "memory_bytes": {
            "frozen_weight": frozen_weight_bytes_sharded,
            "trainable_weight": trainable_weight_bytes_sharded,
            "grad": grad_bytes_sharded,
            "optimizer": optimizer_state_bytes_sharded,
            "master_weight": master_weight_bytes_sharded,
            "model_states_total": model_states_bytes,
            "linear_activations": linear_activation_bytes,
            "attention_scores": attention_score_bytes,
            "activations_total": activation_total_bytes,
            "extra_workspace": extra_workspace_bytes,
            "subtotal_before_safety_margin": subtotal_bytes,
            "safety_margin": safety_margin_bytes,
            "total_peak": total_peak_bytes,
        },
        "assumptions": {
            "trainable_weight_bytes_per_param": trainable_weight_bytes_per_param,
            "grad_bytes_per_param": grad_bytes_per_param,
            "frozen_weight_bytes_per_param": frozen_weight_bytes_per_param,
            "master_weight_bytes_per_param": master_weight_bytes_per_param,
            "qlora_base_bytes_per_param": qlora_base_bytes_per_param,
            "activation_coeff": activation_coeff,
            "activation_note": activation_note,
            "attention_score_factor": args.attention_score_factor,
        },
    }


def print_report(result: Dict[str, Any]) -> None:
    inputs = result["inputs"]
    model = result["model"]
    memory = result["memory_bytes"]
    assumptions = result["assumptions"]
    lora = result["lora"]

    print("=" * 78)
    print("LLM 训练显存估算")
    print("=" * 78)

    print("\n[输入假设]")
    print(f"  mode:                  {inputs['mode']}")
    print(f"  model source:          {inputs['model_source']}")
    print(f"  optimizer:             {inputs['optimizer']} ({inputs['optimizer_state_bytes_per_trainable_param']} bytes/param, {inputs['optimizer_note']})")
    print(f"  param dtype:           {inputs['param_dtype']}")
    print(f"  grad dtype:            {inputs['grad_dtype']}")
    print(f"  activation dtype:      {inputs['activation_dtype']}")
    print(f"  master weights:        {inputs['master_weights']}")
    print(f"  activation ckpt:       {inputs['activation_checkpointing']}")
    print(f"  flash attention:       {inputs['flash_attn']}")
    print(f"  sharding:              {inputs['sharding']} (world_size={inputs['world_size']})")
    print(f"  seq_len:               {inputs['seq_len']}")
    print(f"  micro_batch_size:      {inputs['micro_batch_size']}")
    print(f"  safety_margin:         {inputs['safety_margin'] * 100:.1f}%")

    print("\n[模型规模]")
    print(f"  total params:          {format_count(model['total_params'])}")
    print(f"  trainable params:      {format_count(model['trainable_params'])}")
    print(f"  frozen params:         {format_count(model['frozen_params'])}")
    if model["hidden_size"] is not None:
        print(f"  hidden_size:           {model['hidden_size']}")
    if model["intermediate_size"] is not None:
        print(f"  intermediate_size:     {model['intermediate_size']}")
    if model["num_hidden_layers"] is not None:
        print(f"  num_hidden_layers:     {model['num_hidden_layers']}")
    if model["num_attention_heads"] is not None:
        print(f"  num_attention_heads:   {model['num_attention_heads']}")
    if model["num_key_value_heads"] is not None:
        print(f"  num_kv_heads:          {model['num_key_value_heads']}")
    if model["vocab_size"] is not None:
        print(f"  vocab_size:            {model['vocab_size']}")
    print(f"  gated_mlp:             {model['gated_mlp']}")
    print(f"  tie_word_embeddings:   {model['tie_word_embeddings']}")

    if lora["rank"] is not None or lora["targets"]:
        print("\n[LoRA 配置]")
        print(f"  rank:                  {lora['rank']}")
        print(f"  targets:               {', '.join(lora['targets']) if lora['targets'] else '(manual param count)'}")
        print(f"  lora_num_layers:       {lora['lora_num_layers']}")
        print(f"  extra_trainable:       {format_count(lora['extra_trainable_params'])}")

    print("\n[模型状态显存]")
    print(f"  frozen weights:        {format_bytes(memory['frozen_weight'])}")
    print(f"  trainable weights:     {format_bytes(memory['trainable_weight'])}")
    print(f"  gradients:             {format_bytes(memory['grad'])}")
    print(f"  optimizer states:      {format_bytes(memory['optimizer'])}")
    print(f"  master weights:        {format_bytes(memory['master_weight'])}")
    print(f"  model states total:    {format_bytes(memory['model_states_total'])}")

    print("\n[激活显存]")
    print(f"  activation note:       {assumptions['activation_note']}")
    if assumptions["activation_coeff"] is not None:
        print(f"  activation coeff:      {assumptions['activation_coeff']}")
    print(f"  linear activations:    {format_bytes(memory['linear_activations'])}")
    print(f"  attention scores:      {format_bytes(memory['attention_scores'])}")
    print(f"  activations total:     {format_bytes(memory['activations_total'])}")

    print("\n[总显存]")
    print(f"  extra workspace:       {format_bytes(memory['extra_workspace'])}")
    print(f"  subtotal:              {format_bytes(memory['subtotal_before_safety_margin'])}")
    print(f"  safety margin:         {format_bytes(memory['safety_margin'])}")
    print(f"  estimated peak / GPU:  {format_bytes(memory['total_peak'])}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-GPU VRAM for LLM pre-training / SFT / LoRA / QLoRA.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pretrain", "full_sft", "lora_sft", "qlora_sft"],
        help="Training scenario to estimate.",
    )

    parser.add_argument("--exp-config", default="", help="Experiment YAML path with model/tokenize/optim references.")
    parser.add_argument("--model-config", default="", help="Model YAML path.")
    parser.add_argument("--total-params", type=int, default=None, help="Override total parameter count.")

    parser.add_argument("--hidden-size", type=int, default=None, help="Manual hidden size.")
    parser.add_argument("--intermediate-size", type=int, default=None, help="Manual MLP intermediate size.")
    parser.add_argument("--num-hidden-layers", type=int, default=None, help="Manual number of transformer layers.")
    parser.add_argument("--num-attention-heads", type=int, default=None, help="Manual number of attention heads.")
    parser.add_argument("--num-key-value-heads", type=int, default=None, help="Manual number of KV heads.")
    parser.add_argument("--vocab-size", type=int, default=None, help="Manual vocab size.")
    add_bool_arg(parser, "tie-word-embeddings", False, "Whether embedding and lm_head are tied.")
    add_bool_arg(parser, "gated-mlp", True, "Whether the MLP is gated (e.g. SwiGLU).")
    parser.add_argument(
        "--gated-mlp-override",
        choices=["true", "false"],
        default=None,
        help="Override model-config gated MLP inference.",
    )
    add_bool_arg(parser, "attention-bias", False, "Include attention bias params in parameter counting.")
    add_bool_arg(parser, "mlp-bias", False, "Include MLP bias params in parameter counting.")

    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for activation estimation.")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Micro batch size per GPU.")
    parser.add_argument(
        "--param-dtype",
        default="auto",
        help="Trainable weight dtype. Default: auto from config, else bf16.",
    )
    parser.add_argument(
        "--grad-dtype",
        default="auto",
        help="Gradient dtype. Default: same as trainable weight dtype.",
    )
    parser.add_argument(
        "--activation-dtype",
        default="auto",
        help="Activation dtype. Default: same as trainable weight dtype.",
    )
    add_bool_arg(parser, "master-weights", False, "Add FP32 master weights for trainable params.")

    parser.add_argument(
        "--optimizer",
        default="auto",
        help="Optimizer name: auto / adamw / adam / sgd / momentum / nesterov / muon.",
    )
    parser.add_argument(
        "--optimizer-state-bytes",
        type=float,
        default=None,
        help="Override optimizer state bytes per trainable param.",
    )

    parser.add_argument("--lora-params", type=int, default=None, help="Directly set LoRA/adapter trainable params.")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank for automatic adapter param counting.")
    parser.add_argument(
        "--lora-targets",
        default="q_proj,v_proj",
        help="Comma-separated LoRA targets. Aliases: attn, mlp, all-linear.",
    )
    parser.add_argument(
        "--lora-num-layers",
        type=int,
        default=None,
        help="How many transformer layers receive LoRA. Default: all layers.",
    )
    parser.add_argument(
        "--extra-trainable-params",
        type=int,
        default=0,
        help="Extra trainable params beyond base/LoRA counting, e.g. norms or special heads.",
    )
    parser.add_argument(
        "--qlora-base-bytes-per-param",
        type=float,
        default=0.57,
        help="Frozen base bytes per param for QLoRA. 0.57 is a common practical heuristic.",
    )

    add_bool_arg(parser, "activation-checkpointing", True, "Assume activation checkpointing is enabled.")
    add_bool_arg(parser, "flash-attn", True, "Assume FlashAttention-style memory optimization is enabled.")
    parser.add_argument(
        "--activation-coeff",
        type=float,
        default=None,
        help="Override the linear activation coefficient c in c*L*B*T*H*b.",
    )
    parser.add_argument(
        "--include-attention-scores",
        choices=["true", "false"],
        default=None,
        help="Whether to include the explicit L*B*A*T^2*b attention-score term. Default: auto (disabled if flash-attn is on).",
    )
    parser.add_argument(
        "--attention-score-factor",
        type=float,
        default=1.0,
        help="Scale factor for the explicit attention-score term.",
    )
    parser.add_argument(
        "--activation-gib",
        type=float,
        default=None,
        help="Directly override total activation memory in GiB.",
    )

    parser.add_argument(
        "--sharding",
        default="none",
        choices=["none", "ddp", "zero1", "zero2", "zero3", "fsdp"],
        help="Approximate model-state sharding strategy.",
    )
    parser.add_argument("--world-size", type=int, default=1, help="Number of GPUs participating in sharding.")

    parser.add_argument(
        "--extra-workspace-gib",
        type=float,
        default=0.0,
        help="Extra fixed workspace / communication buffer in GiB per GPU.",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.15,
        help="Fractional safety margin for fragmentation and runtime spikes.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a human-readable report.")

    args = parser.parse_args(argv)

    if args.gated_mlp_override is not None:
        args.gated_mlp_override = args.gated_mlp_override == "true"

    if args.include_attention_scores is not None:
        args.include_attention_scores = args.include_attention_scores == "true"

    if args.world_size < 1:
        raise SystemExit("--world-size must be >= 1")
    if args.safety_margin < 0:
        raise SystemExit("--safety-margin must be >= 0")
    if args.exp_config == "":
        args.exp_config = None
    if args.model_config == "":
        args.model_config = None

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = compute_estimate(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
