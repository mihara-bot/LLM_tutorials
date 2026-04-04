"""
MoE Model Explorer
==================
探索 MoE 模型的结构、路由机制和 Expert 分布。
可以只加载 config 来分析结构，也可以加载完整权重来分析路由行为。

Usage:
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode structure
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode architecture
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode compare --dense-model Qwen/Qwen3-4B
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode routing --load-weights

Requirements:
    pip install transformers torch accelerate
    # Optional for quantized loading:
    pip install bitsandbytes
"""

import argparse
from collections import Counter

import torch
from transformers import AutoConfig, AutoTokenizer


# ============================================================
# Part 1: 结构分析（不需要加载权重，只需要 config）
# ============================================================

def analyze_structure(model_name: str):
    """分析 MoE 模型的架构参数，对比 Dense 模型。"""
    print("=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print("\n📋 完整 config 关键字段：")

    moe_keywords = [
        "num_experts", "experts_per_tok", "num_experts_per_tok",
        "n_routed_experts", "n_shared_experts", "num_local_experts",
        "moe", "expert", "router", "gate", "shared",
        "intermediate_size", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads",
        "moe_intermediate_size", "shared_expert_intermediate_size",
        "norm_topk_prob", "routed_scaling_factor",
    ]

    config_dict = config.to_dict()
    for key, value in sorted(config_dict.items()):
        if any(kw in key.lower() for kw in moe_keywords):
            print(f"  {key}: {value}")

    # 提取关键参数
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)

    num_experts = (
        getattr(config, "num_experts", None) or
        getattr(config, "num_local_experts", None) or
        getattr(config, "n_routed_experts", None)
    )
    experts_per_tok = (
        getattr(config, "num_experts_per_tok", None) or
        getattr(config, "experts_per_tok", None) or
        getattr(config, "num_experts_per_token", None) or
        getattr(config, "top_k", None)
    )
    n_shared_experts = getattr(config, "n_shared_experts", 0) or 0

    moe_inter = (
        getattr(config, "moe_intermediate_size", None) or
        getattr(config, "intermediate_size", None)
    )
    shared_inter = getattr(config, "shared_expert_intermediate_size", moe_inter)

    print(f"\n🏗️  架构概览:")
    print(f"  Hidden Size:          {hidden_size}")
    print(f"  Num Layers:           {num_layers}")
    print(f"  Routed Experts:       {num_experts}")
    print(f"  Shared Experts:       {n_shared_experts}")
    print(f"  Activated per Token:  {experts_per_tok} routed + {n_shared_experts} shared")
    print(f"  Expert FFN Dim:       {moe_inter}")
    if n_shared_experts > 0:
        print(f"  Shared Expert FFN Dim:{shared_inter}")

    # 参数量估算
    if all(v is not None for v in [hidden_size, num_layers, num_experts, moe_inter]):
        params_per_expert = 3 * hidden_size * moe_inter
        total_routed = num_experts * params_per_expert * num_layers
        total_shared = n_shared_experts * 3 * hidden_size * shared_inter * num_layers
        active_routed = experts_per_tok * params_per_expert * num_layers

        num_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        attn_params_per_layer = (
            hidden_size * (num_heads * head_dim) +
            hidden_size * (num_kv_heads * head_dim) +
            hidden_size * (num_kv_heads * head_dim) +
            (num_heads * head_dim) * hidden_size
        )
        total_attn = attn_params_per_layer * num_layers
        router_params = hidden_size * num_experts * num_layers

        total_params = total_routed + total_shared + total_attn + router_params
        active_params = active_routed + total_shared + total_attn + router_params

        print(f"\n📊 参数量估算:")
        print(f"  每个 Routed Expert FFN:     {params_per_expert / 1e6:.1f}M")
        print(f"  全部 Routed Expert FFN:     {total_routed / 1e9:.1f}B")
        print(f"  Shared Expert FFN 合计:     {total_shared / 1e9:.1f}B")
        print(f"  Attention 合计:             {total_attn / 1e9:.1f}B")
        print(f"  Router 合计:                {router_params / 1e6:.1f}M")
        print(f"  ─────────────────────────")
        print(f"  总参数量 (估算):            {total_params / 1e9:.1f}B")
        print(f"  每 token 激活参数 (估算):   {active_params / 1e9:.1f}B")
        print(f"  激活比例:                   {active_params / total_params * 100:.1f}%")

        print(f"\n🔍 与 Dense 模型对比:")
        print(f"  等价容量 Dense 模型:  ~{total_params / 1e9:.0f}B 参数")
        print(f"  等价算力 Dense 模型:  ~{active_params / 1e9:.0f}B 参数")
        print(f"  → MoE 用 {active_params / total_params * 100:.0f}% 的计算量获得了 100% 的模型容量")


# ============================================================
# Part 2: 模型结构打印
# ============================================================

def print_model_architecture(model_name: str):
    """打印模型架构，对比 Dense 层和 MoE 层。"""
    print("\n" + "=" * 70)
    print("模型层级结构（注意 MoE 层和 Dense 层的区别）")
    print("=" * 70)

    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        )

    for name, module in model.named_modules():
        depth = name.count(".")
        if depth <= 4:
            parts = name.split(".")
            skip = False
            for part in parts:
                if part.isdigit() and int(part) > 1:
                    skip = True
                    break
            if not skip:
                indent = "  " * depth
                class_name = module.__class__.__name__
                if any(kw in class_name.lower() for kw in ["moe", "expert", "gate", "router"]):
                    print(f"{indent}🔶 {name}: {class_name}")
                elif any(kw in class_name.lower() for kw in ["attention", "attn"]):
                    print(f"{indent}🔷 {name}: {class_name}")
                else:
                    print(f"{indent}   {name}: {class_name}")


# ============================================================
# Part 3: 路由分析（需要加载权重）
# ============================================================

def analyze_routing(model_name: str, texts: list = None):
    """加载模型，分析每层的 expert 选择分布。"""
    from transformers import AutoModelForCausalLM

    if texts is None:
        texts = [
            "The capital of France is",
            "def fibonacci(n):\n    if n <= 1:\n        return n",
            "在自然语言处理领域，大型语言模型",
            "E = mc^2 is the famous equation by Einstein that",
        ]

    print("\n" + "=" * 70)
    print("路由分析: 每个 token 被发送到哪些 Expert")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("\n正在加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("提示: 请确保有足够 GPU 显存，或在代码中启用 load_in_4bit=True")
        return

    model.eval()

    # 注册 hook 捕获 router 输出
    router_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                router_outputs[layer_idx] = output
            else:
                router_outputs[layer_idx] = output
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ["gate", "router"]):
            parts = name.split(".")
            layer_idx = None
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break
            if layer_idx is not None:
                hook = module.register_forward_hook(make_hook(layer_idx))
                hooks.append(hook)

    print(f"已注册 {len(hooks)} 个 router hook\n")

    for text in texts:
        print(f"📝 Input: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        router_outputs.clear()
        with torch.no_grad():
            model(**inputs)

        if not router_outputs:
            print("  ⚠️ 未捕获到 router 输出\n")
            continue

        total_expert_counts = Counter()

        for layer_idx in sorted(router_outputs.keys())[:5]:
            output = router_outputs[layer_idx]
            try:
                if hasattr(output, "shape"):
                    logits = output
                elif isinstance(output, tuple) and len(output) >= 2:
                    logits = output[0] if hasattr(output[0], "shape") else output[1]
                else:
                    continue

                if logits.dim() >= 2:
                    k = min(8, logits.shape[-1])
                    topk_indices = torch.topk(
                        logits.view(-1, logits.shape[-1]), k=k, dim=-1
                    ).indices
                    expert_counts = Counter(
                        topk_indices.cpu().numpy().flatten().tolist()
                    )

                    print(f"  Layer {layer_idx}: Top-{k} experts per token")
                    most_common = expert_counts.most_common(10)
                    bar_width = 30
                    max_count = most_common[0][1] if most_common else 1
                    for expert_id, count in most_common:
                        bar = "█" * int(count / max_count * bar_width)
                        print(f"    Expert {expert_id:3d}: {bar} ({count})")
                        total_expert_counts[expert_id] += count

            except Exception as e:
                print(f"  Layer {layer_idx}: 解析失败 ({e})")

        if total_expert_counts:
            print(f"\n  📊 Expert 使用分布 (前 5 层汇总):")
            all_counts = list(total_expert_counts.values())
            print(f"    使用的 Expert 总数: {len(total_expert_counts)}")
            print(f"    最常用 Expert: {total_expert_counts.most_common(3)}")
            print(f"    最少用 Expert: {total_expert_counts.most_common()[-3:]}")
            if len(all_counts) > 1:
                mean_count = sum(all_counts) / len(all_counts)
                std_count = (
                    sum((c - mean_count) ** 2 for c in all_counts) / len(all_counts)
                ) ** 0.5
                print(f"    负载均衡 (CV): {std_count / mean_count:.3f} (越低越均衡)")
        print()

    for hook in hooks:
        hook.remove()


# ============================================================
# Part 4: Dense vs MoE 配置对比
# ============================================================

def compare_dense_vs_moe(dense_model: str, moe_model: str):
    """对比 Dense 和 MoE 模型的 config 差异。"""
    print("\n" + "=" * 70)
    print(f"Dense vs MoE 配置对比")
    print(f"  Dense: {dense_model}")
    print(f"  MoE:   {moe_model}")
    print("=" * 70)

    dense_config = AutoConfig.from_pretrained(
        dense_model, trust_remote_code=True
    ).to_dict()
    moe_config = AutoConfig.from_pretrained(
        moe_model, trust_remote_code=True
    ).to_dict()

    all_keys = sorted(set(list(dense_config.keys()) + list(moe_config.keys())))
    print(f"\n{'Key':<40} {'Dense':<20} {'MoE':<20}")
    print("─" * 80)

    for key in all_keys:
        dense_val = dense_config.get(key, "—")
        moe_val = moe_config.get(key, "—")
        if dense_val != moe_val:
            marker = "🆕" if dense_val == "—" else "📝"
            print(f"{marker} {key:<38} {str(dense_val):<20} {str(moe_val):<20}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Model Explorer")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-30B-A3B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--mode", type=str, default="structure",
        choices=["structure", "architecture", "routing", "compare"],
        help="分析模式: structure(参数分析), architecture(层级结构), "
             "routing(路由分析), compare(Dense对比)",
    )
    parser.add_argument(
        "--load-weights", action="store_true",
        help="是否加载完整权重（routing 模式需要）",
    )
    parser.add_argument(
        "--dense-model", type=str, default="Qwen/Qwen3-4B",
        help="用于对比的 Dense 模型（compare 模式）",
    )

    args = parser.parse_args()

    if args.mode == "structure":
        analyze_structure(args.model)

    elif args.mode == "architecture":
        analyze_structure(args.model)
        print_model_architecture(args.model)

    elif args.mode == "routing":
        if not args.load_weights:
            print("⚠️  routing 模式建议加上 --load-weights")
            analyze_structure(args.model)
        else:
            analyze_routing(args.model)

    elif args.mode == "compare":
        compare_dense_vs_moe(args.dense_model, args.model)
