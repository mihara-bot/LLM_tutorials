# MoE 模型快速上手教程

> **目标读者**：熟悉 LLaMA / Qwen 等 Dense Transformer 预训练的研究者
> **核心思路**：以 Mixtral 建立直觉，以 DeepSeek-V3 理解前沿设计，以代码脚本动手探索

---

## 1. 从 Dense 到 MoE：一句话理解核心区别

你已经非常熟悉 LLaMA / Qwen 的标准 Transformer Block：

```
Input → RMSNorm → Attention → Residual → RMSNorm → FFN(SwiGLU) → Residual → Output
```

**MoE 模型只改了一个地方：把单个 FFN 替换成了 N 个并行的 FFN（称为 Expert），每个 token 只选其中 K 个来计算。** 其他一切（Attention、Normalization、Residual、Tokenizer）基本不变。

```
                        ┌─── Expert 0 (SwiGLU FFN)
                        ├─── Expert 1 (SwiGLU FFN)
Input → RMSNorm → Attn → Router ──┤    ...                    → 加权求和 → Residual → Output
                        ├─── Expert N-2 (SwiGLU FFN)
                        └─── Expert N-1 (SwiGLU FFN)
                              ↑
                        只激活 Top-K 个
```

这带来一个关键性质：**总参数量远大于激活参数量**。例如 DeepSeek-V3 总共 671B 参数，但每个 token 只激活 37B——推理成本接近一个 37B 的 Dense 模型，但模型容量（知识储备）接近 671B。

### 1.1 为什么要这么做？

Dense model 的 scaling law 告诉我们：模型越大，性能越好。但大模型的训练和推理成本是线性甚至超线性增长的。MoE 提供了一条"低成本获得大容量"的路径：

| 维度 | Dense (如 LLaMA-3 70B) | MoE (如 Mixtral 8x7B) |
|------|------------------------|----------------------|
| 总参数量 | 70B | ~47B |
| 每 token 激活参数量 | 70B | ~13B |
| 推理速度 | 基准 | 快得多（激活量小） |
| 显存占用 | 70B weights | ~47B weights（全部要加载） |
| 预训练效率 | 基准 | 每 FLOP 的学习效率更高 |
| 性能 | 基准 | 相当或更好 |

**MoE 的核心 trade-off：推理显存不省（所有 expert 都要加载），但计算量大幅下降。**

---

## 2. MoE Layer 的四个核心组件

### 2.1 Expert（专家网络）

每个 Expert 就是一个标准的 FFN。以 Mixtral 为例，每个 Expert 就是一个和 Mistral-7B 完全相同的 SwiGLU 子网络：

```python
# Mixtral 的每个 Expert（伪代码）
def expert_forward(x):
    # 和 Mistral-7B 的 FFN 完全一样
    gate = self.w1(x)          # Linear: hidden_dim → intermediate_dim
    up = self.w3(x)            # Linear: hidden_dim → intermediate_dim
    return self.w2(silu(gate) * up)  # Linear: intermediate_dim → hidden_dim
```

不同 MoE 模型的 expert 设计差异主要在**粒度**上：

- **Mixtral**：8 个 expert，每个和 Mistral-7B 的 FFN 一样大（coarse-grained）
- **DeepSeek-V3**：256 个 routed expert + 1 个 shared expert，每个 expert 更小（fine-grained）
- **Qwen3 MoE (235B-A22B)**：128 个 expert，激活 8 个，无 shared expert

### 2.2 Router / Gate（路由器 / 门控网络）

Router 决定每个 token 被发送给哪些 Expert。它通常是一个简单的线性层：

```python
# Router 伪代码
router_logits = x @ W_gate  # (batch*seq, hidden_dim) @ (hidden_dim, num_experts) → (batch*seq, num_experts)
router_weights = softmax(topk(router_logits, k=K))  # 只保留 Top-K 个 expert 的权重
```

几种常见的 routing 策略：

| 策略 | 描述 | 代表模型 |
|------|------|---------|
| Top-2 + Softmax | 选得分最高的 2 个 expert，softmax 归一化 | Mixtral |
| Top-K + Sigmoid + Normalize | 用 sigmoid（而非 softmax）计算 affinity，然后 Top-K 后归一化 | DeepSeek-V3 (K=8) |
| Expert Choice | 反过来让 expert 选 token（而非 token 选 expert） | Google EC Routing |

### 2.3 Load Balancing（负载均衡）

这是 MoE 训练中最关键的挑战。如果没有负载均衡机制，训练很容易出现 **Expert Collapse**——少数 expert 被频繁选中，其余 expert 被闲置，最终退化成一个小的 Dense 模型。

**为什么会坍缩？** 直觉上：如果 Expert A 当前比 Expert B 稍好一点 → Router 更多地选 A → A 获得更多训练数据 → A 变得更好 → Router 更偏向 A → 正反馈循环 → 最终只有少数 expert 被使用。

**三种主流解法：**

**方法一：Auxiliary Loss（辅助损失）**

在主训练 loss 之外，加一个鼓励 token 均匀分配的 loss：

```python
# Switch Transformer / Mixtral 风格的 auxiliary loss
# f_i: expert i 被选中的频率
# P_i: router 分配给 expert i 的平均概率
aux_loss = alpha * N * sum(f_i * P_i for i in range(N))
# alpha 通常取 0.01 左右，N 是 expert 数量
```

问题：alpha 很难调。太大→强制均匀分配，损害模型性能；太小→无法阻止坍缩。

**方法二：Auxiliary-Loss-Free（DeepSeek-V3 的创新）**

不加任何辅助 loss。取而代之，给每个 expert 的 gate logit 加一个可调的 bias 项：

```python
# DeepSeek-V3 的做法
gate_logits = x @ W_gate + bias  # bias 是每个 expert 的偏置
# bias 不参与梯度更新，而是根据 expert 负载动态调整：
# 如果 expert_i 负载过高 → 降低 bias_i
# 如果 expert_i 负载过低 → 增加 bias_i
```

这种方法不会干扰主 loss 的梯度，是当前最先进的方案。

**方法三：Expert Choice Routing**

让 expert 去选 token（而非 token 选 expert），天然保证每个 expert 处理固定数量的 token。

### 2.4 Output Aggregation（输出聚合）

选中 K 个 expert 后，最终输出是它们的加权和：

```python
# 假设 token x 被路由到 expert_i 和 expert_j
output = w_i * Expert_i(x) + w_j * Expert_j(x)
# 其中 w_i, w_j 是 router 给出的归一化权重
```

---

## 3. 典型架构深度剖析：DeepSeek-V3

DeepSeek-V3 是当前信息最公开、架构创新最多的 MoE 模型。我们来逐层拆解。

### 3.1 总体规格

| 参数 | 值 |
|------|-----|
| 总参数量 | 671B |
| 激活参数量 | 37B（每 token） |
| Transformer 层数 | 61 |
| Hidden Dimension | 7168 |
| Attention | Multi-head Latent Attention (MLA) |
| FFN 类型 | DeepSeekMoE |
| Expert 总数 | 1 shared + 256 routed（每层） |
| 激活 Expert 数 | 1 shared + 8 routed（每 token） |
| 每个 Expert 的中间维度 | 2048 (routed), 2048 (shared) |
| Vocabulary | 128K tokens |
| 上下文长度 | 128K tokens |
| 预训练数据 | 14.8T tokens |
| 训练成本 | 2.788M H800 GPU hours |

### 3.2 MLA（Multi-head Latent Attention）

这不是 MoE 独有的，但它是 DeepSeek-V3 的重要创新，值得了解。

你熟悉的 GQA（Grouped-Query Attention）通过减少 KV head 来降低 KV cache。MLA 走了一条不同的路：**把 K 和 V 联合压缩到一个低维 latent 向量中**，推理时只缓存这个压缩向量，需要时再解压。

```
标准 MHA:  cache K (d_head × n_heads) + V (d_head × n_heads) per token
GQA:       cache K (d_head × n_kv_groups) + V (d_head × n_kv_groups) per token
MLA:       cache 一个 compressed_kv (d_compressed) per token，其中 d_compressed << d_head × n_heads
```

**对你的研究的启示**：MLA 和 MoE 是正交的两个优化——MLA 优化 Attention 的 KV cache，MoE 优化 FFN 的计算量。DeepSeek 把两者结合在了同一个模型中。

### 3.3 DeepSeekMoE：Fine-Grained Experts

**关键设计理念：更多、更小的 expert，比更少、更大的 expert 好。**

对比一下：

| 设计 | Expert 数 | 激活 Expert 数 | 每个 Expert 中间维度 | 总激活 FFN 维度 |
|------|----------|--------------|-------------------|---------------|
| Mixtral 8x7B | 8 | 2 | 14336 | 28672 |
| DeepSeek-V3 | 256 | 8 | 2048 | 16384 |
| Qwen3-235B-A22B | 128 | 8 | 2560 | 20480 |

DeepSeek 的论点是：fine-grained expert 允许更灵活的知识组合。8 选 2（C(8,2)=28 种组合）vs 256 选 8（C(256,8) ≈ 4×10^13 种组合），后者的表达能力指数级更强。

### 3.4 Shared Expert（共享专家）

DeepSeek-V3 始终激活 1 个 "Shared Expert"，不经过 router。

```python
# DeepSeek-V3 MoE 层的伪代码
def deepseekmoe_forward(x):
    # 1. Shared Expert（始终激活，不经过 router）
    shared_out = shared_expert(x)

    # 2. Routed Experts（经过 router 选择 Top-8）
    gate_logits = sigmoid(x @ W_gate + bias)
    topk_indices, topk_weights = topk(gate_logits, k=8)
    topk_weights = topk_weights / topk_weights.sum()  # 归一化

    routed_out = sum(
        topk_weights[i] * experts[topk_indices[i]](x)
        for i in range(8)
    )

    # 3. 加和
    return shared_out + routed_out
```

**设计意图**：Shared Expert 学习跨任务的通用知识（如语法、常识），让 routed experts 可以更专注于特定领域知识。

有趣的是，**Qwen3 MoE 去掉了 Shared Expert**。Qwen 团队没有公开解释原因，但社区猜测：当 routed expert 数量足够多（Qwen3 用 128 个）时，shared expert 的作用可以被多个被频繁共同选中的 routed expert 自然替代。

### 3.5 Auxiliary-Loss-Free Load Balancing

前面提到过，这是 DeepSeek 最重要的训练创新之一。详细展开：

```python
# 传统方法（如 Mixtral）：
total_loss = language_model_loss + alpha * auxiliary_balance_loss
# 问题：auxiliary_balance_loss 的梯度会干扰 language_model_loss 的梯度

# DeepSeek-V3 的方法：
# 1. 每个 expert 维护一个 bias_i（不参与反向传播）
gate_logits = sigmoid(x @ W_gate) + bias  # bias 仅用于 Top-K 选择
gate_weights = sigmoid(x @ W_gate)  # 实际的权重计算不加 bias

# 2. 训练过程中动态调整 bias：
# 每若干步，统计每个 expert 的 token 负载
# if expert_i 过载 → bias_i -= gamma
# if expert_i 空闲 → bias_i += gamma
```

核心思想：**用 bias 影响路由决策（谁被选中），但不影响权重计算（选中后的贡献大小），从而不干扰 LM loss 的梯度。**

### 3.6 Multi-Token Prediction (MTP)

这不是 MoE 特有的，但 DeepSeek-V3 证明了 MTP 和 MoE 配合很好。

标准语言模型只预测下一个 token。MTP 在每个位置额外预测未来 1-2 个 token。

```
标准 LM:    position t → predict token t+1
MTP (depth=1): position t → predict token t+1 AND token t+2
```

好处有两个：一是增加了训练信号密度（每个位置提供更多梯度信息），二是训练出的模型可以直接用于 speculative decoding 来加速推理。

### 3.7 DeepSeek-V3 vs Mixtral vs Qwen3 MoE 对比

| 特性 | Mixtral 8x7B | DeepSeek-V3 | Qwen3-235B-A22B |
|------|-------------|------------|----------------|
| 总参数 | ~47B | 671B | 235B |
| 激活参数 | ~13B | 37B | 22B |
| Expert 数 | 8 | 256 routed + 1 shared | 128 routed |
| 激活 Expert | 2 | 8 routed + 1 shared | 8 |
| Expert 粒度 | 粗粒度 | 细粒度 | 中等 |
| Shared Expert | 无 | 有 | 无 |
| Routing | Top-2 Softmax | Top-8 Sigmoid + Norm | Top-8 Softmax |
| 负载均衡 | Auxiliary Loss | Auxiliary-Loss-Free (bias) | Auxiliary Loss |
| Attention | GQA | MLA | GQA |
| MoE 层位置 | 每层都是 MoE | 前 3 层 Dense + 后 58 层 MoE | 每层都是 MoE |

注意 DeepSeek-V3 的前 3 层是标准 Dense FFN（没有 MoE），这也是一个有趣的设计：早期层学习通用 token 表示，不需要专家化。

---

## 4. MoE 训练的关键工程挑战

作为一个做 pre-training 的研究者，你一定关心这些实操问题：

### 4.1 Expert Parallelism

Dense 模型的并行策略你很熟悉：Data Parallel (DP)、Tensor Parallel (TP)、Pipeline Parallel (PP)。MoE 引入了第四种：**Expert Parallelism (EP)**。

核心思想：不同的 expert 放在不同的 GPU 上。每个 token 经过 router 后，需要通过 All-to-All 通信发送到对应 expert 所在的 GPU。

```
GPU 0: Expert 0, 1, 2, 3     GPU 1: Expert 4, 5, 6, 7
          ↑ ↓ All-to-All ↑ ↓
   Token A → Expert 2 (local)    Token B → Expert 5 (local)
   Token C → Expert 6 (remote → 发送到 GPU 1)
```

**关键瓶颈：All-to-All 通信**。这是 MoE 训练中最大的工程挑战。DeepSeek-V3 通过 computation-communication overlap（计算和通信重叠）来缓解这个问题。

### 4.2 显存管理

MoE 的显存特征和 Dense 很不一样：

- **参数显存**：所有 expert 的权重都要加载（比同等激活量的 Dense 模型大很多）
- **激活显存**：只有被选中的 expert 产生激活（和 Dense 类似）
- **优化器状态**：每个 expert 都有自己的 Adam states（巨大）

实践中，通常 Expert Parallelism + Data Parallelism 混合使用，让每张 GPU 只持有部分 expert。

### 4.3 FP8 训练

DeepSeek-V3 首次在超大规模 MoE 上验证了 FP8 训练的可行性。使用 fine-grained quantization（tile-wise 1x128 for activations，block-wise 128x128 for weights），在大幅降低训练成本的同时没有明显的精度损失。这对 MoE 模型尤其重要，因为 MoE 的总参数量巨大，减少每个参数的内存占用效果显著。

---

## 5. 动手探索：MoE 模型结构分析脚本

下面提供一个 Python 脚本，帮助你用代码直接探索一个真实的 MoE 模型的内部结构。

我们选用 **Qwen3-30B-A3B**（也可以替换为 Mixtral-8x7B），因为：
- 你熟悉 Qwen 体系，方便对比 Dense 版本
- 30B 总参数，模型 config 可以直接加载；权重加载则需要对应的 GPU 显存

### 5.1 环境准备

```bash
pip install transformers torch accelerate
# 如果要加载完整权重（需要足够的 GPU 显存 / RAM）：
# pip install bitsandbytes  # 用于量化加载
```

### 5.2 完整探索脚本

```python
"""
MoE Model Explorer
==================
探索 MoE 模型的结构、路由机制和 Expert 分布。
可以只加载 config 来分析结构，也可以加载完整权重来分析路由行为。

Usage:
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode structure
    python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode routing --load-weights
"""

import argparse
import json
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
    # 打印所有和 MoE 相关的 config 字段
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

    # 提取关键参数（适配不同模型的命名方式）
    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)

    # Expert 相关参数（不同模型命名不同）
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

    # FFN 中间维度
    moe_inter = getattr(config, "moe_intermediate_size", None) or getattr(config, "intermediate_size", None)
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
        # 每个 routed expert 的 FFN 参数：3 * hidden_size * moe_inter (gate, up, down in SwiGLU)
        params_per_expert = 3 * hidden_size * moe_inter
        # 所有 routed experts
        total_routed = num_experts * params_per_expert * num_layers
        # Shared experts
        total_shared = n_shared_experts * 3 * hidden_size * shared_inter * num_layers
        # 激活的 routed expert 参数
        active_routed = experts_per_tok * params_per_expert * num_layers

        # Attention 参数（粗略估算）
        num_heads = getattr(config, "num_attention_heads", 32)
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        head_dim = hidden_size // num_heads
        attn_params_per_layer = (
            hidden_size * (num_heads * head_dim) +       # Q
            hidden_size * (num_kv_heads * head_dim) +    # K
            hidden_size * (num_kv_heads * head_dim) +    # V
            (num_heads * head_dim) * hidden_size          # O
        )
        total_attn = attn_params_per_layer * num_layers

        # Router 参数
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

        # 和等价 Dense 模型对比
        print(f"\n🔍 与 Dense 模型对比:")
        print(f"  等价容量 Dense 模型需要:    ~{total_params / 1e9:.0f}B 参数")
        print(f"  等价算力 Dense 模型:        ~{active_params / 1e9:.0f}B 参数")
        print(f"  → MoE 用 {active_params / total_params * 100:.0f}% 的计算量获得了 100% 的模型容量")


# ============================================================
# Part 2: 模型结构打印（加载模型类，不加载权重）
# ============================================================

def print_model_architecture(model_name: str):
    """打印模型架构，对比 Dense 层和 MoE 层。"""
    print("\n" + "=" * 70)
    print("模型层级结构（注意 MoE 层和 Dense 层的区别）")
    print("=" * 70)

    # 使用 meta device 避免实际加载权重
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        )

    # 打印前 2 层的结构
    for name, module in model.named_modules():
        depth = name.count(".")
        # 只打印前几层的详细结构
        if depth <= 4:
            # 过滤：只显示前 2 个 transformer 层
            parts = name.split(".")
            for part in parts:
                if part.isdigit() and int(part) > 1:
                    break
            else:
                indent = "  " * depth
                class_name = module.__class__.__name__
                # 高亮 MoE 相关的模块
                if any(kw in class_name.lower() for kw in ["moe", "expert", "gate", "router"]):
                    print(f"{indent}🔶 {name}: {class_name}")
                elif any(kw in class_name.lower() for kw in ["attention", "attn"]):
                    print(f"{indent}🔷 {name}: {class_name}")
                else:
                    print(f"{indent}   {name}: {class_name}")


# ============================================================
# Part 3: 路由分析（需要加载权重）
# ============================================================

def analyze_routing(model_name: str, texts: list[str] = None):
    """
    加载模型，对输入文本进行 forward pass，
    分析每层的 expert 选择分布。

    ⚠️ 需要足够的 GPU 显存。对于大模型，建议使用量化加载。
    """
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

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 加载模型（根据显存选择加载方式）
    print("\n正在加载模型（这可能需要几分钟）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # 如果显存不够，取消注释下一行使用 4-bit 量化:
            # load_in_4bit=True,
        )
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("提示: 对于大模型，请确保有足够的 GPU 显存，或使用 load_in_4bit=True")
        return

    model.eval()

    # 注册 hook 来捕获 router 的输出
    router_outputs = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # 不同模型的 router 输出格式不同，这里做通用处理
            if isinstance(output, tuple):
                router_outputs[layer_idx] = output
            else:
                router_outputs[layer_idx] = output
        return hook_fn

    # 找到所有 router/gate 模块并注册 hook
    hooks = []
    for name, module in model.named_modules():
        if any(kw in name.lower() for kw in ["gate", "router"]):
            # 从模块名提取层号
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

    # 对每段文本进行分析
    for text in texts:
        print(f"📝 Input: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        router_outputs.clear()
        with torch.no_grad():
            model(**inputs)

        # 分析 router 输出
        if not router_outputs:
            print("  ⚠️ 未捕获到 router 输出，可能模型结构不匹配\n")
            continue

        # 统计每个 expert 在所有层被选中的总次数
        total_expert_counts = Counter()
        num_tokens = len(tokens)
        num_layers = len(router_outputs)

        for layer_idx in sorted(router_outputs.keys())[:5]:  # 只显示前 5 层
            output = router_outputs[layer_idx]
            # 尝试从 output 中提取 expert 索引
            # 这部分需要根据具体模型调整
            try:
                if hasattr(output, "shape"):
                    logits = output
                elif isinstance(output, tuple) and len(output) >= 2:
                    logits = output[0] if hasattr(output[0], "shape") else output[1]
                else:
                    continue

                if logits.dim() >= 2:
                    # 取 top-k expert indices
                    k = min(8, logits.shape[-1])
                    topk_indices = torch.topk(logits.view(-1, logits.shape[-1]), k=k, dim=-1).indices
                    expert_counts = Counter(topk_indices.cpu().numpy().flatten().tolist())

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

        # 总体统计
        if total_expert_counts:
            print(f"\n  📊 Expert 使用分布 (前 5 层汇总):")
            all_counts = list(total_expert_counts.values())
            print(f"    使用的 Expert 总数: {len(total_expert_counts)}")
            print(f"    最常用 Expert: {total_expert_counts.most_common(3)}")
            print(f"    最少用 Expert: {total_expert_counts.most_common()[-3:]}")
            if len(all_counts) > 1:
                mean_count = sum(all_counts) / len(all_counts)
                std_count = (sum((c - mean_count) ** 2 for c in all_counts) / len(all_counts)) ** 0.5
                print(f"    负载均衡 (CV): {std_count / mean_count:.3f} (越低越均衡)")
        print()

    # 清理 hooks
    for hook in hooks:
        hook.remove()


# ============================================================
# Part 4: Dense vs MoE config 对比
# ============================================================

def compare_dense_vs_moe(dense_model: str, moe_model: str):
    """对比 Dense 和 MoE 模型的 config 差异。"""
    print("\n" + "=" * 70)
    print(f"Dense vs MoE 配置对比")
    print(f"  Dense: {dense_model}")
    print(f"  MoE:   {moe_model}")
    print("=" * 70)

    dense_config = AutoConfig.from_pretrained(dense_model, trust_remote_code=True).to_dict()
    moe_config = AutoConfig.from_pretrained(moe_model, trust_remote_code=True).to_dict()

    # 找出差异
    all_keys = sorted(set(list(dense_config.keys()) + list(moe_config.keys())))
    print(f"\n{'Key':<40} {'Dense':<20} {'MoE':<20}")
    print("─" * 80)

    for key in all_keys:
        dense_val = dense_config.get(key, "—")
        moe_val = moe_config.get(key, "—")
        if dense_val != moe_val:
            # 高亮 MoE 特有的字段
            marker = "🆕" if dense_val == "—" else "📝"
            print(f"{marker} {key:<38} {str(dense_val):<20} {str(moe_val):<20}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Model Explorer")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B",
                        help="HuggingFace model name or path")
    parser.add_argument("--mode", type=str, default="structure",
                        choices=["structure", "architecture", "routing", "compare"],
                        help="分析模式")
    parser.add_argument("--load-weights", action="store_true",
                        help="是否加载完整权重（routing 模式需要）")
    parser.add_argument("--dense-model", type=str, default="Qwen/Qwen3-4B",
                        help="用于对比的 Dense 模型（compare 模式）")

    args = parser.parse_args()

    if args.mode == "structure":
        analyze_structure(args.model)

    elif args.mode == "architecture":
        analyze_structure(args.model)
        print_model_architecture(args.model)

    elif args.mode == "routing":
        if not args.load_weights:
            print("⚠️  routing 模式建议加上 --load-weights 以加载完整权重")
            print("   不加载权重时只展示结构分析\n")
            analyze_structure(args.model)
        else:
            analyze_routing(args.model)

    elif args.mode == "compare":
        compare_dense_vs_moe(args.dense_model, args.model)
```

### 5.3 脚本使用示例

```bash
# 1. 只看结构和参数量（不需要 GPU）
python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode structure

# 2. 打印详细的模型层级架构
python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode architecture

# 3. 对比 Dense 和 MoE 的 config 差异（推荐！直觉上最有收获）
python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode compare --dense-model Qwen/Qwen3-4B

# 4. 分析路由行为（需要 GPU，至少 ~20GB 用 4-bit 量化）
python explore_moe.py --model Qwen/Qwen3-30B-A3B --mode routing --load-weights

# 也可以换成 Mixtral
python explore_moe.py --model mistralai/Mixtral-8x7B-v0.1 --mode structure
python explore_moe.py --model mistralai/Mixtral-8x7B-v0.1 --mode compare --dense-model mistralai/Mistral-7B-v0.1
```

---

## 6. 推荐阅读路线

按照以下顺序阅读，效率最高：

### 第一步：建立基本概念（1-2 小时）

- **HuggingFace Blog: Mixture of Experts Explained**
  https://huggingface.co/blog/moe
  最好的入门材料，涵盖基本概念、训练技巧、推理优化

### 第二步：理解经典 MoE 架构（2-3 小时）

- **Mixtral of Experts** (Jiang et al., 2024)
  https://arxiv.org/abs/2401.04088
  从你最熟悉的 LLaMA 架构出发，Mixtral 只改了 FFN→MoE，最容易建立直觉

### 第三步：深入前沿设计（3-5 小时）

- **DeepSeek-V3 Technical Report** (DeepSeek-AI, 2024)
  https://arxiv.org/abs/2412.19437
  当前最详尽的 MoE 工程实践。重点读 Section 2 (Architecture) 和 Section 3 (Pre-training)

- **Qwen3 Technical Report** (Qwen Team, 2025)
  https://arxiv.org/abs/2505.09388
  你已经熟悉 Qwen Dense，对比 MoE 版本的差异非常直观

### 第四步：架构对比和可视化（1 小时）

- **Sebastian Raschka: LLM Architecture Gallery**
  https://sebastianraschka.com/llm-architecture-gallery/
  横向对比所有主流 LLM（包括 MoE）的架构图和关键参数

- **Sebastian Raschka: A Technical Tour of the DeepSeek Models**
  https://magazine.sebastianraschka.com/p/technical-deepseek
  深度解读 DeepSeek 架构演进

### 可选：训练工程细节

- **DeepSeek: Insights into Scaling Challenges and Reflections on Hardware**
  https://arxiv.org/abs/2505.09343
  如果你关心 MoE 训练的通信优化、FP8 训练等工程细节

---

## 7. 一句话总结

**MoE 不是一种新架构，而是对 Dense Transformer 的一种稀疏化策略**：把每层的 FFN 复制成 N 份（experts），每个 token 只用其中 K 份。你掌握的所有 Dense 预训练知识（数据处理、学习率调度、Loss 设计、并行策略等）几乎全部适用，只需要额外关注三件事：Router 怎么设计、Load Balance 怎么做、以及 Expert Parallelism 怎么搞。

从研究的角度，MoE 最核心的开放问题是：**Expert 到底学到了什么？** 是不是真的 specialize 了？怎么控制它们的 specialization？这些都是活跃的研究方向。
