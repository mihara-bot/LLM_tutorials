# 从Goldberg论文到LLM训练：浮点数完全指南

## 目录

1. [第一章：浮点数的本质——为什么不能精确表示0.1](#第一章)
2. [第二章：IEEE 754标准——位级别的拆解](#第二章)
3. [第三章：舍入误差——为什么你的计算会"漂移"](#第三章)
4. [第四章：Float16——半精度的诱惑与陷阱](#第四章)
5. [第五章：BFloat16——Google为什么要重新发明轮子](#第五章)
6. [第六章：INT8/INT4量化——推理时的极致压缩](#第六章)
7. [第七章：Mixed Precision Training——工程实践](#第七章)
8. [第八章：完整实例——手搓一个混合精度训练循环](#第八章)
9. [第九章：LLM Pre-training/SFT的显存占用计算](#第九章)

---

## 第一章：浮点数的本质——为什么不能精确表示0.1 {#第一章}

### 1.1 从科学计数法说起

你在日常生活中已经见过浮点数了——科学计数法：

```
光速 = 2.998 × 10^8 米/秒
```

这里有三个要素：
- **符号**：正
- **有效数字（significand/mantissa）**：2.998，共4位
- **指数（exponent）**：8
- **基数（base）**：10

Goldberg论文用一个通用公式来描述所有浮点系统：

```
± d₀.d₁d₂...d_{p-1} × β^e
```

其中：
- β = 基数（base），计算机中通常是2
- p = 精度（precision），有多少位有效数字
- e = 指数，范围在 [e_min, e_max] 之间
- 每个数字 d_i 满足 0 ≤ d_i < β

### 1.2 为什么二进制不能精确表示0.1

在十进制中，0.1是"精确"的。但在二进制中：

```
0.1（十进制）= 0.0001100110011001100110011... （二进制，无限循环）
```

就像 1/3 在十进制中是 0.333... 永远写不完一样，0.1在二进制中也永远写不完。

论文原文（Section 1.1）指出：当β=2时，十进制的0.1大约等于：
```
1.10011001100110011001101 × 2^(-4)
```
注意末尾的"101"——这是因为只有有限位数，后面的被截断（或舍入）了。

**关键洞察**：这不是bug，而是有限位数表示无限多实数的必然代价。论文开篇就说"Squeezing infinitely many real numbers into a finite number of bits requires an approximate representation."

### 1.3 浮点数的密度不均匀

这一点对理解LLM训练至关重要。看论文的Figure 1：当β=2, p=3, e_min=-1, e_max=2时，所有可表示的归一化浮点数是：

```
数轴: 0    1    2    3    4    5    6    7
      |    |++++|  + |  + |    +   |    +   |    +    |
```

所有规格化浮点数的 significand 形式是 1.b₁b₂，其中 b₁, b₂ $\in \{0,1\}$。
所以 significand 只有 4 种可能值：1.00, 1.01, 1.10, 1.11。

当 e = -1 时，能表示的数字是：

1.00 × 0.5 = 0.5
1.01 × 0.5 = 0.625
1.10 × 0.5 = 0.75
1.11 × 0.5 = 0.875

当 e = 0 时，能表示的数字是：

1.00 × 1 = 1.0
1.01 × 1 = 1.25
1.10 × 1 = 1.5
1.11 × 1 = 1.75

当 e = 1 时，能表示的数字是：

1.00 × 2 = 2.0
1.01 × 2 = 2.5
1.10 × 2 = 3.0
1.11 × 2 = 3.5

当 e = 2 时，能表示的数字是：

1.00 × 4 = 4.0
1.01 × 4 = 5.0
1.10 × 4 = 6.0
1.11 × 4 = 7.0

在[1,2)之间，数是均匀分布的，间距为 2^(1-p) = 2^(-2) = 0.25
在[2,4)之间，间距翻倍，变成 2^(2-p) = 2^(-1) = 0.5
在[4,8)之间，间距再翻倍，变成 2^(3-p) = 2^0 = 1

**规律**：接近零的地方浮点数最密，远离零的地方浮点数最稀疏。间距每隔2的幂次就翻倍。

浮点数系统保证的是相对精度而非绝对精度。在任何一个指数区间 $[β^e, β^(e+1)]$ 内，相邻浮点数的间距是固定的，但跨到下一个指数区间，间距就乘以 β。

对一个指数为 e 的浮点数，有:

- 1 ulp (unit in the last place, 最后一位上的单位) = $\beta^{e-(p-1)}$.

- 相邻浮点数的最大舍入误差是 1/2 ulp，因为相邻两个浮点数的 significand 差恰好是 1 ulp，任何落在两个浮点数之间的实数，到两端的距离最多是这个间距的一半。

**与训练模型的关联**:

- 权重初始化（如 Xavier/He 初始化）把初始权重设在均值0、方差接近1的分布里。
如果权重初始化为很大的数（比如10⁶），那 1 ulp 就很大，梯度更新的微小变化会被直接吞掉。
如果权重太小（比如10⁻²⁰），又可能落入 denormalized number 的范围，精度急剧下降。

- Batch Normalization 把每层的激活值强制拉回到均值0、方差1附近。这不仅帮助训练稳定性，还间接保证了数值留在浮点数精度最好的区间。

- 梯度裁剪（gradient clipping） 防止梯度爆炸到很大的数，那样会丧失精度。

- 混合精度训练（用 float16）时这个问题更突出。float16 只有 p = 11（10位显式 + 1位隐藏位），ε 很大。如果数值偏离1太远，精度损失会非常严重，训练直接崩溃。这就是为什么混合精度训练需要 loss scaling: 把 loss 乘一个大数，让梯度值落在 float16 能表示的范围内。

本质上，现代深度学习的许多工程技巧，都是在和浮点数的非均匀精度做斗争。

### 1.4 归一化（Normalization）和隐藏位（Hidden Bit）

论文指出，浮点数表示不是唯一的。比如 0.1 可以表示为：
```
1.00 × 10^(-1)     （归一化）
0.01 × 10^(1)      （非归一化）
0.10 × 10^(0)      （非归一化）
```

为了保证唯一性，要求**前导数字不为零**（d₀ ≠ 0）。这叫做归一化（normalized）表示。

在二进制中，如果 d₀ ≠ 0，那 d₀ 一定等于 1（因为二进制只有0和1）。既然它总是1，就没必要存储它，这样就白赚一位精度！这就是**隐藏位（hidden bit）**。

```
实际存储：[符号1位][指数8位][尾数23位]
实际精度：1（隐藏）+ 23（存储）= 24位有效位
```

---

## 第二章：IEEE 754标准——位级别的拆解 {#第二章}

### 2.1 Float32的位布局

IEEE 754单精度（float32）一共32位：

```
 31  30........23  22...................0
 [S] [EEEEEEEE] [MMMMMMMMMMMMMMMMMMMMMMM]
  │       │                │
  │       │                └─ 尾数（mantissa/fraction）: 23位
  │       └─ 指数（exponent）: 8位，偏移量127
  └─ 符号（sign）: 0=正, 1=负
```

**实际值的计算公式**：
```
value = (-1)^S × 1.M × 2^(E - 127)
```

手动例子：把 float32 的 -6.5 编码为二进制

```
步骤1：符号位 → 负数，S = 1
步骤2：6.5 = 110.1（二进制）= 1.101 × 2^2
步骤3：指数 = 2 + 127 = 129 = 10000001
步骤4：尾数 = 101后面补零 = 10100000000000000000000

完整编码：1 10000001 10100000000000000000000
十六进制：0xC0D00000
```

### 2.2 特殊值——论文Section 2.2

IEEE 754用特殊的指数来编码特殊值：

| 指数（偏移后） | 尾数 | 表示的值 |
|---|---|---|
| 全0 (e_min - 1) | 全0 | ±0 |
| 全0 (e_min - 1) | 非0 | 非归一化数（denormalized） |
| 正常范围 | 任意 | ±1.M × 2^(E-bias) |
| 全1 (e_max + 1) | 全0 | ±∞ |
| 全1 (e_max + 1) | 非0 | NaN |

**NaN（Not a Number）**：论文用零除零的例子来说明为什么需要NaN。在LLM训练中，NaN的出现通常意味着训练崩溃——可能是梯度爆炸导致overflow变成Infinity，然后Infinity参与运算产生NaN。

**非归一化数（denormals）**：论文Section 2.2.4详细讨论了这个概念。普通归一化数的最小值是 1.0 × 2^(e_min)。在这个值和0之间有一个"空隙"。非归一化数填充了这个空隙，形式为 0.M × 2^(e_min)。论文称之为"gradual underflow（渐进下溢）"。

没有非归一化数时，一个关键性质会被违反：
```
x ≠ y  ⇔  x - y ≠ 0   （这个性质在flush-to-zero时会失败！）
```

论文举例：β=10, p=3, e_min=-98，取 x=6.87×10^(-97), y=6.81×10^(-97)。
x-y = 0.06×10^(-97) = 6.0×10^(-99)，这比最小归一化数 1.00×10^(-98) 还小，
如果flush to zero，x-y=0，但 x≠y。有了非归一化数，可以表示为 0.6×10^(-98)。

### 2.3 指数偏移（Bias）的设计

论文Section 2.1.3解释了为什么用偏移表示而不是二进制补码。好处是：
**非负浮点数可以直接当整数来比较大小**。

```
float32: bias = 127, 指数范围 [-126, 127]
float64: bias = 1023, 指数范围 [-1022, 1023]
```

为什么 |e_min| < e_max？论文解释：这样最小正数的倒数就不会溢出。
```
最小正归一化数 = 2^(-126) ≈ 1.18 × 10^(-38)
它的倒数 = 2^126 ≈ 8.51 × 10^37 < 最大值 ≈ 3.4 × 10^38  ✓
```

### 2.4 Machine Epsilon——衡量精度的标尺

论文Section 1.2定义了machine epsilon：

```
ε = (β/2) × β^(-p)
```

对float32: ε = 2^(-24) ≈ 5.96 × 10^(-8)

这意味着：**任何一次基本浮点运算（加减乘除），结果的相对误差最多是 ε**。

更精确地说，如果 a⊕b 表示浮点加法的结果，那么：
```
a ⊕ b = (a + b)(1 + δ),  |δ| ≤ ε
```

这个公式（对应论文的eq. 19-21）是理解一切误差分析的基础。

---

## 第三章：舍入误差——为什么你的计算会"漂移" {#第三章}

### 3.1 两种度量方式：ULP vs 相对误差

论文Section 1.2介绍了两种度量误差的方式：

**ULP（Unit in the Last Place）**：最后一位的单位。如果精确值是3.14159，float表示为3.14，误差是0.159 ulps。

**相对误差（Relative Error）**：误差/真值。上面的例子中是 0.00159/3.14159 ≈ 0.0005。

论文的关键观察（eq. 2）：1/2 ulp对应的相对误差会"摆动"（wobble）β倍。
```
(1/2)β^(-p) ≤ (1/2 ulp的相对误差) ≤ (β/2)β^(-p)
```

当β=2时摆动因子最小（只有2倍），这是IEEE 854偏好β=2的原因之一。

### 3.2 Guard Digit——一个bit价值百万

论文Section 1.3用一个惊人的例子展示了没有guard digit的后果：

```
计算 10.1 - 9.93，精度p=3

没有guard digit:
  x = 1.01 × 10^1
  y = 0.99 × 10^1    （9.93对齐后只保留p位，变成9.9）
  x - y = 0.02 × 10^1 = 0.2

正确答案 = 0.17
误差 = 30 ulps，每一位都是错的！
```

论文Theorem 1证明：没有guard digit时，减法的相对误差可以大到 β-1。当β=2时，误差可以和结果一样大！

加一个guard digit（多保留一位）后：
```
  x = 1.010 × 10^1
  y = 0.993 × 10^1
  x - y = 0.017 × 10^1 = 0.17  ✓ 精确！
```

论文Theorem 2证明：有一个guard digit，减法的相对误差最多 2ε。代价？对54位加法器只增加不到2%的成本。

### 3.3 灾难性抵消（Catastrophic Cancellation）

这是论文Section 1.4的核心内容，也是LLM训练中最需要警惕的问题。

**两种抵消**：
- **灾难性抵消（catastrophic）**：操作数本身有舍入误差，相减后误差被放大
- **良性抵消（benign）**：操作数是精确的，减法不会放大误差

经典例子——二次方程：
```
r = (-b ± √(b²-4ac)) / 2a
```

当 b² ≈ 4ac 时，√(b²-4ac) ≈ |b|，那么 -b + √(b²-4ac) 中的加法就是灾难性抵消。

论文给出的解法：用恒等变换消除抵消：
```
r₁ = 2c / (-b - √(b²-4ac))
r₂ = 2c / (-b + √(b²-4ac))
```
根据b的符号选择不会产生抵消的公式。

**对LLM训练的影响**：
- Layer Normalization中计算方差：var = E[x²] - (E[x])²，当方差很小时，两个接近的量相减会产生灾难性抵消。这就是为什么实际实现用 var = E[(x-mean)²] 的形式。
- Softmax中 exp(x_i) / Σexp(x_j)，如果不减去max，大的指数会溢出。
- 梯度差分：(loss(θ+δ) - loss(θ-δ)) / 2δ 中，当δ很小时。

### 3.4 误差累积——长求和的问题

论文Section 4.3和Theorem 8讨论了求和误差。

朴素求和 s = x₁ + x₂ + ... + xₙ 的误差分析：
```
s_n = Σx_j(1 + δ_j) + O(nε²)Σ|x_j|
```
其中 |δ_j| < (n-j)ε。第一个被加数的误差最大，达到 nε。

**Kahan求和算法**（论文Theorem 8）：
```python
s = x[0]
c = 0          # 补偿值
for j in range(1, n):
    y = x[j] - c       # 先减去上次的补偿
    t = s + y           # 加到总和
    c = (t - s) - y     # 计算丢失的低位
    s = t
```

误差降到 |δ_j| ≤ 2ε，不随n增长！

**核心思想**：c 变量捕获了每次加法中丢失的低位bit，在下一次迭代中补偿回来。

论文警告：如果优化器把 C = (T-S)-Y 简化为 C = ((S+Y)-S)-Y = 0，算法就被毁了。这就是为什么编译器不能随意重排浮点运算的括号。

---

## 第四章：Float16——半精度的诱惑与陷阱 {#第四章}

### 4.1 Float16的位布局

```
 15  14...10  9.........0
 [S] [EEEEE] [MMMMMMMMMM]
  │     │         │
  │     │         └─ 尾数: 10位（有效精度11位，含隐藏位）
  │     └─ 指数: 5位，偏移量15
  └─ 符号
```

| 属性 | Float16 | Float32 |
|---|---|---|
| 总位数 | 16 | 32 |
| 有效精度 | 11位 (≈3.3位十进制) | 24位 (≈7.2位十进制) |
| 指数范围 | [-14, 15] | [-126, 127] |
| Machine epsilon | 2^(-11) ≈ 4.88×10^(-4) | 2^(-24) ≈ 5.96×10^(-8) |
| 最大值 | 65504 | ≈3.4×10^38 |
| 最小正归一化数 | 2^(-14) ≈ 6.1×10^(-5) | 2^(-126) ≈ 1.18×10^(-38) |

### 4.2 Float16的三大问题

**问题1：溢出（Overflow）太容易发生**

最大值只有65504。在LLM训练中，梯度乘以学习率后的中间值很容易超过这个范围。
```
假设某个激活值 = 300，它的平方 = 90000 > 65504 → 溢出为 Inf
```

**问题2：下溢（Underflow）也很容易**

最小正归一化数是 6.1×10^(-5)。很多梯度值比这小得多。
```
一个7B模型的典型梯度大小可能在 10^(-6) ~ 10^(-8)
这些在float16中全部下溢为0
```

**问题3：精度不够**

只有3.3位十进制精度意味着：
```
float16(1.0 + 0.001) = float16(1.0) = 1.0
因为 0.001 < ε(float16) = 0.000488...
这个加法完全没有效果！
```

这在权重更新中是致命的：
```
weight_new = weight_old + learning_rate × gradient
如果 learning_rate × gradient << weight_old × ε
那么 weight_new == weight_old，训练停滞！
```

### 4.3 Loss Scaling——Float16训练的必需品

Loss Scaling的核心思想是利用论文中讨论的指数特性：

```
原始梯度可能很小:          g = 1.5 × 10^(-7)  （在float16中下溢为0）
放大loss（比如×1024）:     g' = g × 1024 = 1.536 × 10^(-4)  （在float16范围内）
反向传播后再缩回:          g = g' / 1024 = 1.5 × 10^(-7)   （在float32中恢复精度）
```

这利用了一个关键事实：链式法则中的梯度与loss成正比，放大loss等价于放大所有梯度。

动态Loss Scaling的策略：
1. 初始 scale = 很大的数（比如2^15）
2. 如果出现Inf/NaN，说明scale太大 → scale减半，跳过这步更新
3. 如果连续N步没问题 → scale翻倍（尝试更激进的放大）

---

## 第五章：BFloat16——Google为什么要重新发明轮子 {#第五章}

### 5.1 BFloat16的设计哲学

Google Brain团队在设计TPU时面临一个选择：Float16精度不够且范围太小，Float32太浪费。他们的解决方案很简单：

**保留Float32的指数位数（8位），只砍尾数位数**。

```
Float32:  [1位符号] [8位指数] [23位尾数]  = 32位
BFloat16: [1位符号] [8位指数] [7位尾数]   = 16位
                     ^^^^^^^^
                     和Float32一样！
```

### 5.2 BFloat16 vs Float16 对比

| 属性 | BFloat16 | Float16 |
|---|---|---|
| 总位数 | 16 | 16 |
| 有效精度 | 8位 (≈2.4位十进制) | 11位 (≈3.3位十进制) |
| 指数范围 | [-126, 127] | [-14, 15] |
| Machine epsilon | 2^(-8) ≈ 0.00391 | 2^(-11) ≈ 0.000488 |
| 最大值 | ≈3.39×10^38 | 65504 |
| 最小正归一化数 | ≈1.18×10^(-38) | ≈6.1×10^(-5) |
| 需要Loss Scaling? | **不需要** | 需要 |

**关键trade-off**：BFloat16用更差的精度换取了Float32级别的指数范围。

### 5.3 为什么BFloat16更适合训练

**理由1：不需要Loss Scaling**

由于指数范围与Float32完全相同，梯度不会下溢，不会溢出。训练代码更简单、更稳定。

**理由2：与Float32的转换极其简单**

BFloat16就是Float32截掉低16位（尾数的低16位）：
```python
# Float32 → BFloat16（概念上）
bf16_bits = float32_bits >> 16   # 直接右移16位！

# 或者更精确一点，加上舍入：
bf16_bits = (float32_bits + 0x8000) >> 16
```

这意味着硬件实现几乎零成本。

**理由3：精度虽低但对训练够用**

ε ≈ 0.004，意味着大约0.4%的相对误差。虽然听起来很大，但：
- 梯度下降本身就是随机的（SGD），每步都有噪声
- 权重更新通常在Float32中进行（master copy）
- 神经网络对个别权重的扰动非常鲁棒

### 5.4 用论文的框架理解BFloat16

把BFloat16代入论文的参数：
```
β = 2
p = 8  （7位尾数 + 1位隐藏位）
e_min = -126
e_max = 127
```

Machine epsilon = (2/2) × 2^(-8) = 2^(-8) ≈ 0.00391

用论文Theorem 2：有guard digit的减法，相对误差最多 2ε ≈ 0.0078 ≈ 0.78%。

用论文eq.(3)：如果某次运算的相对误差是 nε，"受污染的数字"位数约为 log₂(n)。
对于n=1000次累加：log₂(1000) ≈ 10，但BFloat16只有8位有效位！
这就是为什么梯度累加必须在Float32中进行。

最佳实践是：如果硬件支持就使用 BF16（A100, H100, MI300），只有在被限制在老 GPU（V100, MI200）上时才谨慎使用 FP16。

当前前沿：FP8 训练
随着大语言模型的增长，混合精度训练已经成为加速训练的关键策略，从 BF16 进一步降低到 FP8 格式。 
FP8 有两种主要变体：

E4M3：4位指数 + 3位尾数，精度更高，范围更小，用于前向传播
E5M2：5位指数 + 2位尾数，精度更低，范围更大，用于反向传播（梯度）

DeepL 在实际生产中使用 NVIDIA Transformer Engine 将训练代码从 BF16 迁移到 FP8，前向传播使用 E4M3，反向传播使用 E5M2。 DeepL这种分工正好呼应了 Goldberg 论文的思想——根据不同计算阶段的需求选择不同的精度/范围权衡。
但 FP8 训练还不完全成熟。研究发现目前可用的 FP8 训练方法还不够鲁棒，无法作为高精度方法的经济替代品。FP16 曾被发现不如 BF16 稳定，而 FP8 比 FP16 位数更少，这引发了对其能否成为成本有效选项的担忧。

---

## 第六章：INT8/INT4量化——推理时的极致压缩 {#第六章}

### 6.1 从浮点到定点

INT8和INT4不是浮点格式，而是**均匀量化的整数**。核心思想：

```
量化：x_int = round(x_float / scale)
反量化：x_float ≈ x_int × scale
```

其中 scale = max(|x|) / (2^(n-1) - 1)

**对比论文中的浮点数**：
- 浮点数：间距不均匀，接近0处密（论文Figure 1）
- 整数量化：间距完全均匀

```
Float: |...........|....|...|..|.|.|..|...|....|...........|
INT8:  |..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|
       ←─── 负 ────────── 0 ────────── 正 ───→
```

### 6.2 对称量化 vs 非对称量化

**对称量化（Symmetric）**：
```
scale = max(|x|) / 127
zero_point = 0
x_quant = round(x / scale)
范围：[-127, 127]
```

**非对称量化（Asymmetric）**：
```
scale = (max(x) - min(x)) / 255
zero_point = round(-min(x) / scale)
x_quant = round(x / scale) + zero_point
范围：[0, 255]
```

对称量化更简单，矩阵乘法时不需要处理zero_point的额外项。LLM量化中通常使用对称量化。

### 6.3 Per-tensor vs Per-channel vs Per-group量化

量化的粒度决定了精度：

```
Per-tensor: 整个权重矩阵共享一个scale       → 精度最差，速度最快
Per-channel: 每个输出通道一个scale          → 精度较好
Per-group:  每g个元素共享一个scale（如g=128）→ 精度很好，是LLM量化的主流
```

### 6.4 INT4量化的挑战

INT4只有16个值（-8到7或0到15）。用论文的框架理解：
```
等效"精度" ≈ 4位
等效"machine epsilon" ≈ 2^(-4) = 0.0625 = 6.25%
```

每次运算引入6.25%的误差是不可接受的。所以INT4量化需要特殊技巧：
- **GPTQ**：逐列量化，每量化一列就补偿其他列的误差
- **AWQ**：保护重要权重（通过分析激活值来确定哪些权重重要）
- **Group quantization**：g=32或g=128，大幅减少量化误差

### 6.5 为什么量化只用于推理不用于训练

论文中的误差累积分析直接回答了这个问题。

训练中，误差通过成千上万步累积：
```
朴素求和误差 ≈ n × ε × Σ|x_j|   （论文eq. 31）
```

对INT4，ε ≈ 0.0625，经过1000步累积，误差 ≈ 62.5 × Σ|x_j|，完全不可接受。
对BFloat16，ε ≈ 0.004，经过1000步，误差 ≈ 4 × Σ|x_j|，配合Float32累加器可以控制。

推理只需要一次前向传播（~100次矩阵乘法），误差累积有限，INT4/INT8够用。

---

## 第七章：Mixed Precision Training——工程实践 {#第七章}

### 7.1 混合精度训练的三个核心组件

```
┌─────────────────────────────────────────────┐
│            Mixed Precision Training          │
│                                              │
│   ┌──────────────┐     ┌──────────────┐     │
│   │ Master Copy  │────→│  Half Copy   │     │
│   │  (Float32)   │     │ (BF16/FP16)  │     │
│   │   权重w_32   │     │   权重w_16   │     │
│   └──────┬───────┘     └──────┬───────┘     │
│          │                    │              │
│          │  ③更新(FP32)       │ ①前向(FP16)  │
│          │                    ↓              │
│          │              ┌──────────┐         │
│          │              │ Loss(FP16)│         │
│          │              └─────┬────┘         │
│          │                    │ ②反向(FP16)  │
│          │                    ↓              │
│   ┌──────┴───────┐     ┌──────────────┐     │
│   │  梯度累加    │←────│ 梯度(FP16)   │     │
│   │  (Float32)   │     │→转为FP32后累加│     │
│   └──────────────┘     └──────────────┘     │
└─────────────────────────────────────────────┘
```

**①前向传播（BF16/FP16）**：矩阵乘法在半精度中完成，速度快2-4倍

**②反向传播（BF16/FP16）**：梯度计算在半精度中完成

**③权重更新（FP32）**：
```python
# 不能这样做（FP16中weight可能不变）：
weight_fp16 += lr * grad_fp16    # 如果 lr*grad << weight*ε，更新无效！

# 正确做法：
grad_fp32 = grad_fp16.float()              # 转换到FP32
weight_fp32 += lr * grad_fp32              # FP32中更新
weight_fp16 = weight_fp32.half()           # 转回FP16用于下一步
```

### 7.2 为什么必须在FP32中更新权重

用论文的分析框架：
```
weight = 10.0   （一个典型的权重值）
lr × gradient = 0.0001   （一个典型的更新量）

在BFloat16中:
  ε = 2^(-8) ≈ 0.00391
  10.0的ulp = 10.0 × ε ≈ 0.039
  更新量0.0001 << 0.039
  → 更新被舍入掉，weight不变！

在Float32中:
  ε = 2^(-24) ≈ 5.96×10^(-8)
  10.0的ulp = 10.0 × ε ≈ 5.96×10^(-7)
  更新量0.0001 >> 5.96×10^(-7)
  → 更新有效 ✓
```

### 7.3 GPU Tensor Core中的精度

现代GPU的Tensor Core在硬件层面实现了混合精度：

```
NVIDIA A100 Tensor Core:
  输入: FP16 × FP16 矩阵乘法
  累加器: FP32        ← 这就是论文中guard digit思想的延伸
  输出: 可以是FP16或FP32

这意味着 C = A × B 中:
  每个乘法 a_ij × b_jk 在FP16中进行
  但所有乘积的求和在FP32中进行
  避免了论文Section 4.3讨论的求和误差累积问题
```

### 7.4 各格式的内存和速度对比

对于一个7B参数的LLM：

| 格式 | 每个参数 | 模型大小 | 训练时显存(含优化器) |
|---|---|---|---|
| Float32 | 4字节 | 28 GB | ~112 GB |
| BFloat16 (混合精度) | 2+4字节 | 14+28 GB | ~84 GB |
| Float16 (混合精度) | 2+4字节 | 14+28 GB | ~84 GB |
| INT8 (推理) | 1字节 | 7 GB | N/A |
| INT4 (推理) | 0.5字节 | 3.5 GB | N/A |

---

## 第八章：完整实例——手搓一个混合精度训练循环 {#第八章}


1. **位级别的浮点数拆解**——亲眼看到符号位、指数、尾数
2. **各种精度格式的对比实验**——Float32/Float16/BFloat16的误差对比
3. **灾难性抵消的现场演示**——重现论文中的经典案例
4. **Kahan求和 vs 朴素求和**——看到误差差异
5. **量化模拟实验**——INT8/INT4量化的精度损失
6. **混合精度权重更新模拟**——为什么FP32 master copy必不可少
7. **Loss Scaling实验**——为什么Float16需要它

每个实验都有详细的中文注释和输出解读。

---

## 第九章：LLM Pre-training/SFT的显存占用计算 {#第九章}

前面我们一直在讲"数值能不能算对"。这一章回答另一个同样现实的问题：

**这份训练任务，到底能不能塞进一张GPU里？**

很多人一提显存，只会说"这是个7B模型，所以大概要14GB/28GB/80GB"。这只说对了一小部分。训练时真正占显存的，至少有四类东西：

- **模型参数（weights）**
- **梯度（gradients）**
- **优化器状态（optimizer states）**
- **激活值（activations）**

而且对 Pre-training、全量 SFT、LoRA SFT、QLoRA SFT，这四块的比例完全不同。

这一章的目标不是给你一个死记硬背的数字，而是给你一套**可复用的估算公式**。你拿到模型配置后，应该能自己在纸上算出：

- 单卡大概要多少显存
- 为什么会OOM
- 降 `micro_batch_size`、降 `seq_len`、开 checkpointing、换 LoRA，到底哪一步最有效

如果你想直接把这些公式跑成命令行工具，可以配合使用脚本 [vram_estimator.py](/Users/xinlin.zhuang/Codes/LLM_tutorials/float_tutorials/vram_estimator.py)。

### 9.1 先把训练显存拆成四块

一个训练任务的峰值显存，可以先写成：

```text
总显存 ≈ 模型状态 + 激活显存 + 临时workspace/通信buffer + 碎片/安全余量
```

其中：

- **模型状态（model states）** = 参数 + 梯度 + 优化器状态
- **激活显存（activations）** = forward过程中为了backward而保存的中间结果
- **workspace / communication buffer** = CUDA kernel临时缓冲、all-gather/reduce-scatter buffer、FlashAttention工作区等
- **安全余量** = PyTorch缓存分配器碎片、算子峰值抖动，工程上通常再留 10% 到 20%

一个非常重要的工程事实：

**决定峰值显存的往往不是 `global_batch_size`，而是 `micro_batch_size`。**

因为梯度累加只是把多个 micro-batch 的梯度在时间上串起来，单次 forward/backward 的峰值显存主要还是由：

```text
micro_batch_size、seq_len、hidden_size、num_layers、precision
```

决定。

另一个常见误区是把**训练激活**和**推理KV cache**混为一谈。这里我们讲的是训练显存，所以：

- **训练时重点看 activations**
- **推理时重点看 KV cache**

它们不是一回事。

### 9.2 第一步：先算总参数量 P_all

如果你已经知道模型是"7B / 13B / 70B"，那可以直接把这个数字当作总参数量 `P_all`。

但如果你手里拿到的是配置文件，那么也可以从结构手算出来。对一个 Llama-like 模型，定义：

```text
H = hidden_size
I = intermediate_size
L = num_hidden_layers
V = vocab_size
A = num_attention_heads
K = num_key_value_heads
H_kv = H × K / A
```

其中 `H_kv` 是 K/V 投影的总维度。例如 GQA 里 `num_key_value_heads < num_attention_heads`，所以 K/V 权重矩阵会比 Q/O 更小。

对每一层 Transformer block，可以近似写成：

```text
Attention参数 ≈ H×H   (q_proj)
             + H×H_kv (k_proj)
             + H×H_kv (v_proj)
             + H×H   (o_proj)
             = 2H² + 2H×H_kv

MLP参数（SwiGLU）≈ H×I + H×I + I×H = 3HI

Norm参数 ≈ 2H
```

所以一层的总参数近似是：

```text
P_layer ≈ 2H² + 2H×H_kv + 3HI + 2H
```

整模型再加上 embedding 和 lm_head：

```text
P_all ≈ L×P_layer + V×H + [V×H if tie_word_embeddings=false else 0] + H
```

最后那个 `+H` 对应最终的 RMSNorm，通常很小，可以忽略不计。

#### 手算Llama-480M

设一个Llama-380M的配置如下：

```text
H = 1024
I = 3584
L = 16
V = 128256
A = 8
K = 2
H_kv = 1024 × 2 / 8 = 256
tie_word_embeddings = false
```

先算每层：

```text
Attention = 2×1024² + 2×1024×256
          = 2,621,440

MLP       = 3×1024×3584
          = 11,010,048

Norm      = 2×1024
          = 2,048

P_layer   = 13,633,536
```

再算全模型：

```text
Transformer blocks = 16 × 13,633,536 = 218,136,576
Embedding          = 128,256 × 1024  = 131,334,144
LM head            = 128,256 × 1024  = 131,334,144
Final norm         = 1,024

P_all = 218,136,576 + 131,334,144 + 131,334,144 + 1,024
      = 480,805,888 ≈ 480M
```

这和配置名里的 `480m` 是对得上的。

### 9.3 第二步：区分总参数 P_all 和可训练参数 P_train

这一点决定了你是在算 **Pre-training / 全量SFT**，还是在算 **LoRA/QLoRA**。

定义两个量：

- `P_all`：模型总参数量
- `P_train`：真正需要更新的参数量

于是：

- **Pre-training**：`P_train = P_all`
- **全量 SFT**：`P_train = P_all`
- **LoRA SFT**：`P_train = P_lora << P_all`
- **QLoRA SFT**：`P_train = P_lora << P_all`，但底座参数还是要加载，只是不训练

LoRA 的 trainable params 怎么算？如果对一个线性层 `W ∈ R^(d_out × d_in)` 注入秩为 `r` 的 LoRA，那么新增参数量是：

```text
P_lora_for_W = r×d_in + r×d_out = r(d_in + d_out)
```

比如你给某层的 `q_proj`（`H × H`）做 rank-8 的 LoRA，那么新增参数量就是：

```text
8(H + H) = 16H
```

把所有被插 LoRA 的线性层加起来，就是总的 `P_lora`。

### 9.4 第三步：算模型状态显存

模型状态显存是最容易写成公式的一部分：

```text
M_model_states
= 参数显存 + 梯度显存 + 优化器显存
= P_frozen × b_frozen + P_train × (b_weight + b_grad + b_opt [+ b_master])
```

这里的 `b_*` 单位都是"字节/参数"。

常见训练配置下，每个**可训练参数**大概对应下面这些字节数：

| 训练配置 | 每个trainable param的显存 |
|---|---|
| FP32 + AdamW | 4（权重） + 4（梯度） + 8（m,v） = **16 bytes** |
| BF16/FP16 + AdamW（无FP32 master） | 2（权重） + 2（梯度） + 8（m,v） = **12 bytes** |
| BF16/FP16 + AdamW（有FP32 master） | 2（权重） + 2（梯度） + 8（m,v） + 4（master） = **16 bytes** |

常见**冻结参数**的显存：

| 参数状态 | 每个frozen param的显存 |
|---|---|
| Frozen BF16/FP16 | **2 bytes** |
| Frozen FP32 | **4 bytes** |
| Frozen 4-bit量化权重（QLoRA底座） | 理想值 **0.5 bytes**，加上scale/metadata后工程上常按 **0.55 到 0.6 bytes** 估算 |

如果你用的不是 AdamW，而是 SGD / Momentum / Muon 这类优化器，那么只需要把上式里的 `b_opt` 换成对应的优化器状态字节数即可。AdamW 之所以常被拿来做容量规划，是因为它最常见，也通常比 momentum-only 优化器更占显存。

所以：

- 对 **Pre-training / 全量 SFT**，默认就是 `P_train = P_all`
- 对 **LoRA / QLoRA**，底座是 `P_frozen = P_all`，只有 LoRA adapter 是 `P_train`

#### 继续用 480M 模型举例

如果我们假设使用 **BF16 + AdamW，无FP32 master weight**，那么：

```text
M_model_states = 480,805,888 × 12 bytes
               = 5,769,670,656 bytes
               ≈ 5.37 GiB
```

如果你的框架额外保留 FP32 master weight，那就要改成：

```text
480,805,888 × 16 bytes ≈ 7.17 GiB
```

注意这里我用了 **GiB**：

```text
1 GiB = 1024^3 bytes
1 GB  = 1000^3 bytes
```

显卡宣传页经常写 GB，但 `nvidia-smi` / PyTorch 更接近 GiB 语境，所以手算时最好统一。

### 9.5 第四步：算激活显存

训练时第二大头通常是 activations，而且它比参数显存更"容易突然爆炸"。

粗略地说，激活显存和下面这些量成正比：

```text
层数 L
micro batch B
序列长度 T
隐藏维度 H
激活精度字节数 b
```

一个非常常用的工程近似是：

```text
M_act ≈ c × L × B × T × H × b
```

其中：

- `B` 是 **micro_batch_size**，不是 global batch
- `T` 是单次 forward 的**总序列长度**
- `b` 对 BF16/FP16 通常是 2，对 FP32 是 4
- `c` 是一个和实现细节有关的常数

经验上：

- **开了 activation checkpointing + FlashAttention**：`c` 常常在 **10 到 12** 左右
- **没开 checkpointing**：`c` 可能到 **20 到 30**，甚至更高

为什么 `T` 这么敏感？因为注意力如果显式保存 attention score / probability，还会额外出现一个平方项：

```text
M_attn_scores ≈ L × B × A × T² × b
```

这就是为什么：

- `seq_len` 从 2048 翻到 4096，不是只涨一倍那么简单
- FlashAttention 之类的优化非常值钱

#### 继续用 480M 配置举例

从 [llama3_1_480m_pe.yaml](/Users/xinlin.zhuang/Codes/pretrain_LLM_pipeline/configs/exp/muon/llama3_1_480m_pe.yaml) 和 [tokenize_llama3_1.yaml](/Users/xinlin.zhuang/Codes/pretrain_LLM_pipeline/configs/data/tokenize_llama3_1.yaml) 可以读到：

```text
micro_batch_size B = 8
seq_len T = 2048
L = 16
H = 1024
b = 2 (bf16)
```

如果我们取一个比较常见的经验值 `c = 12`：

```text
M_act ≈ 12 × 16 × 8 × 2048 × 1024 × 2 bytes
      ≈ 6.00 GiB
```

如果没有 FlashAttention，attention score 的平方项单独就大约是：

```text
M_attn_scores ≈ 16 × 8 × 8 × 2048² × 2 bytes
              ≈ 8.00 GiB
```

这两个数字放在一起，你就能立刻理解为什么长上下文训练这么容易OOM。

### 9.6 Pre-training、全量SFT、LoRA、QLoRA到底分别怎么算

现在可以把几种常见场景统一起来了。

#### 场景A：Pre-training

Pre-training 的特点是：

- 全参数训练，所以 `P_train = P_all`
- 通常有 packing，padding 浪费相对少
- global batch 很大，但真正影响单卡峰值的是 `micro_batch_size`

因此一个非常实用的估算公式是：

```text
M_pretrain ≈ P_all × bytes_per_trainable_param
           + c × L × B × T × H × b
           + overhead
```

其中：

- 对 BF16 + AdamW（无master）可先用 `bytes_per_trainable_param = 12`
- `overhead` 工程上先按前两项之和的 10% 到 20% 预留

#### 场景B：全量 SFT

全量 SFT 在显存公式上和 Pre-training **几乎完全一样**：

```text
M_full_sft ≈ P_all × bytes_per_trainable_param
           + c × L × B × T × H × b
           + overhead
```

差别主要不是公式，而是数据形态：

- SFT 样本长度波动更大，padding 可能更严重
- 经常是长 prompt + 短 answer，导致有效监督token比例不高
- 即使只对 answer 区域算 loss，**整段 prompt + answer 还是都要过模型**，激活显存并不会只按 answer 长度算

所以很多人看到"我只训练回答部分"就以为显存会小很多，这通常是错觉。

#### 场景C：LoRA SFT

LoRA 的关键变化是：

- 底座参数冻结，只占权重显存
- 只有 adapter 参数需要梯度和优化器状态

所以：

```text
M_lora ≈ P_all × b_frozen
       + P_lora × bytes_per_trainable_param
       + c × L × B × T × H × b
       + overhead
```

最重要的结论是：

**LoRA 大幅降低的是模型状态显存，但 activation 显存几乎不变。**

也就是说：

- 7B 全量 SFT 跑不动
- 7B LoRA 能跑
- 但如果你把 `seq_len` 从 2K 拉到 8K，LoRA 一样可能OOM

#### 场景D：QLoRA SFT

QLoRA 进一步把**冻结底座**从 BF16 压到 4-bit：

```text
M_qlora ≈ P_all × b_4bit_frozen
        + P_lora × bytes_per_trainable_param
        + c × L × B × T × H × b
        + dequant_workspace
        + overhead
```

其中：

- `b_4bit_frozen` 可以先按 **0.55 到 0.6 bytes/param** 估
- `dequant_workspace` 是量化矩阵乘法过程中的额外临时显存，通常没有 activations 大，但不能完全忽略

QLoRA 节省的仍然主要是**模型状态显存**，不是激活显存。

### 9.7 两个最典型的完整算例

#### 算例1：你仓库里的 480M Pre-training 配置

我们把前面的结果合起来：

- 模型：约 `480.8M` 参数
- 精度：BF16
- 假设优化器按 AdamW 的 `12 bytes/trainable param` 估
- `micro_batch_size = 8`
- `seq_len = 2048`
- 激活经验系数 `c = 12`

先算模型状态：

```text
M_model_states ≈ 480.8M × 12 bytes ≈ 5.37 GiB
```

再算激活：

```text
M_act ≈ 12 × 16 × 8 × 2048 × 1024 × 2 bytes ≈ 6.00 GiB
```

两者相加：

```text
5.37 + 6.00 = 11.37 GiB
```

再给 15% 安全余量：

```text
M_total ≈ 11.37 × 1.15 ≈ 13.1 GiB
```

所以这个配置在一张 **24GB** 卡上，从纯显存预算角度看是比较有希望跑起来的。

如果你：

- 不开 activation checkpointing
- 不用 FlashAttention
- 框架还保留 FP32 master weight

那峰值显存会明显再往上走。

#### 算例2：7B 模型做全量SFT vs LoRA/QLoRA

先看**全量 SFT**。如果是 7B 模型，假设 BF16 + AdamW（无master）：

```text
M_model_states ≈ 7B × 12 bytes = 84 GB ≈ 78.2 GiB
```

注意这还**没算 activations**。

所以：

- 单张 24GB 卡：几乎不可能
- 单张 48GB 卡：通常也不够
- 要么多卡分片（FSDP / ZeRO-3），要么改成 LoRA/QLoRA

再看 **LoRA SFT**。如果 7B 底座冻结成 BF16：

```text
Frozen base ≈ 7B × 2 bytes = 14 GB ≈ 13.0 GiB
```

假设 LoRA 一共只有 `8M` 个可训练参数，按 `12 bytes/trainable param` 算：

```text
LoRA trainable states ≈ 8M × 12 bytes ≈ 0.09 GiB
```

这时总显存的主力就变成：

- 冻结底座权重
- activations

如果进一步做 **QLoRA**，把底座压成 4-bit，按 `0.57 bytes/param` 估：

```text
Frozen base ≈ 7B × 0.57 bytes ≈ 3.7 GiB
```

这就是为什么：

- 全量SFT一个7B模型，单卡24GB通常不现实
- LoRA在24GB上常常可行
- QLoRA会更宽松，但长上下文时 activations 仍然可能成为瓶颈

### 9.8 分布式训练会怎样改这个公式

前面的公式是"单卡视角"。分布式训练的本质，是把某些状态切碎到多张卡上。

#### Data Parallel（DP / DDP）

- 每张卡都保留完整模型、完整梯度、完整优化器状态
- 所以**单卡模型状态显存几乎不降**
- 只是总吞吐变高了

#### ZeRO-1

- 优化器状态分片
- 单卡大致变成：参数 + 梯度 + `优化器状态 / N`

#### ZeRO-2

- 优化器状态和梯度都分片
- 单卡大致变成：参数 + `梯度 / N` + `优化器状态 / N`

#### ZeRO-3 / FSDP Full Shard

- 参数、梯度、优化器状态都分片
- 单卡大致接近：

```text
(参数 + 梯度 + 优化器状态) / N + activations + 通信buffer
```

注意这里说的是**平均占用**。真实峰值还会因为 all-gather / reduce-scatter 出现局部抬高，所以不能天真地直接除以 `N` 就完事。

### 9.9 一个最好用的实战口诀

真正做容量规划时，可以按下面四步走：

1. **先算 `P_all` 和 `P_train`**
2. **根据精度和优化器，选每个trainable param多少bytes**
3. **用 `micro_batch_size` 和 `seq_len` 估 activations**
4. **最后乘 1.1 到 1.2，留安全余量**

可以把它压缩成三条速记公式：

```text
Pre-training / 全量SFT:
VRAM ≈ P_all × bytes_trainable + cLBTHb + overhead

LoRA SFT:
VRAM ≈ P_all × bytes_frozen + P_lora × bytes_trainable + cLBTHb + overhead

QLoRA SFT:
VRAM ≈ P_all × bytes_4bit_frozen + P_lora × bytes_trainable + cLBTHb + overhead
```

其中 `cLBTHb` 只是：

```text
c × num_layers × micro_batch_size × seq_len × hidden_size × activation_bytes
```

最后给一个非常实用的判断：

- **想省模型状态显存**：优先 LoRA / QLoRA / FSDP / ZeRO
- **想省激活显存**：优先降 `micro_batch_size`、降 `seq_len`、开 activation checkpointing、开 FlashAttention
- **想提升吞吐但不增加单卡峰值显存**：优先用 gradient accumulation 拉 global batch

到这里，你应该已经能自己估算：

- 一个 Pre-training 配置能不能在 8×24GB 上跑
- 一个 7B 全量SFT为什么上不去
- 一个 LoRA/QLoRA 任务为什么明明"只训练几百万参数"，还是会因为长上下文而OOM

这就是 LLM 显存规划最核心的思维方式：**先分清是谁在占显存，再看它是按参数量增长，还是按序列长度和batch增长。**

---

## 附录：从论文定理到代码的速查表

| 论文概念 | 对应公式 | LLM训练中的体现 |
|---|---|---|
| Machine epsilon (ε) | (β/2)β^(-p) | 决定了每种格式能表达的最小相对变化 |
| Theorem 2 (guard digit) | 减法相对误差 ≤ 2ε | GPU Tensor Core中FP32累加器 |
| 灾难性抵消 (Sec 1.4) | 接近量相减误差爆炸 | LayerNorm方差计算、Softmax溢出 |
| Theorem 8 (Kahan求和) | 误差从nε降到2ε | 梯度累加的精度控制 |
| Denormalized numbers (Sec 2.2.4) | 渐进下溢 | 保证 x≠y → x-y≠0 |
| NaN传播 (Sec 2.2.1) | 任何含NaN的运算结果都是NaN | 训练"爆炸"的传播机制 |
| Overflow (Sec 2.2.2) | 超出范围→Inf | FP16训练中的梯度爆炸 |
| 论文eq.(31) 求和误差 | 误差 ≈ nε·Σ|x_j| | 大batch梯度累加需要高精度 |
| 显存预算 | 参数/梯度/优化器/激活分解 | 训练前判断单卡/多卡是否会OOM |
